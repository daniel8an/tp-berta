from typing import Optional

import math
# from lib.util import get_device

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class TPBertaOutputWithGating(BaseModelOutputWithPoolingAndCrossAttentions):
    def __init__(self, *args, gating_loss=None, stochastic_gate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gating_loss = gating_loss
        self.stochastic_gate = stochastic_gate

class GatingNetwork(nn.Module):
    def __init__(self, max_sequence_length, hidden_dim):
        super(GatingNetwork, self).__init__()
        self.max_sequence_length = max_sequence_length
        
        # Input normalization
        self.input_norm = nn.LayerNorm(max_sequence_length)
        
        # Attention directly on input
        self.attention = nn.MultiheadAttention(embed_dim=max_sequence_length, num_heads=4)
        
        # First block
        self.fc2 = nn.Linear(max_sequence_length, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Second block with residual
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
        # Third block with bottleneck
        self.fc6 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm5 = nn.LayerNorm(hidden_dim // 2)
        self.fc7 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.norm6 = nn.LayerNorm(hidden_dim // 4)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim // 4, max_sequence_length)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Activation functions
        self.gelu = nn.GELU()
        
        # Initialize weights with small values for better gradient flow
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
                    
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x.float())
        
        # Apply attention directly to input
        x = x.unsqueeze(0)  # Add batch dimension for attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)  # Remove batch dimension
        
        # First block
        identity1 = self.fc2(x)  # Project after attention
        x = self.dropout(self.gelu(self.norm1(identity1)))
        x = x + identity1  # First residual connection
        
        # Second block with residual
        identity2 = x
        x = self.dropout(self.gelu(self.norm3(self.fc4(x))))
        x = self.dropout(self.gelu(self.norm4(self.fc5(x))))
        x = x + identity2  # Second residual connection
        
        # Third block with bottleneck
        x = self.dropout(self.gelu(self.norm5(self.fc6(x))))
        x = self.dropout(self.gelu(self.norm6(self.fc7(x))))
        
        # Output projection
        x = self.fc_out(x)
        
        # Clamp outputs to prevent extreme values
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        return x

class TPBertaWithGates(nn.Module):
    def __init__(self, max_sequence_length, gate_hidden_dim, a=1, sigma=0.1, pad_token_id=0,
                 lambda1=0.001, lambda2=0.001, init_temperature=1.0, min_temperature=0.1, 
                 temperature_decay=0.99, min_keep_ratio=0.3, entropy_weight=0.01):
        super(TPBertaWithGates, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.gating_network = GatingNetwork(max_sequence_length, gate_hidden_dim)
        self.a = a
        self.sigma = sigma
        self.pad_token_id = pad_token_id
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.temperature = init_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.min_keep_ratio = min_keep_ratio
        self.entropy_weight = entropy_weight
        
        # Initialize gates with positive bias
        for m in self.gating_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)
        
        self.register_buffer('gate_activations', torch.zeros(1))
        self.register_buffer('total_gates', torch.zeros(1))
        self.register_buffer('gate_ema', torch.zeros(max_sequence_length))
        self.register_buffer('ema_updates', torch.zeros(1))
        self.ema_decay = 0.99
        
    def compute_entropy_loss(self, probs):
        """Compute entropy regularization loss"""
        eps = 1e-7
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1).mean()
        return -self.entropy_weight * entropy  # Negative because we want to maximize entropy
        
    def forward(self, input_ids: torch.Tensor, train_gates: bool = True):
        original_size = input_ids.size(1)
        input_ids_padded = F.pad(
            input_ids,
            (0, self.max_sequence_length - input_ids.shape[1]),
            value=self.pad_token_id
        )
        
        # Get gating logits
        mu = self.gating_network(input_ids_padded)
        
        # Add noise during training
        if self.training and train_gates:
            noise = torch.randn_like(mu) * self.sigma
            mu = mu + noise
            
            # Update temperature with annealing
            self.temperature = max(
                self.min_temperature,
                self.temperature * self.temperature_decay
            )
        
        # Apply temperature scaling and sigmoid
        z = torch.sigmoid(mu / self.temperature)
        
        # Compute gate probabilities with smoothing
        probs = z.clone()
        if self.training:
            probs = 0.9 * probs + 0.05  # Smoothing to avoid extreme values
        
        # Smoothness loss - encourage similar features to have similar gates
        # but only when they are correlated
        z_diff = torch.abs(z[:, 1:] - z[:, :-1])  # Shape: [batch_size, seq_len-1]
        feature_correlations = torch.corrcoef(z.T)  # Shape: [seq_len, seq_len]
        
        # Fix the indexing for correlation mask
        correlation_mask = (feature_correlations > 0.5).float()  # Shape: [seq_len, seq_len]
        correlation_mask = correlation_mask[:-1, 1:]  # Shape: [seq_len-1, seq_len-1]
        # Take diagonal elements for pair-wise correlations
        correlation_mask = torch.diagonal(correlation_mask)  # Shape: [seq_len-1]
        
        R2_loss = self.lambda2 * (z_diff * correlation_mask[None, :]).mean()
        
        # Update EMA of gate values during training
        if self.training and train_gates:
            with torch.no_grad():
                batch_mean_gates = z[:, :original_size].mean(dim=0)
                padded_mean_gates = F.pad(
                    batch_mean_gates,
                    (0, self.max_sequence_length - original_size),
                    value=0.0
                )
                self.gate_ema = self.ema_decay * self.gate_ema + (1 - self.ema_decay) * padded_mean_gates
                self.ema_updates += 1
        
        # Dynamic minimum keep ratio based on training progress
        current_keep_ratio = max(
            self.min_keep_ratio,
            self.min_keep_ratio + (1 - self.min_keep_ratio) * (1 - self.ema_updates / 10000).clamp(0, 1)
        )
        
        # Ensure minimum tokens are kept
        min_tokens = max(1, int(current_keep_ratio * original_size))
        mask = (z < 0.5) & (input_ids_padded != self.pad_token_id)
        
        for i in range(len(mask)):
            if mask[i].sum() > original_size - min_tokens:
                _, indices = torch.topk(z[i, :original_size], k=min_tokens)
                mask[i, indices] = False
        
        # Keep special tokens
        mask[:, 0] = False  # CLS token
        if input_ids_padded[:, -1].eq(self.pad_token_id).any():
            mask[:, -1] = False  # SEP token
        
        gated_input_ids = input_ids_padded.clone()
        gated_input_ids[mask] = self.pad_token_id
        
        # Update statistics
        with torch.no_grad():
            self.gate_activations += (z > 0.5).float().sum()
            self.total_gates += z.numel()
        
        # Remove fixed sparsity target and modify loss computation
        # Enhanced loss computation with adaptive regularization
        
        # Sparsity loss - use entropy-based adaptive regularization
        gate_probs = torch.sigmoid(mu)
        entropy = -(gate_probs * torch.log(gate_probs + 1e-7) + (1 - gate_probs) * torch.log(1 - gate_probs + 1e-7))
        entropy_per_feature = entropy.mean(dim=0)  # Average entropy across batch
        
        # Encourage decisive gating (either fully on or off) while allowing flexibility
        R1_loss = -self.lambda1 * (entropy_per_feature.mean())  # Minimize entropy to encourage decisive gating
        
        # Add consistency loss across batches using EMA
        if self.training and train_gates:
            with torch.no_grad():
                current_mean = z.mean(dim=0)
                if not hasattr(self, 'feature_ema'):
                    self.register_buffer('feature_ema', current_mean.clone())
                else:
                    self.feature_ema = 0.9 * self.feature_ema + 0.1 * current_mean
            
        consistency_loss = 0.001 * torch.abs(z.mean(dim=0) - self.feature_ema).mean() if hasattr(self, 'feature_ema') else 0
        
        # Total gating loss
        gating_loss = R1_loss + R2_loss + consistency_loss
        
        # Add warmup factor for gating loss
        warmup_factor = min(1.0, self.ema_updates / 1000)
        gating_loss = gating_loss * warmup_factor

        return gated_input_ids[:, :original_size], gating_loss, z[:, :original_size]

    def get_sparsity_stats(self):
        if self.total_gates == 0:
            return 0.0
        sparsity = 1.0 - (self.gate_activations / self.total_gates).item()
        self.gate_activations.zero_()
        self.total_gates.zero_()
        return sparsity

    def get_gate_statistics(self):
        """Get statistics about gate usage"""
        if self.ema_updates == 0:
            return {
                'mean_gate': 0.0,
                'gate_std': 0.0,
                'active_gates': 0.0
            }
        
        gate_mean = self.gate_ema.mean().item()
        gate_std = self.gate_ema.std().item()
        active_ratio = (self.gate_ema > 0.5).float().mean().item()
        
        return {
            'mean_gate': gate_mean,
            'gate_std': gate_std,
            'active_gates': active_ratio
        }

# The rest of the code remains unchanged
