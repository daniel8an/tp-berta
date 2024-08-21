import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()  # Use sigmoid to produce gate values in [0, 1]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        gates = self.activation(self.fc2(x))
        return gates


class TPBertaWithGates(nn.Module):
    def __init__(self, original_tpberta, gate_hidden_dim, apply_gates_to='input'):
        super(TPBertaWithGates, self).__init__()
        self.tpberta = original_tpberta
        self.apply_gates_to = apply_gates_to

        # Initialize the gating network based on where we want to apply the gates
        if self.apply_gates_to == 'input':
            input_dim = original_tpberta.config.hidden_size
            self.gating_network = GatingNetwork(input_dim, gate_hidden_dim, input_dim)
        elif self.apply_gates_to == 'hidden':
            input_dim = original_tpberta.config.hidden_size
            self.gating_network = GatingNetwork(input_dim, gate_hidden_dim, input_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        # Apply gating to input embeddings
        if self.apply_gates_to == 'input':
            embeddings = self.tpberta.embeddings(input_ids)
            gates = self.gating_network(embeddings)
            gated_embeddings = embeddings * gates
            output = self.tpberta.encoder(gated_embeddings, attention_mask, head_mask)

        # Apply gating to hidden layers
        elif self.apply_gates_to == 'hidden':
            output = self.tpberta(input_ids, attention_mask, token_type_ids, position_ids, head_mask)
            hidden_states = output.last_hidden_state
            gates = self.gating_network(hidden_states)
            gated_hidden_states = hidden_states * gates
            output.last_hidden_state = gated_hidden_states

        return output
