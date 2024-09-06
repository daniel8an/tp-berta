from typing import Optional

# from lib.util import get_device

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, input_output_dim, hidden_dim):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_output_dim)
        self.activation = nn.Sigmoid()  # Use sigmoid to produce gate values in [0, 1]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        gates = self.activation(self.fc2(x))
        return gates


class TPBertaWithGates(nn.Module):
    def __init__(self, input_dim, gate_hidden_dim, apply_to="input"):
        super(TPBertaWithGates, self).__init__()
        self.input_dim = input_dim
        self.gating_network = GatingNetwork(self.input_dim, gate_hidden_dim)
        self.apply_to = apply_to

    def forward(
        self,
        input_i: Optional[torch.Tensor] = None,
    ):
        if self.apply_to == "input":
            one_hot_input = F.one_hot(
                input_i, num_classes=self.input_dim
            ).float()
            gates = self.gating_network(one_hot_input)
            gated_input = (one_hot_input * gates).argmax(dim=-1)
        elif self.apply_to == "hidden":
            gates = self.gating_network(input_i)
            gated_input = (input_i * gates)

        return gated_input

# class GatingNetwork(nn.Module):
#     def __init__(self, input_output_dim, hidden_dim):
#         super(GatingNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_output_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, input_output_dim)
#         self.activation = nn.Sigmoid()  # Use sigmoid to produce gate values in [0, 1]
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         gates = self.activation(self.fc2(x))
#         return gates
#
#
# class TPBertaWithGates(nn.Module):
#     def __init__(self, original_tpberta, gate_hidden_dim, apply_gates_to="hidden"):
#         super(TPBertaWithGates, self).__init__()
#         self.tpberta = original_tpberta
#         self.apply_gates_to = apply_gates_to
#         self.input_dim = (
#             self.tpberta.config.vocab_size
#             if apply_gates_to == "input"
#             else original_tpberta.config.hidden_size
#         )
#         self.gating_network = GatingNetwork(self.input_dim, gate_hidden_dim)
#         self.device = original_tpberta.device
#
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         input_scales: Optional[torch.Tensor] = None,
#         feature_cls_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         tail_prompt: Optional[torch.Tensor] = None,
#     ):
#
#         if self.apply_gates_to == "input":
#             one_hot_input = F.one_hot(
#                 input_ids, num_classes=self.input_dim
#             ).float()
#             gates = self.gating_network(one_hot_input)
#             gated_input_ids = (one_hot_input * gates).argmax(dim=-1)
#
#             output = self.tpberta(
#                 gated_input_ids,
#                 input_scales=input_scales,
#                 feature_cls_mask=feature_cls_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#
#         elif self.apply_gates_to == "hidden":
#             logits, tpberta_output = self.tpberta(
#                 input_ids,
#                 input_scales=input_scales,
#                 feature_cls_mask=feature_cls_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#             gates = self.gating_network(tpberta_output.last_hidden_state)
#             gated_hidden_state = tpberta_output.last_hidden_state * gates
#             tpberta_output.last_hidden_state = gated_hidden_state
#             logits = self.tpberta.classifier(gated_hidden_state).squeeze(-1)
#             output = logits, tpberta_output
#
#         return output
