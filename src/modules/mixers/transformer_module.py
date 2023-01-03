import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class TransformerModule(nn.Module):

    def __init__(self, input_dim: int, ntoken: int, d_model: int = 64, nhead: int = 4, d_hid: int = 64):
        super().__init__()
        self.model_type = 'Transformer'
        self.nhead = nhead
        self.Ks = nn.ModuleList()
        self.Qs = nn.ModuleList()
        self.Vs = nn.ModuleList()
        for i in range(self.nhead):  # multi-head attention
            self.Ks.append(nn.Sequential(nn.Linear(input_dim, d_model),
                                                    nn.ReLU(),
                                                    nn.Linear(d_model, d_model)))  # K
            self.Qs.append(nn.Sequential(nn.Linear(input_dim, d_model),
                                                    nn.ReLU(),
                                                    nn.Linear(d_model, d_model)))  # Q
            self.Vs.append(nn.Sequential(nn.Linear(input_dim, d_model),
                                                    nn.ReLU(),
                                                    nn.Linear(d_model, d_model)))  # V
        self.layer_norm = nn.LayerNorm(input_dim)
        self.output_layer_norm = nn.LayerNorm(int(d_model*nhead))
        self.d_model = d_model
        self.decoder = nn.Sequential(nn.Linear(d_model, d_hid),
                                                    nn.ReLU(),
                                                    nn.Linear(d_hid, ntoken))
        self.output = nn.Sequential(nn.Linear(d_model, d_model),
                                                    nn.ReLU(),
                                    nn.Linear(d_model, ntoken),
                                                    nn.ReLU())    

    def forward(self, src: Tensor) -> Tensor:
        seq_length = src.size(1)
        batch_size = src.size(0)
        src = self.layer_norm(src)
        all_Ks = [K(src) for K in self.Ks]
        all_Qs = [Q(src) for Q in self.Qs]
        all_Vs = [V(src) for V in self.Vs]
        attention_values = []
        for K, Q, V in zip(all_Ks, all_Qs, all_Vs):
            norm_QK = torch.bmm(Q, K.transpose(1,2))/math.sqrt(self.d_model)
            softmax_QK = F.softmax(norm_QK, dim=-1)
            attention_value = torch.bmm(softmax_QK, V)
            attention_values.append(attention_value) # [batch_size, seq_len, ntoken]
        transformer_output = torch.stack(attention_values).sum(0).view(batch_size, seq_length, -1)
        transformer_output = self.decoder(transformer_output).mean(1)
        return self.output(transformer_output)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

