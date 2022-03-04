import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.linear_h = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_z = nn.Linear(hidden_size, 1)
        self.linear_s = nn.Linear(hidden_size, hidden_size)
        self.linear_beta = nn.Linear(hidden_size, 1)
        self.dropout = dropout

    def forward(self, v, h, s):
        """
        :param v: torch.Tensor [bsz x 49 x 512]
        :param h: torch.Tensor [bsz x 512]
        :param s: torch.Tensor [bsz x 512]
        :return:
        """
        h = self.linear_h(h)  # [bsz x 512]
        v = F.dropout(v, self.dropout, self.training)
        v_proj = self.linear_v(v)  # [bsz x 49 x 512]
        z = F.tanh(h.unsqueeze(1).expand_as(v_proj) + v_proj)  # [bsz x 49 x 512]
        z = F.dropout(z, self.dropout, self.training)
        z = self.linear_z(z).squeeze(2)  # [bsz x 49]
        beta = F.tanh((self.linear_s(s) + h) * math.sqrt(0.5))  # [bsz x 512]
        beta = F.dropout(beta, self.dropout, self.training)
        beta = self.linear_beta(beta)  # [bsz x 1]
        a = F.softmax(torch.cat([z, beta], dim=1), dim=1)  # [bsz x 50]
        v = torch.cat([v, s.unsqueeze(1)], dim=1)  # [bsz x 50 x 512]
        c = (a.unsqueeze(2) * v).sum(dim=1)  # [bsz x 512]
        return c # TODO: return attention and sentinel gate