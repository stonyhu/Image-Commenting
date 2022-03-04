import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.linear_patch = nn.Linear(2048, hidden_size)
        self.linear_avg = nn.Linear(2048, hidden_size)
        self.dropout = dropout

    def forward(self, v):
        """
        :param v: torch.Tensor [bsz x 2048 x 7 x 7]
        :return:
            v: torch.Tensor [bsz x 49 x 512]
            v_avg: torch.Tensor [bsz x 512]
        """
        v = v.view(-1, 2048, 49).transpose(1, 2)  # [bsz x 49 x 2048]
        v_avg = torch.mean(v, dim=1)  # [bsz x 2048]
        v = self.linear_patch(v)  # [bsz x 49 x 512]
        v = F.relu(v, inplace=not self.training)  # [bsz x 49 x 512]
        v_avg = self.linear_avg(v_avg)  # [bsz x 512]
        v_avg = F.relu(v_avg, inplace=not self.training)  # [bsz x 512]
        return v, v_avg
