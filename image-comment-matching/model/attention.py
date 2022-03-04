import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_w = nn.Linear(input_size, input_size, bias=False)
        self.linear_s = nn.Linear(input_size, 1, bias=False)

    def forward(self, h):
        """
        :param h: torch.Tensor [bsz x seq_len x input_size]
        :return c: self-attentive context vector # [bsz x input_size]
        """
        m = F.tanh(self.linear_w(h))
        a = F.softmax(self.linear_s(m).squeeze(2), dim=1)
        c = (a.unsqueeze(2) * h).sum(dim=1)
        return c
