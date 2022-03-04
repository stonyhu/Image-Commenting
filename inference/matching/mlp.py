import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, input_size // 2)
        self.linear_2 = nn.Linear(input_size // 2, 50)
        self.linear_3 = nn.Linear(50, output_size)

    def forward(self, x):
        y = F.relu(self.linear_1(x))
        y = F.relu(self.linear_2(y))
        y = self.linear_3(y)
        return y
