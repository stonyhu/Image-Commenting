import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import CNN
from .mlp import MLP
from .encoder import CNNTextEncoder, BiLSTMEncoder


class MatchingNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, ft_start_layer, dropout,
                 pretrained_embeddings=None, num_filters=100, window_sizes=(3, 4, 5)):
        super().__init__()
        self.cnn = CNN(ft_start_layer)
        # self.linear_w1 = nn.Linear(2048, hidden_size)
        # self.linear = nn.Linear(2048, num_filters * len(window_sizes))
        self.linear_i = nn.Linear(2048, 300)
        self.linear_t = nn.Linear(hidden_size * 2, 300)
        # self.encoder = CNNTextEncoder(vocab_size, embed_size,
        #                               num_filters=num_filters,
        #                               window_sizes=window_sizes,
        #                               pretrained_embeddings=pretrained_embeddings)
        self.encoder = BiLSTMEncoder(vocab_size, embed_size, hidden_size,
                                     pretrained_embeddings=pretrained_embeddings)
        self.mlp = MLP(num_filters * len(window_sizes), 2)
        # self.mlp = MLP(hidden_size * 2 + 2048, 2)  # for BiLSTM encoder
        self.dropout = dropout

    def tune_cnn(self, flag):
        self.cnn.tuning(flag)

    def forward(self, image, caption):
        v = self.cnn(image)
        # v = v.view(v.size(0), -1)  # [bsz x 2048]
        # v = F.relu(self.linear_w1(v))
        # v = self.linear_w2(v)  # [bsz x (F x W)]
        c = self.encoder(caption)  # [bsz x (F x W)]
        # s = torch.cat([v, c], dim=1)
        s = self.linear_i(v) * self.linear_t(c)  # [bsz x (F x W)]
        s = F.dropout(s, self.dropout, self.training)
        return self.mlp(s)

