import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Vocab
from .attention import SelfAttention


class CNNTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 num_filters=100, window_sizes=(3, 4, 5), pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=Vocab.pad())
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (window_size, embed_size)) for window_size in window_sizes
        ])

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        ms = []
        for conv in self.convs:
            m = F.relu(conv(x))
            m = m.squeeze(-1)
            m = F.max_pool1d(m, m.size(2))
            ms.append(m)
        c = torch.cat(ms, dim=2)
        return c.view(c.size(0), -1)


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=Vocab.pad())
        self.bilstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.attention = SelfAttention(hidden_size * 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 0)  # [seq_len x bsz x 512]
        h, _ = self.bilstm(x)  # [seq_len x bsz x 1024]
        c = self.attention(h.transpose(1, 0))  # [bsz x 1024]
        return c

