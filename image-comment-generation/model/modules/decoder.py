import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data import Vocab
from .attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout,
                 pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=Vocab.pad())
        self.lstm = nn.LSTMCell(input_size=hidden_size + embed_size,
                                hidden_size=hidden_size)
        self.linear = nn.Linear(embed_size + 2 * hidden_size, hidden_size)
        self.attention = Attention(hidden_size, dropout)
        self.linear_out = nn.Linear(hidden_size, vocab_size)
        self.sos_symbol = nn.Parameter(torch.LongTensor([Vocab.eos()]), requires_grad=False)
        self.h0 = nn.Parameter(torch.randn(1, hidden_size) * 0.1)
        self.m0 = nn.Parameter(torch.randn(1, hidden_size) * 0.1)
        self.dropout = dropout

    def forward(self, x, state, enc_out):
        """
        :param x: torch.LongTensor [bsz x 1]
        :param state: (torch.Tensor [bsz x 512], torch.Tensor [bsz x 512])
        :param enc_out: (torch.Tensor [bsz x 49 x 512], torch.Tensor [bsz x 512])
        :return:
            logits: [bsz x |V|]
            state: ([bsz x 512], [bsz x 512])
        """
        v, v_g = enc_out
        h, m = state

        x = self.embedding(x).squeeze(1)  # [bsz x 300]
        x = torch.cat([x, v_g], dim=1)  # [bsz x 812]
        x = F.dropout(x, self.dropout, self.training)
        g = torch.cat([x, h], dim=-1)  # [bsz x 1324]
        g = F.sigmoid(self.linear(g))  # [bsz x 512]
        h, m = self.lstm(x, (h, m))
        s = g * F.tanh(m)  # [bsz x 512]
        c = self.attention(v, h, s)  # [bsz x 512]
        c = (h + c) * math.sqrt(0.5)
        c = F.dropout(c, self.dropout, self.training)
        return self.linear_out(c), (h, m)

    def get_initials(self, bsz):
        x = self.sos_symbol.expand(bsz, 1)
        h = self.h0.expand(bsz, -1)
        m = self.m0.expand(bsz, -1)
        return x, (h, m)

