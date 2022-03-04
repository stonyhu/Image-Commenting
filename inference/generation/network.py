import torch
import torch.nn as nn
from .modules import Encoder, Decoder, CNN


class Adaptive(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout, ft_start_layer,
                 pretrained_embeddings=None):
        super().__init__()
        self.cnn = CNN(ft_start_layer)
        self.encoder = Encoder(hidden_size, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, dropout,
                               pretrained_embeddings=pretrained_embeddings)

    def tune_cnn(self, flag):
        self.cnn.tuning(flag)

    def forward(self, image, target):
        enc_out = self.encoder(self.cnn(image))
        bsz = image.size(0)
        x0, state = self.decoder.get_initials(bsz)
        prev_outs = torch.cat([x0, target[:, :-1]], dim=1)
        outputs = []
        for prev_out in torch.split(prev_outs, 1, dim=1):
            logits, state = self.decoder(prev_out, state, enc_out)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)  # [bsz x seq_len x |V|]

    def generate(self, image, generator):
        bsz = image.size(0)
        enc_out = self.encoder(self.cnn(image))
        x0, state = self.decoder.get_initials(bsz)
        return generator(self.decoder, x0, state, enc_out)
