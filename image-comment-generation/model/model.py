import os
import math
import random
import itertools
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data import Vocab
from .network import Adaptive
import torch.nn.parallel


class Model:
    def __init__(self, args, state_dict=None):
        self.args = args
        self.vocab = Vocab.load(os.path.join(args.data_dir, 'vocab.txt'))
        self.pretrained_embeddings = self.vocab.load_embeddings(args.embeddings_file)
        # model
        self.model = Adaptive(len(self.vocab), args.embed_size, args.hidden_size, args.dropout,
                              args.ft_start_layer, pretrained_embeddings=self.pretrained_embeddings)
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        # optimizer
        self.cnn_params = list(filter(lambda x: x.requires_grad, self.model.module.cnn.parameters()))
        model_params = itertools.chain(self.model.module.encoder.parameters(), self.model.module.decoder.parameters())

        self.model_params = list(filter(lambda x: x.requires_grad, model_params))
        if args.optimizer == 'adam':
            self.opt = torch.optim.Adam([
                {'params': self.model_params, 'lr': args.lr_model},
                {'params': self.cnn_params, 'lr': args.lr_cnn},
            ], betas=(args.beta1, args.beta2))
        elif args.optimizer == 'sgd':
            self.opt = torch.optim.SGD([
                {'params': self.model_params, 'lr': args.lr_model},
                {'params': self.cnn_params, 'lr': args.lr_cnn},
            ], lr=args.lr_model, momentum=args.momentum, weight_decay=args.weight_decay)
        # updates
        self.updates = state_dict['updates'] if state_dict else 0

        if state_dict:
            new_state = set(self.model.state_dict().keys())
            for k in list(state_dict['model'].keys()):
                if k not in new_state:
                    del state_dict['model'][k]
            self.model.load_state_dict(state_dict['model'])
            self.opt.load_state_dict(state_dict['opt'])

    def tune_cnn(self, flag):
        self.model.module.tune_cnn(flag)

    def update(self, inputs, captions):
        self.model.train()
        self.opt.zero_grad()
        target = self.vectorize_text(captions)
        inputs = inputs.to(self.device)
        target = target.to(self.device)
        output = self.model(inputs, target)
        loss, nll_loss, _ = self.get_loss(output, target)
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model_params, self.args.grad_clipping)
        cnn_grad_norm = torch.nn.utils.clip_grad_norm_(self.cnn_params, self.args.grad_clipping)

        self.opt.step()
        self.updates += 1
        stats = {
            'updates': self.updates,
            'loss': loss.item(),
            'nll_loss': nll_loss.item(),
            'ppl': self.get_ppl(nll_loss.item()),
            'model_gnorm': model_grad_norm,
            'cnn_gnorm': cnn_grad_norm,
        }
        return stats

    def validate(self, data):
        self.model.eval()
        total_loss = 0
        total_nll_loss = 0
        total_tokens = 0

        for inputs, captions in data:
            target = self.vectorize_text(captions, sample=False)
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                output = self.model(inputs, target)
                loss, nll_loss, sample_size = self.get_loss(output, target, average=False)
            total_loss += loss.item()
            total_nll_loss += nll_loss.item()
            total_tokens += sample_size
        loss = total_loss / total_tokens
        nll_loss = total_nll_loss / total_tokens
        ppl = self.get_ppl(nll_loss)
        stats = {
            'loss': loss,
            'nll_loss': nll_loss,
            'ppl': ppl,
        }
        return stats

    def generate(self, inputs, generator):
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            prediction = self.model.module.generate(inputs, generator)
        return self.devectorize_text(prediction), prediction

    def get_loss(self, logits, target, average=True):
        """
        :param logits: unnormalized prediction (torch.Tensor[B x T x C])
        :param target: target labels (torch.LongTensor[B x T])
        :return: reduced loss (torch.Tensor[1])
        """
        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)
        non_pad_mask = target.ne(self.vocab.pad())
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        sample_size = nll_loss.size(0)
        agg = torch.mean if average else torch.sum
        nll_loss = agg(nll_loss)
        smooth_loss = agg(smooth_loss)
        eps = self.args.label_smoothing
        eps_i = eps / lprobs.size(-1)
        loss = (1. - eps) * nll_loss + eps_i * smooth_loss

        return loss, nll_loss, sample_size

    @staticmethod
    def get_ppl(nll_loss):
        try:
            return math.pow(2, nll_loss)
        except OverflowError:
            return float('inf')

    def schedule_lr(self, epoch):
        if 15 < epoch <= 20:
            factor = 0.5 ** ((epoch - 15) / 5)
        elif epoch > 40:
            factor = 0.5 ** ((epoch - 40) / 5)
        else:
            factor = 1.
        self.opt.param_groups[0]['lr'] = self.args.lr_model * factor
        self.opt.param_groups[1]['lr'] = self.args.lr_cnn * factor

    def vectorize_text(self, text, sample=True):
        batch_size = len(text[0])
        if sample:
            batch_text = (text[random.randrange(0, len(text))][i] for i in range(batch_size))
        else:
            batch_text = text[0]
        batch = [[self.vocab.index(word) for word in sent.split()] + [self.vocab.eos()] for sent in batch_text]
        input_len = max(len(x) for x in batch)
        text_vec = torch.LongTensor(batch_size, input_len).fill_(self.vocab.pad())
        for i, doc in enumerate(batch):
            text_vec[i, :len(doc)] = torch.LongTensor(doc)
        return text_vec

    def devectorize_text(self, generation):
        for line in generation:
            for beam in line:
                beam['tokens'] = [self.vocab[idx] for idx in beam['tokens']]
        return [' '.join(line[0]['tokens'][:-1]) for line in generation]

    def save(self, states, name=None):
        epoch, best_eval, best_epoch = states
        if name:
            filename = os.path.join(self.args.output_dir, f'{name}.pt')
        else:
            filename = os.path.join(self.args.output_dir, 'epoch_{}.pt'.format(epoch))
        params = {
            'state_dict': {
                'model': self.model.state_dict(),
                'opt': self.opt.state_dict(),
                'updates': self.updates,
            },
            'args': self.args,
            'epoch': epoch,
            'best_eval': best_eval,
            'best_epoch': best_epoch,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state()
        }
        if self.args.cuda:
            params['torch_cuda_state'] = torch.cuda.get_rng_state()
        torch.save(params, filename)

    @classmethod
    def load(cls, args, file):
        checkpoint = torch.load(file, map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ))
        prev_args = checkpoint['args']
        # update args
        for key in dir(args):
            if key.startswith('__'):
                continue
            setattr(prev_args, key, getattr(args, key))
        return cls(prev_args, state_dict=checkpoint['state_dict']), checkpoint
