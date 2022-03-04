import os
import random
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Vocab
from .network import MatchingNetwork
import torch.nn.parallel


class Model:
    def __init__(self, args, state_dict=None):
        self.args = args
        self.vocab = Vocab.load(os.path.join(args.data_dir, 'vocab_matching_model.txt'))
        self.pretrained_embeddings = self.vocab.load_embeddings(args.embeddings_file, padding=True)
        # self.pretrained_embeddings = None
        # model
        self.model = MatchingNetwork(len(self.vocab), args.embed_size, args.hidden_size, args.ft_start_layer,
                                     args.dropout, pretrained_embeddings=self.pretrained_embeddings)
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.cnn_params = list(filter(lambda x: x.requires_grad, self.model.module.cnn.parameters()))
        # model_params = itertools.chain(self.model.module.encoder.parameters(),
        #                                self.model.module.mlp.parameters())
        self.lstm_params = list(filter(lambda x: x.requires_grad, self.model.module.encoder.parameters()))
        mlp_params = itertools.chain(self.model.module.linear_i.parameters(),
                                     self.model.module.linear_t.parameters(),
                                     self.model.module.mlp.parameters())
        self.mlp_params = list(filter(lambda x: x.requires_grad, mlp_params))

        # self.model_params = list(filter(lambda x: x.requires_grad, model_params))
        if args.optimizer == 'adam':
            self.opt = torch.optim.Adam([
                # {'params': self.model_params, 'lr': args.lr_model},
                {'params': self.mlp_params, 'lr': args.lr_model},
                {'params': self.lstm_params, 'lr': args.lr_model},
                {'params': self.cnn_params, 'lr': args.lr_cnn},
            ], betas=(args.beta1, args.beta2))
        elif args.optimizer == 'sgd':
            self.opt = torch.optim.SGD([
                # {'params': self.model_params, 'lr': args.lr_model},
                {'params': self.mlp_params, 'lr': args.lr_model},
                {'params': self.lstm_params, 'lr': args.lr_model},
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

    def update(self, inputs, captions, target):
        self.model.train()
        self.opt.zero_grad()
        captions = self.vectorize_text(captions)
        inputs = inputs.to(self.device)
        captions = captions.to(self.device)
        target = target.to(self.device).squeeze(1)
        output = self.model(inputs, captions)
        loss = self.criterion(output, target)
        loss.backward()
        cnn_grad_norm = torch.nn.utils.clip_grad_norm_(self.cnn_params, self.args.grad_clipping)
        lstm_grad_norm = torch.nn.utils.clip_grad_norm_(self.lstm_params, self.args.grad_clipping)
        mlp_grad_norm = torch.nn.utils.clip_grad_norm_(self.mlp_params, self.args.grad_clipping)

        self.opt.step()
        self.updates += 1
        stats = {
            'updates': self.updates,
            'loss': loss.item(),
            'lstm_gnorm': lstm_grad_norm,
            'mlp_gnorm': mlp_grad_norm,
            'cnn_gnorm': cnn_grad_norm,
        }
        return stats

    def validate(self, data):
        self.model.eval()
        total_loss = 0
        corrects = 0.
        sample_size = 0
        for inputs, captions, target in tqdm(data, desc='validating'):
            captions = self.vectorize_text(captions, sample=False)
            inputs = inputs.to(self.device)
            captions = captions.to(self.device)
            target = target.to(self.device).squeeze(1)
            with torch.no_grad():
                output = self.model(inputs, captions)
                loss = self.criterion(output, target)
                _, pred = torch.max(F.softmax(output, dim=1), dim=1)
                corrects += torch.sum(pred == target).item()
                sample_size += inputs.size(0)
            total_loss += loss.item()
        loss = total_loss / len(data)
        accuracy = corrects / sample_size
        stats = {
            'loss': loss,
            'accuracy': accuracy,
        }
        return stats

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
