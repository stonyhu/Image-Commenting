import os
import sys
import time
import torch
import random
import argparse
from functools import partial
from model import Model
from model.generator import greedy, beam_search
from utils import str2bool
from utils.data import load_data
from utils.sampler import sampler
from utils.metrics import evaluate
from utils.logger import Logger
import torch.distributed as dist
import torch.nn.parallel


def main():
    command = sys.argv[1:]
    trainer = Trainer(command)
    trainer.train()


class Trainer(object):
    def __init__(self, command):
        parser = self._initialize_parser()
        self.args = parser.parse_args(command)
        self.log = Logger(self.args)

        self.args.distributed = self.args.world_size > 1
        if self.args.distributed:
            dist.init_process_group(self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size,
                                    rank=self.args.dist_rank)

    @staticmethod
    def _initialize_parser():
        parser = argparse.ArgumentParser('Training Knowing When to Look')
        routine = parser.add_argument_group('routine')
        routine.add_argument('-d', '--data-dir', metavar='PATH', default='data/')
        routine.add_argument('-o', '--output-dir', metavar='PATH', default='output/')
        routine.add_argument('-emb', '--embeddings-file', type=str,
                             default='data/twitter-vectors.512d.txt',
                             help='the pre-trained word embeddings')
        routine.add_argument('--log-per-updates', type=int, metavar='N', default=1,
                             help='log model loss per x updates (mini-batches).')
        routine.add_argument('--sample-size', metavar='N', type=int, default=50)
        routine.add_argument('--seed', type=int, default=123,
                             help='random seed for data shuffling, dropout, etc.')
        routine.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                             const=True, default=torch.cuda.is_available(),
                             help='whether to use GPU acceleration.')
        routine.add_argument('--tensorboard', action='store_true')

        training = parser.add_argument_group('training')
        training.add_argument('-e', '--epochs', type=int, metavar='N', default=50)
        training.add_argument('-bs', '--batch-size', type=int, metavar='N', default=80)
        training.add_argument('-rs', '--resume', type=str, metavar='FILE', default='last.pt')
        training.add_argument('-gc', '--grad-clipping', metavar='x', type=float, default=10)
        training.add_argument('-wd', '--weight-decay', metavar='x', type=float, default=0)
        training.add_argument('-opt', '--optimizer', type=str, default='adam', help='model optimizer.')
        training.add_argument('--lr-model', metavar='x', type=float, default=5e-4)
        training.add_argument('--lr-cnn', metavar='x', type=float, default=1e-5)
        training.add_argument('--momentum', type=float, default=0.9, help='momentum')

        training.add_argument('--beta1', metavar='x', type=float, default=0.8)
        training.add_argument('--beta2', metavar='x', type=float, default=0.999)
        training.add_argument('--label-smoothing', default=0., type=float, metavar='x',
                              help='epsilon for label smoothing, 0 means no label smoothing')
        training.add_argument('--start-cnn-tuning', metavar='N', type=int, default=20,
                              help='start finetuning CNN after N epochs.')
        training.add_argument('--ft-start-layer', metavar='N', type=int, default=6,
                              help='start finetuning CNN after N layers.')
        training.add_argument('-es', '--early-stopping', type=int, metavar='N', default=6,
                              help='how many stalling epochs are allowed; use an extremely large number to turn it off.')
        training.add_argument('--world-size', type=int, default=1, help='number of distributed processes.')
        training.add_argument('--dist-url', type=str, default='tcp://10.172.136.60:23456',
                              help='url used to set up distribued training.')
        training.add_argument('--dist-backend', type=str, default='gloo', help='distributed backend.')
        training.add_argument('--dist-rank', type=int, default=0, help='rank of distributed process.')
        training.add_argument('--sync_steps', type=int, default=500, help='steps interval of params sync.')

        model = parser.add_argument_group('model')
        model.add_argument('--hidden-size', type=int, metavar='D', default=512)
        model.add_argument('--embed-size', type=int, metavar='D', default=512)
        model.add_argument('--beam-size', type=int, metavar='N', default=3)
        model.add_argument('--dropout', type=float, metavar='x', default=0.2)
        return parser

    def train(self):
        self.log(f'[{str(vars(self.args))}]')
        self.log(f'gpu num: {torch.cuda.device_count()}')

        model, states = self.build_model()
        start_epoch, best_eval, best_epoch = states

        train, train_sampler = load_data(self.args.data_dir, 'train', self.args.batch_size, self.args.distributed)
        self.log(f'batch_sampler len: {len(train)}')
        valid, _ = load_data(self.args.data_dir, 'valid', self.args.batch_size)


        for epoch in range(start_epoch, self.args.epochs + 1):
            if self.args.distributed:
                train_sampler.set_epoch(epoch)
            model.tune_cnn(epoch > self.args.start_cnn_tuning)
            model.schedule_lr(epoch)
            self.log.set_progress(epoch, len(train))

            for images, captions in train:
                stats = model.update(images, captions)

                if self.args.distributed and stats['updates'] % self.args.sync_steps == 0:
                    self.log(f'Begin to sync params after {stats["updates"]} updates..')
                    for para in model.cnn_params + model.model_params:
                        dist.all_reduce(para.grad.data, op=dist.reduce_op.SUM)
                        para.grad.data /= float(dist.get_world_size())
                    self.log(f'finish.')

                self.log.update(stats)

            metrics_group = {}
            stats = model.validate(valid)
            metrics, _ = evaluate(model, valid, partial(beam_search, beam_size=10))
            metrics_group['beam10'] = metrics
            metrics, _ = evaluate(model, valid, greedy)
            metrics_group['greedy'] = metrics
            metrics, outputs = evaluate(model, valid, partial(beam_search, beam_size=3))
            metrics_group['beam3'] = metrics
            self.log.log_eval(stats, metrics_group)
            samples = sampler(valid.dataset, self.args.sample_size, outputs, metrics)
            self.log.log_samples(samples)

            score = metrics['CIDEr'][0]
            if score > best_eval:
                best_eval = score
                best_epoch = epoch
                self.log('[New best found.]')
                model.save((epoch, best_eval, best_epoch), name='best')
            model.save((epoch, best_eval, best_epoch))
            model.save((epoch, best_eval, best_epoch), name='last')
            if epoch - best_epoch > self.args.early_stopping:
                self.log('[Tolerance reached. Training is stopped early.]')
                break

    def build_model(self):
        if self.args.resume == 'last.pt' and not os.path.exists(os.path.join(self.args.output_dir, self.args.resume)):
            self.args.resume = None

        if not self.args.resume:
            self.log('[start training from scratch.]')
            model = Model(self.args)
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.cuda:
                torch.cuda.manual_seed(self.args.seed)
            start_epoch = 1
            best_eval = 0.
            best_epoch = 0

            if self.args.distributed:
                self.log('[average initial params.]')
                for para in model.model_params:
                    dist.all_reduce(para.data, op=dist.reduce_op.SUM)
                    para.data /= self.args.world_size
        else:
            self.log('[loading previous model...]')
            model, checkpoint = Model.load(self.args, os.path.join(self.args.output_dir, self.args.resume))
            start_epoch = checkpoint['epoch'] + 1
            best_eval = checkpoint['best_eval']
            best_epoch = checkpoint['best_epoch']
            random.setstate(checkpoint['random_state'])
            torch.random.set_rng_state(checkpoint['torch_state'])
            if self.args.cuda and 'torch_cuda_state' in checkpoint:
                torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
        return model, (start_epoch, best_eval, best_epoch)


if __name__ == '__main__':
    main()

    # distributed training.
    # python train.py -d data -o output -bs 32 --start-cnn-tuning 0 --ft-start-layer -17 --tensorboard
