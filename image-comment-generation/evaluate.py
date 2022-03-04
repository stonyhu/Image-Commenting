import os
import torch
import argparse
from functools import partial
from model import Model
from utils import str2bool, draw_caption, save_image
from utils.data import load_data
from utils.metrics import evaluate
from utils.sampler import sampler
from model.generator import greedy, beam_search


parser = argparse.ArgumentParser('generate image captions')
parser.add_argument('-d', '--data-dir', metavar='PATH', default='data/')
parser.add_argument('--split', default='valid', choices=['valid', 'test'])
parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                              const=True, default=torch.cuda.is_available(),
                              help='whether to use GPU acceleration.')
parser.add_argument('--model-file', type=str, metavar='PATH', default='output/best.pt')
parser.add_argument('-bs', '--batch-size', type=int, metavar='N', default=256)
parser.add_argument('--max-len', type=int, metavar='N', default=18)
parser.add_argument('--beam-size', type=int, metavar='N', default=3)
parser.add_argument('--sample-size', type=int, metavar='N', default=50)
parser.add_argument('--sample-dir', metavar='PATH', default='samples')

args = parser.parse_args()
print(vars(args))

valid, _ = load_data(args.data_dir, args.split, args.batch_size)
model, _ = Model.load(args, args.model_file)

if args.beam_size == 1:
    generator = greedy
else:
    generator = partial(beam_search, max_len=args.max_len, beam_size=args.beam_size,
                        len_penalty=0., dup_penalty=0., no_unk=True, mono_penalty=0., min_len=1,)

# calculate scores
metrics, outputs = evaluate(model, valid, generator)
for name, (score, _) in metrics.items():
    print(name, score)

samples = sampler(valid.dataset, args.sample_size, outputs, metrics)

for label, samples in samples.items():
    sample_dir = os.path.join(args.sample_dir, label)
    os.makedirs(sample_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        image = draw_caption(sample['image'], sample['gen'], sample['ref'], sample['score'], sample['conf'])
        save_image(image, os.path.join(sample_dir, f'{i}.png'))
