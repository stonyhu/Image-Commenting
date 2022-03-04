import json
import torch
import argparse
from functools import partial
from tqdm import tqdm
from model import Model
from utils import str2bool
from utils.data import load_images
from model.generator import greedy, beam_search
import time


parser = argparse.ArgumentParser('generate image captions')
parser.add_argument('-d', '--image-dir')
parser.add_argument('--mode', type=int, default=0,
                              help='whether to filter universal replies')
parser.add_argument('--cuda', type=str2bool, nargs='?',
                              const=True, default=torch.cuda.is_available(),
                              help='whether to use GPU acceleration.')
parser.add_argument('--model-file', type=str, default='output/best.pt')
parser.add_argument('-o', '--output-file', type=str, default='generation.json')
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('--max-len', type=int, default=18)
parser.add_argument('--beam-size', type=int, default=20)
args = parser.parse_args()

images = load_images(args.image_dir, args.batch_size)
print(f'{time.asctime()}-Images loaded.')
model, _ = Model.load(args, args.model_file)
print(f'{time.asctime()}-Model loaded.')

if args.beam_size == 1:
    generator = greedy
else:
    # generator = partial(beam_search, max_len=args.max_len, beam_size=args.beam_size,
    #                     len_penalty=0., dup_penalty=0., no_unk=True, mono_penalty=5., min_len=1,)
    generator = partial(beam_search, max_len=args.max_len, beam_size=args.beam_size,
                        len_penalty=0., dup_penalty=5., no_unk=True, mono_penalty=5., min_len=1, )
result = []


def load_reply(filename):
    return set([l.strip().split('\t')[0] for l in open(filename)])


def generate(images):
    for inputs, paths in tqdm(images, desc='generating'):
        captions, _ = model.generate(inputs, generator)
        for caption, path in zip(captions, paths):
            result.append({'caption': caption.replace(' ', ''),
                           'image_path': path})


def generate_beams(images):
    for inputs, paths in tqdm(images, desc='generating all beams'):
        final_generation = []
        _, generation = model.generate(inputs, generator)
        for line in generation:
            generated_captions = [(' '.join(beam['tokens'][:-1]), '%.8f' % beam['score']) for beam in line]
            final_generation.append(generated_captions)
        for caption, path in zip(final_generation, paths):
            result.append({'caption': '\t'.join([y for x in caption for y in x]),
                           'image_path': path})


def filter_universal_reply(images, replies):
    for inputs, paths in tqdm(images, desc='generating with trick'):
        final_generation = []
        _, generation = model.generate(inputs, generator)
        for line in generation:
            for beam in line:
                generated_caption = ''.join(beam['tokens'][:-1])
                if generated_caption not in replies:
                    final_generation.append(generated_caption)
                    break
        for caption, path in zip(final_generation, paths):
            result.append({'caption': caption,
                           'image_path': path})


def save2json(filename):
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def save2text(filename):
    with open('generation.txt', 'w') as f:
        lines = []
        for item in result:
            caption = item['caption']
            image = item['image_path'].split('/')[-1]
            lines.append(image + '\t' + caption)
        f.write('\n'.join(lines))


if __name__ == '__main__':
    # save2json(args.output_file)
    # filter_universal_reply(images, load_reply('universal.txt')) if args.mode == 1 else generate(images)
    generate_beams(images)
    save2text(args.output_file)


