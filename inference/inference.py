import json
import argparse
from functools import partial
import torch
import torch.nn.functional as F
from tqdm import tqdm
from generation import Model as GenerativeModel
from matching import Model as MatchingModel
from utils import str2bool
from utils import load_images
from utils import load_image_captions
from generation.generator import greedy, beam_search
import time

parser = argparse.ArgumentParser('Generate Image Comments')
parser.add_argument('-d', '--image-dir')
parser.add_argument('-o', '--output-file', type=str, default='generated_comments.txt')
parser.add_argument('--generative-model', type=str, default='checkpoints/best_generative_model.pt')
parser.add_argument('--matching-model', type=str, default='checkpoints/best_matching_model.pt')
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('--beam-size', type=int, default=20)
parser.add_argument('--max-len', type=int, default=18)
parser.add_argument('--cuda', type=str2bool, nargs='?',
                              const=True, default=torch.cuda.is_available(),
                              help='whether to use GPU acceleration.')
args = parser.parse_args()


def generate_beams(model, generator, images):
    result = []
    for inputs, paths in tqdm(images, desc='Generating all beams'):
        final_generation = []
        _, generation = model.generate(inputs, generator)
        for line in generation:
            generated_captions = [{'tokens': ' '.join(beam['tokens'][:-1]), 'score': '%.8f' % beam['score']} for beam in line]
            final_generation.append(generated_captions)
        for captions, path in zip(final_generation, paths):
            result.append({'captions': captions,
                           'image_path': path})
    return result


def matching_rank(model, samples):
    result = {}
    model.model.eval()
    for images, captions, image_paths in tqdm(samples, desc='Predict image-caption matching score'):
        caption_vectors = model.vectorize_text(captions, sample=False)
        outputs = model.model(images, caption_vectors)
        outputs = F.softmax(outputs, dim=1).transpose(1, 0)[1]
        outputs = outputs.data.cpu().numpy()
        for image_path, caption, score in zip(image_paths, captions[0], outputs):
            image_path = image_path.split('/')[-1]
            if image_path not in result:
                pairs = list()
                pairs.append((caption, score))
                result[image_path] = pairs
            else:
                result[image_path].append((caption, score))
    rank_result = []
    for image_path, pairs in result.items():
        ordered = sorted(pairs, key=lambda tup: tup[1], reverse=True)
        rank_result.append((image_path, ordered))
    return rank_result


def save2txt(result, output_file):
    with open(output_file, 'w') as f:
        for image_path, ordered in result:
            ss_str = '\t'.join([caption + '\t' + '%.8f' % score for caption, score in ordered])
            f.write(f'{image_path}\t{ss_str}\n')


if __name__ == '__main__':
    images = load_images(args.image_dir, args.batch_size)
    print(f'{time.asctime()}-Images loaded.')
    generative_model, _ = GenerativeModel.load(args, args.generative_model)
    print(f'{time.asctime()}-Generative Model loaded.')
    matching_model, _ = MatchingModel.load(args, args.matching_model)
    print(f'{time.asctime()}-Matching Model loaded.')

    if args.beam_size == 1:
        generator = greedy
    else:
        # generator = partial(beam_search, max_len=args.max_len, beam_size=args.beam_size,
        #                     len_penalty=0., dup_penalty=0., no_unk=True, mono_penalty=5., min_len=1,)
        generator = partial(beam_search, max_len=args.max_len, beam_size=args.beam_size,
                            len_penalty=0., dup_penalty=5., no_unk=True, mono_penalty=5., min_len=1, )
    # Generate beams for input images
    beam_result = generate_beams(generative_model, generator, images)
    # Rank beams by matching score
    image_caption_pairs = [(x['image_path'].split('/')[-1], y['tokens']) for x in beam_result for y in x['captions']]
    samples = load_image_captions(args.image_dir, image_caption_pairs, args.batch_size)
    final_result = matching_rank(matching_model, samples)
    # Save the generated comments to text file
    save2txt(final_result, args.output_file)

# Usage
# python inference.py -d images/
#                     --generative-model checkpoints/best_generative_model.pt
#                     --matching-model checkpoints/best_matching_model.pt

