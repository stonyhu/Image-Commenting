import os
import time
import random
import MeCab
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from model import Model
from utils.functional import get_test_loader, str2bool


mecab = MeCab.Tagger('-Ochasen')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--query-file', default='',
                    help='dialog queries')
parser.add_argument('-d', '--image-dir', default='/mnt/data/thumbnail-3M-resized',
                    help='image path')
parser.add_argument('--cuda', type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--model-file', type=str, default='output/best.pt')
parser.add_argument('--output-file', type=str, default='retrieval.result.txt')
parser.add_argument('-bs', '--batch-size', type=int, default=1024)
parser.add_argument('--max-len', type=int, default=18)
args = parser.parse_args()

images = os.listdir(args.image_dir)
random.shuffle(images)
images = images[:500_000]
queries = [l.strip().split('\t')[0] for l in open(args.query_file)]
queries = queries[:50]
pairs = [(image, query) for image in images for query in queries]
print(f'Queries: {len(queries)}, Images: {len(images)}')

dataloader = get_test_loader(args.image_dir, pairs, args.batch_size)
print(f'{time.asctime()}-Images and Queries loaded.')
matching_model, _ = Model.load(args, args.model_file)
print(f'{time.asctime()}-Matching Model loaded.')


def predict(dataloader, batch_size):
    matching_model.model.eval()
    i_batch = 0
    data_size = len(images) * len(queries)
    sims = np.zeros(data_size)
    for imgs, caps, img_paths in tqdm(dataloader, desc='Predicting'):
        caps_vecs = matching_model.vectorize_text(caps, sample=False)
        output = matching_model.model(imgs, caps_vecs)
        output = F.softmax(output, dim=1).transpose(1, 0)[1]
        output = output.data.cpu().numpy()

        start_idx = i_batch * batch_size
        end_idx = min((i_batch + 1) * batch_size, data_size)
        sims[start_idx:end_idx] = output
        i_batch += 1

    sims = sims.reshape(len(images), len(queries))
    return sims.T


def save(sims):
    with open(args.output_file, 'w') as f:
        for i in range(len(queries)):
            top5 = np.argsort(sims[i])[::-1][:5]
            query = queries[i]
            image_paths = [os.path.join(args.image_dir, images[idx]) for idx in top5]
            f.write(query + '\t' + '\t'.join(image_paths) + '\n')


if __name__ == '__main__':
    sims = predict(dataloader, args.batch_size)
    save(sims)

