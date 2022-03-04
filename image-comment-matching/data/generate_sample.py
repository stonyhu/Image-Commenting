import argparse
import numpy as np


parser = argparse.ArgumentParser('Negative Sampler')
parser.add_argument('-i', '--input-file')
parser.add_argument('-n', '--negative-num', type=int,
                    help='the number of negative samples')
args = parser.parse_args()


samples = []
for line in open(args.input_file):
    items = line.strip().split('\t')
    samples.append([items[0], items[1]])

indices = np.arange(len(samples)).tolist()
for i, sample in enumerate(samples):
    pool = indices[:i] + indices[i + 1:]
    negative_sample_indices = np.random.choice(pool, args.negative_num, replace=False)
    print(f'{sample[0]}\t{sample[1]}\t1')
    for idx in negative_sample_indices:
        caption = samples[idx][1]
        print(f'{sample[0]}\t{caption}\t0')


