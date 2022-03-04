import argparse
from collections import Counter

parser = argparse.ArgumentParser('Reply Count')
parser.add_argument('-i', '--input-file', type=str, default='generation.txt')
parser.add_argument('-o', '--output-file', type=str, default='replay.count.txt')
args = parser.parse_args()

replies = [line.strip().split('\t')[1] for line in open(args.input_file)]
reply_dict = Counter(replies)

[print(f'{reply}\t{cnt}') for reply, cnt in reply_dict.most_common(10)]

