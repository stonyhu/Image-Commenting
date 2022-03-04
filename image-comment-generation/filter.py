# coding=utf-8
import os
import sys
import json
import string
import random
from tqdm import tqdm

image_dir = '/mnt/data/thumbnail-5M'
input_file = '/mnt/data/image-caption/twitter-comments-wordseg-5M.txt'

image_set1 = set(os.listdir(image_dir))

image_set2 = set()
for line in open(input_file):
    items = line.strip().split('\t')
    idx = items[0].rfind('/')
    image_file = items[0][idx + 1:]
    image_set2.add(image_file)

print(f'image_dir - twitter_comment_image: {len(image_set1 - image_set2)}')
print(f'twitter_comment_image - image_dir: {len(image_set2 - image_set1)}')

for filename in (image_set1 - image_set2):
    image_file = os.path.join(image_dir, filename)
    os.remove(image_file)
