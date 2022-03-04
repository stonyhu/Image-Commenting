import os
import sys
import numpy as np

image_dir = '/mnt/data/thumbnail-5M-resized'
print(f'Loading images from the directory -> {image_dir}')
image_set = set(os.listdir(image_dir))

train_set = set([line.strip().split('\t')[0].split('/')[-1] for line in open(sys.argv[1])])

#print(f'image_set - train_set -> {len(image_set - train_set)}')
print(f'train_set - image_set -> {len(train_set - image_set)}')

