import json
import shutil
import random
import numpy as np

import os

from tqdm import tqdm

valid_json_file = 'data/valid.json'
sample_num = 200
sample_output_dir = f'valid_sample_{sample_num}'
images_output_dir = os.path.join(sample_output_dir, 'images')
caption_file = os.path.join(sample_output_dir, 'captions.txt')
if not os.path.exists(images_output_dir):
    os.makedirs(images_output_dir)


id2image_dict = {}
id2caption_dict = {}

with open(valid_json_file, 'r')as p:
    data_set = json.load(p)
    images, annotations = data_set['images'], data_set['annotations']
    for im in tqdm(images, total=len(images)):
        im_id, im_file = im['id'], im['file_name']
        id2image_dict[im_id]= im_file

    for anno in tqdm(annotations, total=len(annotations)):
        im_id, caption = anno['image_id'], anno['caption']
        id2caption_dict[im_id] = caption

order = np.arange(len(id2image_dict))
np.random.shuffle(order)
sample_idx = order[:200]


for index, sample in enumerate(id2image_dict.items()):
    if index in sample_idx:
        im_id, im_file = sample
        base_name = os.path.split(im_file)[-1]
        caption = id2caption_dict[im_id]
        shutil.copy2(im_file, images_output_dir)
        with open(caption_file, 'a')as p:
            p.write(f'{base_name}\t{caption}\n')

print('Bingo')

