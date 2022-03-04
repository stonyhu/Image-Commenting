import os
import json
from tqdm import tqdm
from utils.vocab import Vocab

"""
sample "images":
{'coco_url': 'http://mscoco.org/images/391895',
 'date_captured': '2013-11-14 11:18:45',
 'file_name': 'COCO_val2014_000000391895.jpg',
 'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
 'height': 360,
 'id': 391895,
 'license': 3,
 'width': 640}

sample "annotations":
{'caption': 'A bicycle replica with a clock as the front wheel.',
 'id': 37,
 'image_id': 203564,
 'label': 1}
"""

# in
anno_file = 'data/twitter-comment-pairs.txt'
image_dir = '/mnt/data/thumbnail-5M-resized'
embeddings_file = 'data/twitter-vectors.512d.txt'
# out
vocab_file = 'data/vocab.txt'
train_out_file = 'data/train.json'
valid_out_file = 'data/valid.json'
test_out_file = 'data/test.json'
# param
vocab_size = 40_000
max_oov = 0.1

data = []
id2img = {}
with open(anno_file) as f:
    for i, line in enumerate(f):
        image_id, comment, label = line.strip().split('\t')
        data.append({'image_id': i, 'caption': comment, 'label': int(label), 'id': i})
        id2img[i] = image_id.split('/')[-1]
train_anno = data[:-200_000]
valid_anno = data[-200_000:-100_000]
test_anno = data[-100_000:]

vocab = Vocab.build_vocab((word for anno in train_anno for word in anno['caption'].split()),
                          max_tokens=vocab_size,
                          pretrained_embeddings=embeddings_file)
vocab.save(vocab_file)


def oov_filter(example):
    comment = example['caption']
    tokens = comment.split()
    ratio = sum(token not in vocab for token in tokens) / len(tokens)
    return ratio < max_oov


print(len(train_anno), end=' -> ')
train_anno = list(filter(oov_filter, train_anno))
print(len(train_anno))

train_images = []
valid_images = []
test_images = []
train_exist = []
valid_exist = []
test_exist = []
for anno, images, image_exist in zip([train_anno, valid_anno, test_anno], [train_images, valid_images, test_images],
                                     [train_exist, valid_exist, test_exist]):
    for sample in tqdm(anno):
        filename = os.path.join(image_dir, id2img[sample['id']])
        image = {'id': sample['image_id']}
        is_exist = os.path.exists(filename)
#        print(filename)
        image_exist.append(is_exist)
        if is_exist:
            image['file_name'] = filename
            images.append(image)

print()
print(len(train_anno), '->', len(train_images))
print(len(valid_anno), '->', len(valid_images))
print(len(test_anno), '->', len(test_images))
train_anno = [sample for sample, flag in zip(train_anno, train_exist) if flag]
valid_anno = [sample for sample, flag in zip(valid_anno, valid_exist) if flag]
test_anno = [sample for sample, flag in zip(test_anno, test_exist) if flag]


train_out = {
    'images': train_images,
    'annotations': train_anno,
}
valid_out = {
    'images': valid_images,
    'annotations': valid_anno,
}
test_out = {
    'images': test_images,
    'annotations': test_anno,
}

with open(train_out_file, 'w') as f:
    json.dump(train_out, f)
with open(valid_out_file, 'w') as f:
    json.dump(valid_out, f)
with open(test_out_file, 'w') as f:
    json.dump(test_out, f)
