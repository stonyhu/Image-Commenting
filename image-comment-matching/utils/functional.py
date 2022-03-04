import os
import argparse
import unicodedata
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed

IMG_SZ = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_data(data_dir, split, batch_size, is_distributed=False):
    """load images and annotations for training or evaluation"""
    dataset = MatchingDataset(root=os.path.join(data_dir, 'images'),
                              annFile=os.path.join(data_dir, f'{split}.json'),
                              transform=transforms.Compose([
                                  transforms.Resize(IMG_SZ),
                                  transforms.RandomCrop(IMG_SZ) if split == 'train' else transforms.CenterCrop(IMG_SZ),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=IMAGENET_MEAN,
                                                       std=IMAGENET_STD),
                              ]))
    if is_distributed and split == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    return DataLoader(dataset, batch_size, shuffle=(split == 'train' and not sampler),
                      pin_memory=True, num_workers=min(4, batch_size), sampler=sampler), sampler


def load_image_caption(image_dir, caption_file, batch_size):
    """load image-caption pairs without labels for matching"""
    pairs = []
    for l in open(caption_file):
        items = l.strip().split('\t')
        for i, caption in enumerate(items[1:]):
            if i % 2 == 0:
                pairs.append((items[0], caption))
    dataset = ImageCaptionDataset(root=image_dir, image_caption_pairs=pairs,
                                  transform=transforms.Compose([
                                      transforms.Resize(IMG_SZ),
                                      transforms.CenterCrop(IMG_SZ),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]))
    return DataLoader(dataset, batch_size, shuffle=False, pin_memory=True, num_workers=min(4, batch_size))


def get_test_loader(image_dir, pairs, batch_size):
    """retrieve image by the given query"""
    # pairs = [(img, cap) for img in images for cap in captions]
    dataset = ImageCaptionDataset(root=image_dir, image_caption_pairs=pairs,
                                  transform=transforms.Compose([
                                      transforms.Resize(IMG_SZ),
                                      transforms.CenterCrop(IMG_SZ),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]))
    return DataLoader(dataset, batch_size, shuffle=False, pin_memory=True, num_workers=min(4, batch_size))


image_mean = None
image_std = None


def restore_image(img):
    global image_mean, image_std
    if image_mean is None:
        image_mean = img.new_tensor([[IMAGENET_MEAN]])
    if image_std is None:
        image_std = img.new_tensor([[IMAGENET_STD]])
    img = img.squeeze(0).permute(1, 2, 0)
    img = img * image_std + image_mean
    img.clamp_(0., 1.)
    return img.cpu().numpy()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize_text(text):
    # return unicodedata.normalize('NFD', text)
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()


class MatchingDataset(Dataset):
    """load all images in a folder with labels"""
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, caption, target).
            target is a list of label for the image-caption pair.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        caption = [ann['caption'] for ann in anns]
        target = [ann['label'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, caption, torch.Tensor(target).long()

    def __len__(self):
        return len(self.ids)


class ImageCaptionDataset(Dataset):
    """load all image-caption pairs in a folder without labels"""
    def __init__(self, root, image_caption_pairs, transform=None):
        self.root = os.path.expanduser(root)
        self.pairs = image_caption_pairs
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.pairs[index][0])
        caption = self.pairs[index][1]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, [caption], image_path

    def __len__(self):
        return len(self.pairs)

