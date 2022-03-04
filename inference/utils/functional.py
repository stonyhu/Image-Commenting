import os
import argparse
import unicodedata
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS
import torchvision.transforms as transforms

IMG_SZ = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_images(image_dir, batch_size):
    """load images without labels for generation"""
    dataset = ImageDataset(root=image_dir,
                           transform=transforms.Compose([
                               transforms.Resize(IMG_SZ),
                               transforms.CenterCrop(IMG_SZ),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_MEAN,
                                                    std=IMAGENET_STD,
                                                    )]))
    return DataLoader(dataset, batch_size, shuffle=False,
                      pin_memory=True, num_workers=min(4, batch_size))


def load_image_captions(image_dir, image_caption_pairs, batch_size):
    dataset = ImageCaptionDataset(root=image_dir, image_caption_pairs=image_caption_pairs,
                                  transform=transforms.Compose([
                                      transforms.Resize(IMG_SZ),
                                      transforms.CenterCrop(IMG_SZ),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                  ]))
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


class ImageDataset(Dataset):
    """load all images in a folder without labels"""
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.images = [path for ext in IMG_EXTENSIONS for path in glob(os.path.join(root, f'*{ext}'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path


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

