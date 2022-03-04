import os
import sys
import logging
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Queue, Pool, Process
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = logging.getLogger("ImageResizer")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
handler = logging.FileHandler('log.resize.txt', mode="w")
handler.setFormatter(formatter)
logger.addHandler(handler)


def resize_image(images, image_dir, output_dir, size):
    for image in images:
        image_path = os.path.join(image_dir, image)
        if not os.path.exists(image_path):
            continue
        with open(image_path, 'r+b') as f:
            #with Image.open(f) as img:
            try:
                img = Image.open(f)
            except OSError as e:
                print(e)
                continue
            img = img.resize([size, size], Image.ANTIALIAS)
            try:
                img.save(os.path.join(output_dir, image), img.format)
            except:
                img = img.convert('RGB')
                img.save(os.path.join(output_dir, image), img.format)
            finally:
                logger.info(f'> {image} processed')


def main(args):
    sample_file = args.sample_file
    input_dir = args.input_dir
    output_dir = args.output_dir
    size = args.image_size

    if not os.path.exists(output_dir):
        os.makedirs(args.output_dir)

    image_set = set(os.listdir(output_dir))
    train_set = set([line.strip().split('\t')[0].split('/')[-1] for line in open(sample_file)])
    images = list(train_set - image_set)
    print(f'train_set - image_set -> {len(images)}')
    n = 70000
    image_lists = [images[i:i + n] for i in range(0, len(images), n)]
    for i, image_list in enumerate(image_lists):
        print(f'list-{i} -> {len(image_list)}')

    processes = []
    for image_list in image_lists:
        process = Process(target=resize_image, args=(image_list, input_dir, output_dir, size,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    logging.info('Resize done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-file', type=str, help='train data file')
    parser.add_argument('--input-dir', type=str,
                        help='directory for source images')
    parser.add_argument('--output-dir', type=str, default='./resized',
                        help='directory for saving resized images')

    parser.add_argument('--image-size', type=int, default=256,  # for cropping purpose
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
