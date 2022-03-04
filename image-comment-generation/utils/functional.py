import os
import unicodedata
import argparse
import numpy as np
from PIL import Image

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import matplotlib.pyplot as plt
plt.switch_backend('agg')


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


fig = None


def draw_caption(image, caption, ref, score, conf):
    global fig
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_axes((0, 0.18, 1, 0.82))  # [left, bottom, width, height]

    ax1.imshow(image)
    ax1.axis('off')
    text = f'CIDEr: {score:.2f} Confidence: {conf:.3f}\n' \
           f'{caption}\n' \
           f'(REF: {ref[0] if len(ref[0]) <= 90 else ref[0][:90] + "..."})'
    fig.text(.5, .07, text, ha='center', fontsize=10)

    fig.set_size_inches(7, 8, forward=True)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def save_image(arr, file):
    im = Image.fromarray(arr)
    im.save(file)
