import time
import MeCab
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from model import Model
from utils.functional import load_image_caption, str2bool


mecab = MeCab.Tagger('-Ochasen')

parser = argparse.ArgumentParser('Rerank all generated captions')
parser.add_argument('-d', '--image-dir')
parser.add_argument('-i', '--caption-file')
parser.add_argument('--cuda', type=str2bool, nargs='?',
                              const=True, default=torch.cuda.is_available(),
                              help='whether to use GPU acceleration.')
parser.add_argument('--model-file', type=str, default='output/best.pt')
parser.add_argument('-o', '--output-file', type=str, default='generation.rerank.txt')
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('--max-len', type=int, default=18)
args = parser.parse_args()


samples = load_image_caption(args.image_dir, args.caption_file, args.batch_size)
print(f'{time.asctime()}-Images and Captions loaded.')
matching_model, _ = Model.load(args, args.model_file)
print(f'{time.asctime()}-Matching Model loaded.')


def predict(samples):
    result = {}
    matching_model.model.eval()
    for images, captions, image_paths in tqdm(samples, desc='Predicting'):
        caption_vecs = matching_model.vectorize_text(captions, sample=False)
        outputs = matching_model.model(images, caption_vecs)
        outputs = F.softmax(outputs, dim=1).transpose(1, 0)[1]
        outputs = outputs.data.cpu().numpy()
        for image_path, caption, score in zip(image_paths, captions[0], outputs):
            image_path = image_path.split('/')[-1]
            if image_path not in result:
                score_dict = dict()
                score_dict[caption] = score
                result[image_path] = score_dict
            else:
                result[image_path][caption] = score

    sorted_result = {}
    for image_path, score_dict in result.items():
        ordered = sorted(score_dict.items(), key=lambda tup: tup[1], reverse=True)
        sorted_result[image_path] = [(caption, '%.8f' % score) for caption, score in ordered]
    return result, sorted_result


def rerank(pred_result):
    gen_result = {}
    for l in open(args.caption_file):
        items = l.strip().split('\t')
        image_path = items[0]
        prob_dict = dict()
        for i in range(1, len(items), 2):
            caption = items[i]
            prob = float(items[i + 1])
            prob_dict[caption] = prob
        gen_result[image_path] = prob_dict

    for image_path, score_dict in pred_result.items():
        for caption in score_dict.keys():
            score_dict[caption] *= gen_result[image_path][caption]

    sorted_result = dict()
    for image_path, score_dict in pred_result.items():
        ordered = sorted(score_dict.items(), key=lambda tup: tup[1], reverse=True)
        sorted_result[image_path] = [(caption, '%.8f' % score) for caption, score in ordered]
    return sorted_result


def filter_ner(result):
    new_result = {}
    for key in result.keys():
        pairs = result[key]
        new_pairs = []
        for caption, score in pairs:
            text = caption.replace(' ', '')
            node = mecab.parseToNode(text)
            flag = False
            while node:
                token = node.surface
                features = node.feature.split(',')
                tag = features[0]
                tag_type = features[1]
                if tag == '名詞' and tag_type == '固有名詞' and len(token) >= 2:
                    flag = True
                    break
                node = node.next
            if not flag:
                new_pairs.append((caption, score))
        new_result[key] = new_pairs
    return new_result


def save2txt(result, filename):
    with open(filename, 'w') as f:
        for l in open(args.caption_file):
            image_path = l.strip().split('\t')[0]
            captions = result[image_path]
            out_str = '\t'.join([y for x in captions for y in x])
            f.write(f'{image_path}\t{out_str}\n')


if __name__ == '__main__':
    pred_result, sorted_result = predict(samples)
    # result1 = filter_ner(sorted_result)
    save2txt(sorted_result, 'generation.matching.txt')
    rerank_result = rerank(pred_result)
    # result2 = filter_ner(rerank_result)
    save2txt(rerank_result, 'generation.rerank.txt')


