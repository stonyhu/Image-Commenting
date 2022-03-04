from random import Random
import numpy as np
from utils.data import restore_image


def sampler(dataset, n, outputs, metrics, seed=123):
    """perform error analysis"""
    assert len(dataset) == len(outputs)
    assert 0 <= n <= len(dataset)
    if n == 0:
        return {}
    scores = np.array(metrics['CIDEr'][1])
    confidence = np.array([np.mean(np.exp(out[0]['positional_scores'])) for out in outputs])

    def get_samples(indices):
        images = []
        captions = []
        texts = []
        for i in indices:
            image, caption = dataset[i]
            image = restore_image(image)
            images.append(image)
            captions.append(caption)
            texts.append(' '.join(outputs[i][0]['tokens'][:-1]))
        return [{'image':image, 'ref': caption, 'gen': text, 'score': score, 'conf': conf}
                for image, caption, text, score, conf in
                zip(images, captions, texts, scores[indices], confidence[indices])]

    groups = {}

    indices = np.argsort(scores)
    groups['high_score'] = get_samples(indices[-n:])
    groups['low_score'] = get_samples(indices[:n])
    indices = np.argsort(confidence)
    groups['confident'] = get_samples(indices[-n:])
    groups['unconfident'] = get_samples(indices[:n])
    random = Random(seed)
    groups['random'] = get_samples(random.sample(range(len(dataset)), n))

    return groups
