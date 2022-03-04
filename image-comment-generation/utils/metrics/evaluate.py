import string
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer

from .bleu.bleu import Bleu
from .rouge.rouge import Rouge
from .cider.cider import Cider


tokenizer = TweetTokenizer()
punc = set(string.punctuation)
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
]


def evaluate(model, data, generator):
    """
    :param model: model.Model
    :param data: torch.utils.DataLoader
    :param generator: generator function in model.generator
    :param sample_scheme:
        None: will return no samples
        int: will return first N samples
        'analysis': will return samples following error analysis scheme
    :return metrics: dict[str: float]
    """
    generation = []
    reference = []
    outputs = []

    for inputs, captions in tqdm(data, desc='generate', leave=False):
        text, output = model.generate(inputs, generator)
        generation.extend(text)
        reference.extend(map(list, zip(*captions)))
        outputs.extend(output)

    for sample in tqdm(reference, desc='tokenizing ref', leave=False):
        for i, caption in enumerate(sample):
            sample[i] = tokenize_ref(caption)
    scorer_result = generation_score(reference, generation)
    return scorer_result, outputs


def tokenize_ref(text):
    return ' '.join(token for token in tokenizer.tokenize(text.lower()) if token not in punc)


def generation_score(reference, generation):
    """
    :param reference: list[list[str]]
    :param generation: list[str]
    :return: dict[str:(float, list[float])]

    Captions are lowered, separated by spaces and without punctuations
    """
    reference = {i: captions for i, captions in enumerate(reference)}
    generation = {i: [captions] for i, captions in enumerate(generation)}
    result = {}
    for scorer, method in scorers:
        print('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(reference, generation)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                result[m] = (sc, scs)
        else:
            result[method] = (score, scores)
    return result
