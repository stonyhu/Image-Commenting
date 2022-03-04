# coding=utf-8
import numpy as np
import torch
from .functional import normalize_text
from collections import Counter


class Vocab:
    def __init__(self):
        self.w2id = {
            self.pad_symbol(): self.pad(),
            self.unk_symbol(): self.unk(),
            self.eos_symbol(): self.eos(),
        }
        self.id2w = {i: w for w, i in self.w2id.items()}
        self.n_spec = len(self.w2id)
        assert self.n_spec == max(self.id2w.keys()) + 1, "empty indices found in special tokens"
        assert len(self.id2w) == len(self.w2id), "index conflict in special tokens"

    def __getitem__(self, index):
        if index not in self.id2w:
            raise IndexError('invalid index {} in vocab.'.format(index))
        return self.id2w[index]

    def __len__(self):
        return len(self.w2id)

    def __contains__(self, item):
        return item in self.w2id

    def index(self, symbol):
        return self.w2id[symbol] if symbol in self.w2id else self.unk()

    @staticmethod
    def build_vocab(words, min_df=1, max_tokens=float('inf'), pretrained_embeddings=None):
        if pretrained_embeddings:
            wv_vocab = Vocab.load_embedding_vocab(pretrained_embeddings)
            print(f'pre-trained embedding lookup table size -> {len(wv_vocab)}')
        else:
            wv_vocab = set()

        counter = Counter(words)
        tokens = sorted([t for t, c in counter.items() if t in wv_vocab or c >= min_df],
                        key=counter.get, reverse=True)
        max_tokens = int(min(max_tokens, len(tokens)))
        tokens = tokens[:max_tokens]
        total = sum(counter.values())
        matched = sum(counter[t] for t in tokens)
        stats = (len(tokens), len(counter), total - matched, total, (total - matched) / total * 100)
        print('vocab coverage {}/{} | OOV occurrences {}/{} ({:.4f}%)'.format(*stats))
        print('top words:\n{}'.format(','.join(tokens[:10])))
        filtered = sorted(list(counter.keys() - set(tokens)), key=counter.get, reverse=True)
        print('filtered words:\n{} ... {}'.format(' '.join(filtered[:10]), ' '.join(filtered[-10:])))
        vocab = Vocab()
        for token in tokens:
            vocab.add_symbol(token)
        if pretrained_embeddings is not None:
            common_num = len(wv_vocab & vocab.w2id.keys())
            hit_rate = common_num * 1.0 / len(vocab.w2id)
            print('{:.2f}% words of vocab are hit in the pre-trained lookup table'.format(hit_rate * 100))
        return vocab

    @staticmethod
    def load_embedding_vocab(file):
        wv_vocab = set()
        with open(file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(' ')
                token = parts[0]
                wv_vocab.add(token)
        return wv_vocab

    def load_embeddings(self, file, padding=True):
        print('load the pre-trained word embeddings...')
        wv_dict = dict()
        dim = 0
        with open(file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    dim = int(line.strip().split(' ')[-1])
                    continue
                parts = line.strip().split(' ')
                token = parts[0]
                vector = [float(x) for x in parts[1:]]
                wv_dict[token] = vector

        weight = []
        count = 0
        for i in range(len(self.id2w)):
            if self.id2w[i] == self.pad_symbol():
                weight.append(np.zeros(dim)) if padding else weight.append(np.random.normal(0, 1, dim))
            elif self.id2w[i] == self.unk_symbol() or self.id2w[i] == self.eos_symbol():
                weight.append(np.random.normal(0, 1, dim))
            else:
                if self.id2w[i] in wv_dict:
                    weight.append(wv_dict[self.id2w[i]])
                    count += 1
                else:
                    weight.append(np.random.normal(0, 1, dim))
        print(f'pre-trained hit rate -> {count * 100.0 / (len(weight) - 3)}%')
        return torch.Tensor(weight)

    def add_symbol(self, symbol):
        if symbol not in self.w2id:
            self.id2w[len(self.id2w)] = symbol
            self.w2id[symbol] = len(self.w2id)

    @staticmethod
    def pad():
        return 0

    @staticmethod
    def unk():
        return 1

    @staticmethod
    def eos():
        return 2

    @staticmethod
    def pad_symbol():
        return '<PAD>'

    @staticmethod
    def unk_symbol():
        return '<UNK>'

    @staticmethod
    def eos_symbol():
        return '<EOS>'

    def save(self, file):
        assert '\t' not in self.w2id and '\n' not in self.w2id, \
            'tabs and new-lines are not supported by current vocab save protocol. ' \
            'Please escape them in preprocessing'
        with open(file, 'w') as f:
            for symbol, index in self.w2id.items():
                if index < self.n_spec:
                    continue
                f.write('{}\n'.format(symbol))

    @staticmethod
    def load(file):
        vocab = Vocab()
        with open(file) as f:
            for line in f:
                vocab.add_symbol(line.rstrip())
        return vocab
