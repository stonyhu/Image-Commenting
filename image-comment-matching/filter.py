import sys
import MeCab

mecab = MeCab.Tagger('-Ochasen')

bracket_pairs = [['[', ']'], ['(', ')'], ['「', '」'], ['『', '』'], ['（', '）'],
                 ['(', '）'], ['（', ')'], ['【', '】']]


def filter_bracket():
    brackets = [y for x in bracket_pairs for y in x]
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        if len(items) != 2:
            continue
        caption = items[1]
        for b in brackets:
            caption = caption.replace(b, '')
        print(f'{items[0]}\t{caption}')


def filter_ner():
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        image = items[0]
        target = 'General Reply'
        for i in range(1, len(items), 2):
            caption = items[i].replace(' ', '')
            flag = False
            node = mecab.parseToNode(caption)
            while node:
                token = node.surface
                features = node.feature.split(',')
                tag = features[0]
                sub_tag = features[1]
                if tag == '名詞' and sub_tag == '固有名詞' and len(token) >= 2:
                    flag = True
                    break
                node = node.next
            if not flag:
                target = items[i] + '\tsame' if i == 1 else items[i] + '\t' + items[1]
                break
        print(f'{image}\t{target}')


def filter_vocab(filename):
    forbidden_vocab = set([l.strip() for l in open(filename)][-10000:])
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        image = items[0]
        target = 'General Reply'
        for i in range(1, len(items), 2):
            tokens = items[i].split()
            flag = False
            for token in tokens:
                if token in forbidden_vocab:
                    flag = True
                    break
            if not flag:
                target = items[i] + '\tsame' if i == 1 else items[i] + '\t' + items[1]
                break
        print(f'{image}\t{target}')


vocab_file = 'vocab.txt'
filter_vocab(vocab_file)

# filter_ner()
