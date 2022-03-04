import re
import argparse
import emoji
import MeCab
import numpy as np
import matplotlib.pyplot as plt

mecab = MeCab.Tagger('-Ochasen')

letters_pattern = re.compile(r'[a-zA-Z]+')
bracket_pairs = [['[', ']'], ['(', ')'], ['「', '」'], ['『', '』'], ['（', '）'],
                 ['(', '）'], ['（', ')']]

# Non-breaking space symbol for html
symbols = ['&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;',
           '&cent;', '&pound;', '&yen;', '&euro']


def has_target_postag(node):
    tokens = []
    has_noun = False
    has_adj = False
    while node:
        tokens.append(node.surface)
        features = node.feature.split(',')
        tag = features[0]
        tag_type = features[1]
        if tag == '名詞' and tag_type == '一般':
            has_noun = True
        #if tag == '形容詞':
            #has_adj = True
        node = node.next
    return tokens[1:-1], has_noun # and has_adj

def has_en_word(tokens):
    has_letter = False
    for token in tokens:
        if letters_pattern.findall(token):
            has_letter = True
            break
    return has_letter


def remove_bracket_content(text, bracket_pairs):
    low = 0
    high = 0
    for left_b, right_b in bracket_pairs:
        low = text.find(left_b)
        high = text.find(right_b, low)
        while low != -1 and high != -1:
            content = text[low:high + 1]
            text = text.replace(content, '')
            low = text.find(left_b)
            high = text.find(right_b, low)
    return text


def remove_special_symbol(text):
    for symbol in symbols:
        text = text.replace(symbol, '')
        text = text.replace(symbol[:-1], '')
    return text


def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(r'', text)


def main(args):
    f = open(args.output_file, 'w')

    freq_dict = dict()
    token_sum = 0
    sample_num = 0
    for line in open(args.input_file):
        items = line.strip().split('\t')
        if len(items) != 2:
            continue
        image = items[0]
        caption = items[1].replace(' ', '')
        # Remove content inside the bracket pairs
        caption = remove_bracket_content(caption, bracket_pairs) 
        # Remove special symbol
        caption = remove_special_symbol(caption)
        # Remove emoji
        caption = remove_emoji(caption)
        # Tokenize caption
        node = mecab.parseToNode(caption)
        tokens, postag_flag = has_target_postag(node)
        # Filter the caption with specific topics or tags
        if caption.find('【') != -1 and caption.find('】') != -1:
            # print(f'{line.strip()}')
            continue
        if len(tokens) < 5 or len(tokens) > 20:
            continue
        if has_en_word(tokens):
            # print(f'{line.strip()}')
            continue

        if postag_flag:
            token_sum += len(tokens)
            sample_num += 1
            if len(tokens) not in freq_dict:
                freq_dict[len(tokens)] = 1
            else:
                freq_dict[len(tokens)] += 1
            new_line = image + '\t' + ' '.join(tokens)
            f.write(new_line + '\n')
            # print(f'{new_line}')
    f.close()
    average_len = token_sum * 1.0 / sample_num
    print(f'Average token length -> {average_len}')

    # Plot the frequency curve
    ordered = sorted(freq_dict.items(), key=lambda tup: tup[0])
    x = np.array([t[0] for t in ordered])
    y = np.array([t[1] for t in ordered])
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(x, y)
    plt.grid(True, linestyle=':')
    plt.savefig('./freq-figure.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clean Train Data')
    parser.add_argument('-i', '--input-file', type=str)
    parser.add_argument('-o', '--output-file', type=str, default='./output.txt')
    args = parser.parse_args()
    main(args)

