# -*- coding: utf-8 -*-
import chardet
import os
import sys
import hashlib
import random
import json
import urllib
# import urllib.parse
from urllib import request, parse
import argparse

appid = 20161026000030798  # 20161029000030988
key = 'TOe0uWnkK1ws9uVfqnap'  # 7lViV3Bo31ygO7VRM8ZS
trans_api_url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

tolangs = ['zh']  # 'en']


def sign_cal(post_dict, key):
    signstr = str(post_dict['appid']) + post_dict['q'] + str(post_dict['salt']) + key
    mymd5 = hashlib.md5()
    mymd5.update(signstr.encode("utf8"))
    return mymd5.hexdigest()


def get_trans_result(query, fromlang, tolang):
    post_dict = dict()
    post_dict['q'] = query
    post_dict['from'] = fromlang
    post_dict['to'] = tolang
    post_dict['appid'] = appid
    post_dict['salt'] = random.randint(0, 9999999)
    post_dict['sign'] = sign_cal(post_dict, key)
    post_dict['sign'] = sign_cal(post_dict, key)

    post_cnt = parse.urlencode(post_dict)
    url = trans_api_url + '?' + post_cnt
    print(f'request url -> {url}')
    for i in range(3):
        try:
            trans_request = request.Request(url)
            response = request.urlopen(trans_request)
            jsonstr = response.read()
            trans_res = json.loads(jsonstr)
            if 'error_code' in trans_res:
                print >> sys.stderr, jsonstr
            else:
                return trans_res['trans_result'][0]['dst']
        except Exception as e:
            print(e)


def translate(args):
    spans = [line.strip().split('\t') for line in open(args.input_file)]
    results = []
    for span in spans:
        caption = span[1]
        translated = get_trans_result(caption, 'jp', tolangs[0])
        results.append(translated)

    with open(args.output_file, 'w') as f:
        for span, translated in zip(spans, results):
            print(f'span -> {span}')
            f.write('\t'.join(span) + '\t' + translated + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Translate Japanese caption to the Chinese')
    parser.add_argument('-i', '--input-file', type=str,)
    parser.add_argument('-o', '--output-file', type=str, default='generation.trans.txt')
    args = parser.parse_args()
    translate(args)
