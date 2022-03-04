import sys
import json
import random
import argparse
from urllib import request


parser = argparse.ArgumentParser('Request Dialog Engine')
parser.add_argument('-i', '--input-file', type=str,)
parser.add_argument('-o', '--output-file', type=str, default='dialog.response.txt')
args = parser.parse_args()

engine_url = 'http://phonechitchat.trafficmanager.net:16484/api/reply/multi?displayAct=false&responseType={0}&replyCnt=5&traceId=23'


def request_response(query, url):
    post_dict = dict()
    post_dict['LastResponse'] = ''
    post_dict['CurrentQuery'] = query
    post_dict['ConversationHistory'] = []

    headers = {
        "Content-Type": "application/json",
    }
    json_str = json.dumps(post_dict).encode('utf-8')

    for i in range(3):
        try:
            dialog_request = request.Request(url, data=json_str, headers=headers)
            response = request.urlopen(dialog_request)
            responses = json.loads(response.read())
            # result = responses[random.randint(0, len(responses))]
            result = responses[0]
            reply = result['ReplyText']
            print(f'reply count -> {len(responses)}\t{reply}')
            return reply
        except Exception as e:
            print(e)


with open(args.output_file, 'w') as f:
    for line in open(args.input_file):
        items = line.strip().split('\t')
        image = items[0]
        caption = items[1]
        reply = request_response(caption, engine_url)
        f.write(f'{image}\t{reply}\n')

