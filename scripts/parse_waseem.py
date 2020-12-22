#! usr/bin/env python
# -*- coding:utf-8 -*-


import sys, os
sys.path.append(os.path.abspath(".."))

import codecs
import pandas as pd
from sklearn.model_selection import train_test_split
import tweepy
from utils.auth import CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET


def main():
    in_filepath = "../raw_data/en/waseem/NAACL_SRW_2016.csv"
    heirarchical_path = "../processed_data/en/waseem/waseem_hier.txt"

    # auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    # auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    #
    #
    #
    # with codecs.open(in_filepath, 'r', 'utf-8') as in_obj:
    #     waseem_rec = {}
    #     waseem_rec['text'] = []
    #     waseem_rec['label'] = []
    #     for i, line in enumerate(in_obj):
    #         tokens = line.replace('\n', '').replace('\r', '').split(',')
    #
    #         # get text from
    #         print(int(tokens[0]))
    #         api = tweepy.API(auth, wait_on_rate_limit=True)
    #         try:
    #             tweet = api.get_status(int(tokens[0]))
    #         except tweepy.TweepError as e:
    #             print(e.args[0][0]['code'])
    #             print(e.args[0][0]['message'])
    #             continue
    #         print(tweet.text)
    #
    #
    #         waseem_rec['text'].append(tweet.text)
    #         if str(tokens[1]) in ['racism', 'sexism']:
    #             waseem_rec['label'].append(1)
    #         else:
    #             waseem_rec['label'].append(0)
    #
    #
    #
    # with codecs.open(heirarchical_path, 'w', 'utf-8') as outfile:
    #     for i, text in enumerate(waseem_rec['text']):
    #         outfile.write(str(text) + '\t' + str(waseem_rec['label'][i]) + '\n')


    # stratified split

    data = pd.read_csv(heirarchical_path, sep='\t', header=0, names=['text', 'label'])

    x_train, x_temp, y_train, y_temp = train_test_split(data['text'].values, data['label'].values, test_size=0.3)

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'train.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_train)):
            try:
                outfile.write(str(x_train[i]) + '\t' + str(int(y_train[i])) + '\n')
            except ValueError:
                continue


    x_test, x_dev, y_test, y_dev = train_test_split(x_temp, y_temp, test_size=0.5)

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'test.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_test)):
            try:
                outfile.write(str(x_test[i]) + '\t' + str(int(y_test[i])) + '\n')
            except ValueError:
                continue

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'dev.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_dev)):
            try:
                outfile.write(str(x_dev[i]) + '\t' + str(int(y_dev[i])) + '\n')
            except ValueError:
                continue

if __name__ == '__main__':
    main()