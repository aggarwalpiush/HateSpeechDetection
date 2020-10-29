#! usr/bin/env python
# -*- coding:utf-8 -*-


import codecs
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main():
    in_filepath = "/Users/aggarwalpiush/github_repos/HateSpeechDetection/HateSpeechDetection/raw_data/en/ethos/Ethos_Dataset_Binary.csv"
    output_path = "/Users/aggarwalpiush/github_repos/HateSpeechDetection/HateSpeechDetection/processed_data/en/ethos/Ethos_Dataset_Binary.jsonl"
    heirarchical_path = "/Users/aggarwalpiush/github_repos/HateSpeechDetection/HateSpeechDetection/processed_data/en/ethos/Ethos_Dataset_Binary_hier.txt"

    gab_record = {}
    gab_text = {}

    with codecs.open(in_filepath, 'r', 'utf-8') as in_obj:
        ethos_rec = {}
        ethos_rec['text'] = []
        ethos_rec['label'] = []
        for i, line in enumerate(in_obj):
            if i == 0:
                continue
            tokens = line.replace('\n', '').replace('\r', '').split(';')
            ethos_rec['text'].append(';'.join(tokens[:-1]))
            if float(tokens[-1]) >= 0.5:
                ethos_rec['label'].append(1)
            else:
                ethos_rec['label'].append(0)



    with codecs.open(heirarchical_path, 'w', 'utf-8') as outfile:
        for i, text in enumerate(ethos_rec['text']):
            outfile.write(str(text) + '\t' + str(ethos_rec['label'][i]) + '\n')


    # stratified split

    data = pd.read_csv(heirarchical_path, sep='\t', header=0, names=['text', 'label'])

    x_train, x_temp, y_train, y_temp = train_test_split(data['text'].values, data['label'].values, test_size=0.3)

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'train.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_train)):
            outfile.write(str(x_train[i]) + '\t' + str(y_train[i]) + '\n')


    x_test, x_dev, y_test, y_dev = train_test_split(x_temp, y_temp, test_size=0.5)

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'test.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_test)):
            outfile.write(str(x_test[i]) + '\t' + str(y_test[i]) + '\n')

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'dev.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_dev)):
            outfile.write(str(x_dev[i]) + '\t' + str(y_dev[i]) + '\n')

if __name__ == '__main__':
    main()