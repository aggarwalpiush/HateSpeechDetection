#! usr/bin/env python
# -*- coding:utf-8 -*-


import codecs
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main():
    in_filepath = "/Users/aggarwalpiush/github_repos/HateSpeechDetection/HateSpeechDetection/raw_data/en/gilbert/hate-speech-dataset-master/annotations_metadata.csv"
    heirarchical_path = "/Users/aggarwalpiush/github_repos/HateSpeechDetection/HateSpeechDetection/processed_data/en/gilbert/gilbert_hier.txt"


    with codecs.open(in_filepath, 'r', 'utf-8') as in_obj:
        gilbert_test_rec = {}
        gilbert_test_rec['text'] = []
        gilbert_test_rec['label'] = []
        gilbert_train_rec = {}
        gilbert_train_rec['text'] = []
        gilbert_train_rec['label'] = []
        for i, line in enumerate(in_obj):
            if i == 0:
                continue
            tokens = line.replace('\n', '').replace('\r', '').split(',')
            test_text_file = os.path.join(os.path.dirname(in_filepath), 'sampled_test', str(tokens[0])+'.txt')
            train_text_file = os.path.join(os.path.dirname(in_filepath), 'sampled_train', str(tokens[0])+'.txt')
            if os.path.exists(test_text_file):
                 with codecs.open(test_text_file, 'r', 'utf-8') as text_obj:
                     file_text = []
                     for line in text_obj:
                         file_text.append(line.replace('\n', '').replace('\r', ''))
                     gilbert_test_rec['text'].append(' '.join(file_text))
                     gilbert_test_rec['label'].append(0 if str(tokens[-1]).strip() == 'noHate' else 1)
            elif os.path.exists(train_text_file):
                with codecs.open(train_text_file, 'r', 'utf-8') as text_obj:
                    file_text = []
                    for line in text_obj:
                        file_text.append(line.replace('\n', '').replace('\r', ''))
                    gilbert_train_rec['text'].append(' '.join(file_text))
                    gilbert_train_rec['label'].append(0 if str(tokens[-1]).strip() == 'noHate' else 1)
            else:
                continue







    with codecs.open(heirarchical_path+'.train', 'w', 'utf-8') as outfile:
        for i, text in enumerate(gilbert_train_rec['text']):
            outfile.write(str(text) + '\t' + str(gilbert_train_rec['label'][i]) + '\n')

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'train.txt'), 'w', 'utf-8') as outfile:
        for i, text in enumerate(gilbert_train_rec['text']):
            outfile.write(str(text) + '\t' + str(gilbert_train_rec['label'][i]) + '\n')

    with codecs.open(heirarchical_path+'.test', 'w', 'utf-8') as outfile:
        for i, text in enumerate(gilbert_test_rec['text']):
            outfile.write(str(text) + '\t' + str(gilbert_test_rec['label'][i]) + '\n')


    # stratified split

    data = pd.read_csv(heirarchical_path+'.test', sep='\t', header=0, names=['text', 'label'])

    x_test, x_dev, y_test, y_dev = train_test_split(data['text'].values, data['label'].values, test_size=0.5)


    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'test.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_test)):
            outfile.write(str(x_test[i]) + '\t' + str(y_test[i]) + '\n')

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'dev.txt'), 'w', 'utf-8') as outfile:
        for i in range(len(x_dev)):
            outfile.write(str(x_dev[i]) + '\t' + str(y_dev[i]) + '\n')

if __name__ == '__main__':
    main()