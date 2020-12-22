#! usr/bin/env python
# -*- coding:utf-8 -*-


import codecs
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main():
    in_filepath1 = "../raw_data/en/hasoc/english_dataset_2019/english_dataset_2019.tsv"
    in_filepath2 = "../raw_data/en/hasoc/english_dataset_2019/hasoc2019_en_test-2919.tsv"
    heirarchical_path = "../processed_data/en/hasoc/hasoc_hier.txt"

    hasoc_train_rec = {}
    hasoc_train_rec['text'] = []
    hasoc_train_rec['label'] = []
    hasoc_test_rec = {}
    hasoc_test_rec['text'] = []
    hasoc_test_rec['label'] = []
    for file in [in_filepath1, in_filepath2]:
        with codecs.open(file, 'r', 'utf-8') as in_obj:
            for i, line in enumerate(in_obj):
                if i == 0:
                    continue
                tokens = line.replace('\n', '').replace('\r', '').split('\t')
                if file == in_filepath1:
                    hasoc_train_rec['text'].append(tokens[1])
                    hasoc_train_rec['label'].append(1 if str(tokens[2]) == 'HOF' else 0)
                else:
                    hasoc_test_rec['text'].append(tokens[1])
                    hasoc_test_rec['label'].append(1 if str(tokens[2]) == 'HOF' else 0)




    with codecs.open(heirarchical_path+'.test', 'w', 'utf-8') as outfile:
        for i, text in enumerate(hasoc_test_rec['text']):
            outfile.write(str(text) + '\t' + str(hasoc_test_rec['label'][i]) + '\n')

    with codecs.open(heirarchical_path+'.train', 'w', 'utf-8') as outfile:
        for i, text in enumerate(hasoc_train_rec['text']):
            outfile.write(str(text) + '\t' + str(hasoc_train_rec['label'][i]) + '\n')

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'train.txt'), 'w', 'utf-8') as outfile:
        for i, text in enumerate(hasoc_train_rec['text']):
            outfile.write(str(text) + '\t' + str(hasoc_train_rec['label'][i]) + '\n')


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