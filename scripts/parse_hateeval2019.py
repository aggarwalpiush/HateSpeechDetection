#! usr/bin/env python
# -*- coding:utf-8 -*-


import codecs
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main():
    in_filepath_dev = "../raw_data/en/hateeval2019/hateval2019_en_dev.csv"
    in_filepath_train = "../raw_data/en/hateeval2019/hateval2019_en_train.csv"
    in_filepath_test = "../raw_data/en/hateeval2019/hateval2019_en_test.csv"
    heirarchical_path = "../processed_data/en/hateeval2019/hateval2019_hier.txt"

    hateeval_train_rec = {}
    hateeval_train_rec['text'] = []
    hateeval_train_rec['label'] = []
    hateeval_test_rec = {}
    hateeval_test_rec['text'] = []
    hateeval_test_rec['label'] = []
    hateeval_dev_rec = {}
    hateeval_dev_rec['text'] = []
    hateeval_dev_rec['label'] = []
    for file in [in_filepath_dev, in_filepath_train, in_filepath_test]:
        with codecs.open(file, 'r', 'utf-8') as in_obj:
            for i, line in enumerate(in_obj):
                if i == 0:
                    continue
                tokens = line.replace('\n', '').replace('\r', '').split(',')
                if file == in_filepath_train:
                    hateeval_train_rec['text'].append(','.join(tokens[1:-3]))
                    hateeval_train_rec['label'].append(int(tokens[-3]))
                elif file == in_filepath_test:
                    hateeval_test_rec['text'].append(','.join(tokens[1:-3]))
                    hateeval_test_rec['label'].append(int(tokens[-3]))
                elif file == in_filepath_dev:
                    hateeval_dev_rec['text'].append(','.join(tokens[1:-3]))
                    hateeval_dev_rec['label'].append(int(tokens[-3]))




    with codecs.open(heirarchical_path+'.test', 'w', 'utf-8') as outfile:
        for i, text in enumerate(hateeval_test_rec['text']):
            outfile.write(str(text) + '\t' + str(hateeval_test_rec['label'][i]) + '\n')

    with codecs.open(heirarchical_path+'.dev', 'w', 'utf-8') as outfile:
        for i, text in enumerate(hateeval_dev_rec['text']):
            outfile.write(str(text) + '\t' + str(hateeval_dev_rec['label'][i]) + '\n')

    with codecs.open(heirarchical_path+'.train', 'w', 'utf-8') as outfile:
        for i, text in enumerate(hateeval_train_rec['text']):
            outfile.write(str(text) + '\t' + str(hateeval_train_rec['label'][i]) + '\n')

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'train.txt'), 'w', 'utf-8') as outfile:
        for i, text in enumerate(hateeval_train_rec['text']):
            outfile.write(str(text) + '\t' + str(hateeval_train_rec['label'][i]) + '\n')

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'dev.txt'), 'w', 'utf-8') as outfile:
        for i, text in enumerate(hateeval_dev_rec['text']):
            outfile.write(str(text) + '\t' + str(hateeval_dev_rec['label'][i]) + '\n')

    with codecs.open(os.path.join(os.path.dirname(heirarchical_path), 'test.txt'), 'w', 'utf-8') as outfile:
        for i, text in enumerate(hateeval_test_rec['text']):
            outfile.write(str(text) + '\t' + str(hateeval_test_rec['label'][i]) + '\n')




if __name__ == '__main__':
    main()