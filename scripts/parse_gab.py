#! usr/bin/env python
# -*- coding:utf-8 -*-


import codecs
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main():
    in_filepath = "../raw_data/en/gab/GabHateCorpus_annotations.tsv"
    heirarchical_path = "../processed_data/en/gab/GabHateCorpus_annotations_hier.txt"

    gab_record = {}
    gab_text = {}

    with codecs.open(in_filepath, 'r', 'utf-8') as in_obj:
        for i, line in enumerate(in_obj):
            if i == 0:
                continue
            tokens = line.replace('\n', '').replace('\r', '').split('\t')
            tokens = tokens[:3] + [tok.strip().replace('', str(0.0)) if tok.strip() == '' else tok.strip() for tok in tokens[3:]]
            if tokens[0] not in gab_record.keys():
                gab_record[tokens[0]] = []
                gab_record[tokens[0]].append(tokens[3:])
                gab_text[tokens[0]] = tokens[2]
            else:
                gab_record[tokens[0]].append(tokens[3:])
        for key, values in gab_record.items():
            gab_record[key] = np.transpose(np.array(values))
            final_val = []
            for val in gab_record[key]:
                val = [float(each_val) for each_val in val]
                final_val.append(np.bincount(val).argmax())
            gab_record[key] = final_val

    json_entry = []
    for key, val in gab_record.items():
        required_format = {}
        required_format["text_id"] = int(key)
        required_format["Text"] = gab_text[key]
        required_format["hd"] = val[1]
        required_format["cv"] = val[2]
        required_format["vo"] = val[3]
        required_format["rel"] = val[4]
        required_format["rae"] = val[5]
        required_format["sxo"] = val[6]
        required_format["gen"] = val[7]
        required_format["idl"] = val[8]
        required_format["nat"] = val[9]
        required_format["pol"] = val[10]
        required_format["mph"] = val[11]
        required_format["ex"] = val[12]
        required_format["im"] = val[13]
        json_entry.append(required_format)

    # converting dict to json and to jsonl

    # with codecs.open(output_path, 'w', 'utf-8') as outfile:
    #     for entry in json_entry:
    #         outfile.write(str(entry))
    #         #json.dump(entry, outfile)
    #         outfile.write('\n')


    with codecs.open(heirarchical_path, 'w', 'utf-8') as outfile:
        for key, val in gab_record.items():
            if any(val):
                outfile.write(str(gab_text[key]) + '\t'+ '1\n')
            else:
                outfile.write(str(gab_text[key]) + '\t' + '0\n')


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

