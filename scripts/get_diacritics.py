#! /usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
from args import get_args


args = get_args()



def get_diacritics_dict(filename):
    small_diacritic_dict = []
    caps_diacritic_dict = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line = line.replace('\r\n', '').strip()
            tokens = line.split('\t')
            if tokens[0].lower() not in small_diacritic_dict.keys():
                small_diacritic_dict[tokens[0].lower()] = []
                caps_diacritic_dict[tokens[0]] = []
            small_diacritic_dict[tokens[0].lower()].append(tokens[1].split(' ')[1])
            caps_diacritic_dict[tokens[0]].append(tokens[1].split(' ')[0])
    for k in caps_diacritic_dict.keys():
        caps_diacritic_dict[k] = set(caps_diacritic_dict[k])
        small_diacritic_dict[k.lower()] = set(small_diacritic_dict[k.lower()])
    return small_diacritic_dict, caps_diacritic_dict


if __name__ == "__main__":
    get_diacritics_dict("../dictionaries/diacritics.txt")