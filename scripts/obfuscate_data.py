#! /usr/bin/env python
# -*- coding : utf-8 -*-

from get_obfuscated_text import Select_span, Obfuscation_strategies
import codecs
import pandas as pd
import argparse
from args import get_args
import os
from nltk.tokenize import TweetTokenizer

args = get_args()


def get_dataset(data_file):
    df = pd.read_csv(data_file, sep="\t", names=['tweet', 'label'])
    return df


def put_dataset(obfuscated_path, data):
    with codecs.open(obfuscated_path, 'w', 'utf-8') as ob_obj:
        for line in data:
            ob_obj.write(line+"\n")

OBFUSCATED_SPAN = [ 'random', 'random_POS', 'all', 'dictionary', 'hierarchical']

#OBFUSCATED_SPAN = ['hierarchical']

OBFUSCATED_STRATEGY = [ 'camelcasing',
                        'snakecasing', 'spacing', 'voweldrop', 'random_masking', 'spelling', 'leetspeak', 'mathspeak',
                        'reversal', 'firstCharacter']

#OBFUSCATED_STRATEGY = ['spelling', 'reversal', 'firstCharacter']

from shutil import copyfile
def main():
    # select dataset
    original_dataset = get_dataset(args.original_data)
    copyfile(args.original_data, os.path.join(os.path.dirname(args.original_data),
                                     os.path.basename(os.path.dirname(args.original_data)) + '_test_dataoriginal_original_obfuscated.txt'))

    for each_span in OBFUSCATED_SPAN:
        for each_strategy in OBFUSCATED_STRATEGY:
            obfuscated_dataset = []
            for index, rows in original_dataset.iterrows():
                # choose span from the input statement
                ss = Select_span(rows['tweet'], random_ngram=args.random_ngram, dict_file=args.dict_file,
                                 is_hatebase=args.is_hatebase, hier_soc_file=args.hier_soc_file,
                                 hier_soc_ngram=args.hier_soc_ngram, hier_soc_thld=args.hier_soc_thld)
                # select random span
                try:
                    span = ss.function_mapping[each_span](ss)
                    # select obfuscation strategy
                    obs = Obfuscation_strategies(span)
                    obfuscated_statement = str(rows['tweet'])
                    if isinstance(span, str):
                        span = [span]
                    for i, entity in enumerate(span):
                        #print(entity)
                        #print(obs.function_mapping[each_strategy](obs)[i])
                        if entity in TweetTokenizer().tokenize(obfuscated_statement):
                            obfuscated_statement = obfuscated_statement.replace(entity, obs.function_mapping[each_strategy](obs)[i])
                    obfuscated_dataset.append(obfuscated_statement + "\t" + str(rows['label']))
                except (IndexError):
                    obfuscated_dataset.append(rows['tweet']+ "\t" + str(rows['label']))
            put_dataset(os.path.join(os.path.dirname(args.original_data),
                                     os.path.basename(os.path.dirname(args.original_data)) +
                                     '_test_data' + '_'.join([each_span, each_strategy]) + '_obfuscated.txt'),
                            obfuscated_dataset)

if __name__ == '__main__':
    main()





