#! /usr/bin/env python
# -*- coding : utf-8 -*-

from get_obfuscated_text import Select_span, Obfuscation_strategies
import codecs
import pandas as pd
import argparse
from args import get_args

args = get_args()


def get_dataset(training_file):
    df = pd.read_csv(training_file, sep="\t", names=['tweet', 'label'])
    return df


def put_dataset(obfuscated_path, data):
    with codecs.open(obfuscated_path, 'w', 'utf-8') as ob_obj:
        for line in data:
            ob_obj.write(line+"\n")


def main():
    # select dataset
    original_dataset = get_dataset(args.train_data)
    obfuscated_dataset = []
    for index, rows in original_dataset.iterrows():
        print(rows['tweet'])
        # choose span from the input statement
        ss = Select_span(rows['tweet'], random_ngram=args.random_ngram, dict_file=args.dict_file,
                         is_hatebase=args.is_hatebase, hier_soc_file=args.hier_soc_file,
                         hier_soc_ngram=args.hier_soc_ngram, hier_soc_thld=args.hier_soc_thld)
        # select random span
        try:
            span = ss.function_mapping[args.span](ss)
            print(span+'\n\n')
            # select obfuscation strategy

            os = Obfuscation_strategies(span)

            #print(rows['tweet'].replace(span, os.function_mapping[args.obfuscation_strategy](os)))
            obfuscated_dataset.append(rows['tweet'].replace(span, os.function_mapping[args.obfuscation_strategy](os))+ "\t" + str(rows['label']))
        except IndexError:
            obfuscated_dataset.append(rows['tweet'])
    put_dataset(args.obfuscated_data_prefix + '_'.join([args.span, args.obfuscation_strategy]) + '_train.txt',
                    obfuscated_dataset)


if __name__ == '__main__':
    main()





