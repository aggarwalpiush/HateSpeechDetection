#! /usr/bin/env python
# -*- coding : utf-8 -*-

from get_obfuscated_text import Select_span, Obfuscation_strategies
import codecs
import pandas as pd
from args import get_args
import os
from nltk.tokenize import TweetTokenizer
from somajo import SoMaJo
from global_variables import OBFUSCATED_SPAN, OBFUSCATED_STRATEGY
import nltk
from nltk.stem import PorterStemmer

import spacy
nlp_en = spacy.load("en_core_web_sm")




ps = PorterStemmer()


args = get_args()

de_tokenizer = SoMaJo("de_CMC")


def get_dataset(data_file):
    df = pd.read_csv(data_file, sep="\t", names=['tweet', 'label'])
    return df


def put_dataset(obfuscated_path, data):
    with codecs.open(obfuscated_path, 'w', 'utf-8') as ob_obj:
        for line in data:
            ob_obj.write(line+"\n")

OBFUSCATED_SPAN = [ 'random', 'random_POS', 'all', 'dictionary', 'hierarchical', 'manual_dict']

#OBFUSCATED_SPAN = [ 'random_POS', 'all', 'dictionary', 'hierarchical']
#OBFUSCATED_SPAN = ['dictionary']
OBFUSCATED_SPAN = ['manual_dict']

OBFUSCATED_STRATEGY = [ 'camelcasing',
                        'snakecasing', 'spacing', 'voweldrop', 'random_masking', 'spelling', 'leetspeak', 'mathspeak',
                        'reversal', 'firstCharacter', 'phonetic', 'charcaterdrop', 'kebabcasing', 'diacritics']

#OBFUSCATED_STRATEGY = ['spacing', 'random_masking', 'leetspeak', 'mathspeak', 'dicritics']

#OBFUSCATED_STRATEGY = [ 'spelling']

from shutil import copyfile
def main():
    # select dataset
    original_dataset = get_dataset(args.original_data)
    copyfile(args.original_data, os.path.join(os.path.dirname(args.original_data),
                                     os.path.basename(os.path.dirname(args.original_data)) + '_dev_dataoriginal_original_obfuscated.txt'))

    for each_span in OBFUSCATED_SPAN:
        for each_strategy in OBFUSCATED_STRATEGY:
            obfuscated_dataset = []
            for index, rows in original_dataset.iterrows():
                # choose span from the input statement
                #print(rows['tweet'])
                ss = Select_span(rows['tweet'], random_ngram=args.random_ngram, dict_file=args.dict_file,
                                 is_hatebase=args.is_hatebase, hier_soc_file=args.hier_soc_file,
                                 manual_gen_lexicon=args.manual_dict_file,
                                 hier_soc_ngram=args.hier_soc_ngram, hier_soc_thld=args.hier_soc_thld)
                # select random span
                try:
                    span = ss.function_mapping[each_span](ss)
                    #print(span)
                    # select obfuscation strategy
                    obs = Obfuscation_strategies(span)
                    obfuscated_statement = str(rows['tweet'])
                    if isinstance(span, str):
                        span = [span]
                    for i, entity in enumerate(span):
                        #print(entity)
                        #print(obs.function_mapping[each_strategy](obs)[i])
                        if args.use_de_tokenizer:
                            all_entities = []
                            sentences = de_tokenizer.tokenize_text([obfuscated_statement])
                            for sentence in sentences:
                                for token in sentence:
                                    all_entities.append(token.text)
                            if entity in all_entities:
                                obfuscated_statement = obfuscated_statement.replace(entity,  obs.function_mapping[each_strategy](obs)[i])
                        else:
                            tokenized_statement = nlp_en(obfuscated_statement)
                            stemmed_statement = [ps.stem(x.text.lower()) for x in tokenized_statement]
                            if entity in stemmed_statement:
                                #print("entity %s found in obfuscatedd_statement: %s" % (entity, obfuscated_statement))
                                #print(stemmed_statement)
                                #print(tokenized_statement[stemmed_statement.index(entity)])
                                #print(obs.function_mapping[each_strategy](obs)[i])
                                obfuscated_statement = obfuscated_statement.replace(tokenized_statement[stemmed_statement.index(entity)].text, obs.function_mapping[each_strategy](obs)[i])
                                #print(obfuscated_statement)
                                #print('====================')
                            else:
                                print("entity %s not found in obfuscatedd_statement: %s" %(entity, obfuscated_statement))

                    obfuscated_dataset.append(obfuscated_statement + "\t" + str(rows['label']))
                except (IndexError):
                    #print('got exception')
                    #print(rows['tweet'])
                    obfuscated_dataset.append(rows['tweet']+ "\t" + str(rows['label']))
            put_dataset(os.path.join(os.path.dirname(args.original_data),
                                     os.path.basename(os.path.dirname(args.original_data)) +
                                     '_dev_data' + '_'.join([each_span, each_strategy]) + '_obfuscated.txt'),
                            obfuscated_dataset)

if __name__ == '__main__':
    main()





