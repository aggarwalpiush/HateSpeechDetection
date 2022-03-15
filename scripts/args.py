from argparse import ArgumentParser
import os

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise

def get_args():
    parser = ArgumentParser(description='choose your training data to obfuscate')

    # model parameters
    parser.add_argument('--train_data', type=str, default='../processed_data/en/hateeval2019/train.txt')
    parser.add_argument('--obfuscated_data_prefix', type=str, default='../processed_data/en/hateeval2019/obfuscated_')
    parser.add_argument('--random_ngram', type=int, default=1)
    parser.add_argument('--dict_file', type=str, default='../dictionaries/hurtlex_lex.txt')
    parser.add_argument('--manual_dict_file', type=str,
                        default='../manual_profane_ranked_dictionary/manually_selected_lexicons_stem.tsv')
    parser.add_argument('--is_hatebase', type=bool, default=False)
    parser.add_argument('--hier_soc_file', type=str, default='../hierarchical_dict/soc.hateeval2019_demo.txt')
    parser.add_argument('--hier_soc_ngram', type=int, default=1)
    parser.add_argument('--hier_soc_thld', type=float, default=-0.7)
    parser.add_argument('--span', type=str, default='random', choices=['original', 'random', 'random_POS', 'all',
                                                                    'dictionary', 'hierarchical', 'manual_dict'])
    parser.add_argument('--obfuscation_strategy', type=str, default='camelcasing', choices=['original', 'camelcasing',
                        'snakecasing', 'spacing', 'voweldrop', 'random_masking', 'spelling', 'leetspeak', 'mathspeak',
                        'reversal', 'firstCharacter', 'phonetic', 'charcaterdrop', 'kebabcasing', 'diacritics'])
    parser.add_argument('--dev_data', type=str, default='../processed_data/en/hateeval2019/dev.txt')
    parser.add_argument('--test_data', type=str, default='../processed_data/en/hateeval2019/test.txt')
    parser.add_argument('--test_path', type=str, default='../processed_data/en/hateeval2019')
    parser.add_argument('--vec_scheme', type=str, default='fasttext', choices=['tfidf', 'fasttext', 'glove'])
    parser.add_argument('--max_features', type=int, default=50000)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--evaluate_label_path', type=str, default='../results')
    parser.add_argument('--original_data', type=str, default='../processed_data/en/hateeval2019/train.txt')
    parser.add_argument('--only_test', type=bool, default=False)
    parser.add_argument('--use_de_tokenizer', type=bool, default=False)




    args = parser.parse_args()
    return args


args = get_args()
