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
    parser.add_argument('--train_data', type=str, default='../processed_data/en/davidson/obfuscated_random_voweldrop_train.txt')
    parser.add_argument('--obfuscated_data_prefix', type=str, default='../processed_data/en/davidson/obfuscated_')
    parser.add_argument('--random_ngram', type=int, default=1)
    parser.add_argument('--dict_file', type=str, default='../dictionaries/hurtlex_lex.txt')
    parser.add_argument('--is_hatebase', type=bool, default=False)
    parser.add_argument('--hier_soc_file', type=str, default='../hierarchical_dict/soc.davidson_demo.txt')
    parser.add_argument('--hier_soc_ngram', type=int, default=3)
    parser.add_argument('--hier_soc_thld', type=int, default=-0.5)
    parser.add_argument('--span', type=str, default='random', choices=['original', 'random', 'random_POS', 'all',
                                                                    'dictionary', 'hierarchical'])
    parser.add_argument('--obfuscation_strategy', type=str, default='camelcasing', choices=['original', 'camelcasing',
                        'snakecasing', 'spacing', 'voweldrop', 'random_masking', 'spelling', 'leetspeak', 'mathspeak',
                        'reversal', 'firstCharacter'])
    parser.add_argument('--dev_data', type=str, default='../processed_data/en/davidson/dev.txt')
    parser.add_argument('--test_data', type=str, default='../processed_data/en/davidson/test.txt')
    parser.add_argument('--vec_scheme', type=str, default='fasttext', choices=['tfidf', 'fasttext', 'glove'])
    parser.add_argument('--max_features', type=int, default=50000)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--model_path', type=str, default='../models/GradB_fasttext_original_original.pkl')
    parser.add_argument('--evaluate_label_path', type=str, default='../results/GradB_fasttext_davidson_original_original.txt')
    parser.add_argument('--original_data', type=str, default='../processed_data/en/davidson/test.txt')




    args = parser.parse_args()
    return args


args = get_args()
