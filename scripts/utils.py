#! usr?bin/env python
# -*- coding : utf-8 -*-

from args import get_args
import pandas as pd
from arc_preprocessor import Arc_preprocessor
from sklearn.utils import shuffle
from sklearn.metrics import  f1_score
import numpy
import pickle
import time
from somajo import SoMaJo

de_tokenizer = SoMaJo("de_CMC")

args = get_args()

arc_obj = Arc_preprocessor()


def load_tab_data(filename = "../processed_data/en/davidson/train.txt", preprocessed=True, test_file=False):
    data = pd.read_csv(filename, sep='\t', header=0,
                       names=['comment', 'isHate'])
    print(len(data.index))
    if not test_file:
        data = shuffle(data)

   # f = open('missing.txt', 'w')
    XT = data['comment'].values
    X = []
    yT = data['isHate'].values
    y = []
    for yt in yT:
        if yt == 1:
            y.append(int(1))
        else:
            y.append(int(0))
    for x in XT:
        if preprocessed:
            if args.use_de_tokenizer:
                tweet_tokens = []
                sentences = de_tokenizer.tokenize_text([str(x)])
                for sentence in sentences:
                    for token in sentence:
                        tweet_tokens.append(token.text)
            #    f.write(' '.join(tweet_tokens) + '\n')
                X.append(' '.join(tweet_tokens))
            else:
                X.append(' '.join(arc_obj.tokenizeRawTweetText(str(x))))
        else:
            X.append(x)
   # f.close()
    return numpy.array(X), numpy.array(y)

def pred_f1(model_filename, X_test, y_test):
    loaded_model = pickle.load(open(model_filename, 'rb'))
    a = time.time()
    y_preds = loaded_model.predict(X_test)
    score_time = time.time() - a
    f1 = f1_score(y_test, y_preds, average='macro')
    return f1, score_time, y_preds



