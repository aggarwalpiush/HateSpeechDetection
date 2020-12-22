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

args = get_args()

arc_obj = Arc_preprocessor()

def load_tab_data(filename = "../processed_data/en/davidson/train.txt", preprocessed=True):
    data = pd.read_csv(filename, sep='\t', header=0,
                       names=['comment', 'isHate'])
    data = shuffle(data)

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
            X.append(' '.join(arc_obj.tokenizeRawTweetText(str(x))))
        else:
            X.append(x)
    return numpy.array(X), numpy.array(y)

def pred_f1(model_filename, X_test, y_test):
    loaded_model = pickle.load(open(model_filename, 'rb'))
    a = time.time()
    y_preds = loaded_model.predict(X_test)
    score_time = time.time() - a
    f1 = f1_score(y_test, y_preds, average='macro')
    return f1, score_time, y_preds



