#! usr/bin/env python
# -*- coding : utf-8 -*-


import args
import codecs
from utils import load_tab_data, pred_f1
from args import get_args

args = get_args()


X_test, y_test = load_tab_data(filename=args.test_data, preprocessed=True)

f1, score_time = pred_f1(args.model_path, X_test, y_test)

