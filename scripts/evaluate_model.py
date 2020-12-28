#! usr/bin/env python
# -*- coding : utf-8 -*-

from args import get_args
import logging
import pickle
import time
from utils import load_tab_data
import codecs
import numpy as np


logging.basicConfig(filename='../logs/evaluate_results.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  datefmt='%H:%M:%S', level=logging.DEBUG)
args = get_args()


def main():
    X_test, y_test = load_tab_data(filename=args.test_data, preprocessed=True)

    loaded_model = pickle.load(open(args.model_path, 'rb'))
    a = time.time()
    y_preds = loaded_model.predict(X_test)

    with codecs.open(args.evaluate_label_path, 'w', 'utf-8') as result_obj:
<<<<<<< HEAD
        result_obj.write(np.append(X_test, np.array(y_preds)))
=======
        for i, val in enumerate(X_test):
            result_obj.write(str(val) + '\t' + str(y_preds[i]) + '\n')
>>>>>>> 16bbc7b5e2039fe6fe800f05725f086ff672f28c

if __name__ == '__main__':
    main()





