#! usr/bin/env python
# -*- coding : utf-8 -*-

from args import get_args
import logging
import pickle
from utils import load_tab_data
import codecs
from glob import glob
import os
import time
from sklearn.metrics import confusion_matrix, f1_score, classification_report


logging.basicConfig(filename='../logs/evaluate_results.log', filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.DEBUG)
args = get_args()


def main():
    for model in glob(os.path.join(args.model_path, '*.pkl')):
        for test_file in glob(os.path.join(args.test_path, '*test_data*obfuscated.txt')):
            X_test, y_test = load_tab_data(filename=test_file, preprocessed=True, test_file=True)

            loaded_model = pickle.load(open(model, 'rb'))
            a = time.time()
            y_preds = loaded_model.predict(X_test)
            score_time = time.time() - a
            result_file = os.path.join(args.evaluate_label_path, os.path.basename(test_file).replace('.txt', '') + '_' +
                    os.path.basename(args.model_path).replace('.pkl', '')+'_predictions.txt')

            with codecs.open(result_file, 'w', 'utf-8') as result_obj:
                for i, val in enumerate(X_test):
                    result_obj.write(str(val) + '\t' + str(y_preds[i]) + '\n')

            y_preds = []
            for i in y_preds:
                if i >= 0.5:
                    y_preds.append(1)
                else:
                    y_preds.append(0)
            logging.info("=================================START====================================")
            logging.info("Model name: %s", os.path.basename(args.model_path).replace('.pkl', ''))
            logging.info("Test file : %s", result_file)
            logging.info("F1 Score: %s", f1_score(y_test, y_preds))
            logging.info("Score_time : %s", score_time)
            logging.info("Confusion Matrix : %s", confusion_matrix(y_test, y_preds))
            target_names = ['Non-hate', 'hate']
            logging.info("Classification Report : %s", classification_report(y_test, y_preds, target_names=target_names))
            logging.info("=================================END====================================\n\n")



if __name__ == '__main__':
    main()





