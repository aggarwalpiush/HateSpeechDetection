#! usr/bin/env python
# -*- coding : utf-8 -*-


import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from get_embedding import MeanEmbeddingTransformer
import numpy
from args import get_args
import time
from utils import load_tab_data, pred_f1
import pickle
import logging

logging.basicConfig(filename='../logs/classification_results.log', filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  datefmt='%H:%M:%S',
                    level=logging.DEBUG)
args = get_args()


def fit_train_save(pipe,parameters,X_train0, y_train0, model_name):
    model_filename = '../models/'+str(model_name)+'_'+str(args.vec_scheme)+'_'+\
                     str(os.path.basename(os.path.dirname(args.train_data))) + '.pkl'
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    clf = GridSearchCV(estimator=pipe, param_grid=parameters, cv=inner_cv, n_jobs=54, verbose=1, scoring='f1')
    a = time.time()
    clf.fit(X_train0, y_train0)
    fit_time = time.time() - a
    pickle.dump(clf, open(model_filename, 'wb'))
    return fit_time


X_train, y_train = load_tab_data(filename=args.train_data, preprocessed=True)

X_dev, y_dev = load_tab_data(filename=args.dev_data, preprocessed=True)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Fasttext Embeddings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


if args.use_de_tokenizer:
    EMBEDDING_PATH = "../embeddings/fasttext_german_twitter_100d.vec"  # FastText embeddings for German
else:
    EMBEDDING_PATH = "../embeddings/twitter_fasttext.vec" #FastText embeddings for english
if not os.path.exists(EMBEDDING_PATH +'.tmp'):
    logging.info('Embedding file pruning')
    MeanEmbeddingTransformer(EMBEDDING_PATH).generate_temp_embfile(X_train)


if args.vec_scheme == 'TF_IDF':
    vec = TfidfVectorizer(analyzer='word')
elif args.vec_scheme == 'fasttext':
    vec = MeanEmbeddingTransformer(EMBEDDING_PATH +'.tmp')


'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run GradBoost
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

grad = GradientBoostingClassifier()


pipe = Pipeline(steps=[('vec', vec), ('grad', grad)])
parameters = [{
    'grad__learning_rate':[0.0001,0.01,0.1,0.5,1],
    'grad__n_estimators':[10,50,100,300],
    'grad__subsample':[0.7,0.85,1],
    'grad__max_features':['sqrt','log2',None]
}]

fit_time = fit_train_save(pipe, parameters, numpy.append(X_train, X_dev), numpy.append(y_train, y_dev), "GradB")
logging.info("GradB Model")
logging.info("dataset name: %s", args.train_data)
logging.info("Fit time : %s", fit_time)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run Logistic Regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1)
pipe = Pipeline(steps=[('vec', vec), ('log', log)])
parameters = [{
    'log__C':[0.5,1,3,5,10,1000],
    'log__solver':['newton-cg', 'lbfgs', 'sag'],
    'log__penalty':['l2']
},{
    'log__C':[0.5,1,3,5,10,1000],
    'log__solver':['saga'],
    'log__penalty':['l1']
}]
fit_time = fit_train_save(pipe, parameters, numpy.append(X_train, X_dev), numpy.append(y_train, y_dev), "LogReg")
logging.info("LogReg Model")
logging.info("dataset name: %s", args.train_data)
logging.info("Fit time : %s", fit_time)
'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run SVM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
svm = SVC(random_state=0)
pipe = Pipeline(steps=[('vec', vec), ('svm', svm)])
parameters = [{
    'svm__kernel':['rbf'],
    'svm__C':[0.25, 0.5, 1, 3, 5, 10, 100, 1000],
    'svm__gamma':[0.05, 0.1, 0.5, 0.9, 1]
},{
    'svm__kernel':['linear'],
    'svm__C':[0.25, 0.5, 1, 3, 5, 10, 100, 1000]
}]

fit_time = fit_train_save(pipe, parameters, numpy.append(X_train, X_dev), numpy.append(y_train, y_dev), "svm")
logging.info("SVM Model")
logging.info("dataset name: %s", args.train_data)
logging.info("Fit time : %s", fit_time)
'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run RandomForest
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
randFor = RandomForestClassifier(random_state=0,n_jobs=-1)
vec = MeanEmbeddingTransformer(EMBEDDING_PATH +'.tmp')
#vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('randFor', randFor)])
parameters = [{
    'randFor__max_depth':[1,10,50,100,200],
    'randFor__max_features':['sqrt','log2',None],
    'randFor__bootstrap':[True,False],
    'randFor__n_estimators': [10,100,500,1000]
}]
fit_time = fit_train_save(pipe, parameters, numpy.append(X_train, X_dev), numpy.append(y_train, y_dev), "randforest")
logging.info("Randforest Model")
logging.info("dataset name: %s", args.train_data)
logging.info("Fit time : %s", fit_time)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run AdaBoost
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ada = AdaBoostClassifier()
pipe = Pipeline(steps=[('vec', vec), ('ada', ada)])
parameters = [{
   # 'vec__ngram_range':[(1,1),(1,2),(1,5)],
   # 'vec__max_features':[5000,10000,50000,100000],
   # 'vec__stop_words':['english', None],
    'ada__base_estimator':[None,DecisionTreeClassifier(max_depth=10),LogisticRegression(C=100)],
    'ada__n_estimators':[10,50,100,300],
    'ada__learning_rate':[0.0001,0.01,0.5,1]
}]
fit_time = fit_train_save(pipe, parameters, numpy.append(X_train, X_dev), numpy.append(y_train, y_dev), "AdaB")
logging.info("AdaBoost Model")
logging.info("dataset name: %s", args.train_data)
logging.info("Fit time : %s", fit_time)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run GradBoost
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
grad = GradientBoostingClassifier()
pipe = Pipeline(steps=[('vec', vec), ('grad', grad)])
parameters = [{
   # 'vec__ngram_range':[(1,1),(1,2),(1,5)],
   # 'vec__max_features':[5000,10000,50000,100000],
   # 'vec__stop_words':['english', None],
    'grad__learning_rate':[0.0001,0.01,0.1,0.5,1],
    'grad__n_estimators':[10,50,100,300],
    'grad__subsample':[0.7,0.85,1],
    'grad__max_features':['sqrt','log2',None]
}]
fit_time = fit_train_save(pipe, parameters, numpy.append(X_train, X_dev), numpy.append(y_train, y_dev), "grad")
logging.info("GradBoost Model")
logging.info("dataset name: %s", args.train_data)
logging.info("Fit time : %s", fit_time)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run Bagging
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
bag = BaggingClassifier(n_jobs=-1)
pipe = Pipeline(steps=[('vec', vec), ('bag', bag)])
parameters = [{
#    'vec__ngram_range':[(1,1),(1,2),(1,5)],
 #   'vec__max_features':[5000,10000,50000,100000],
  #  'vec__stop_words':['english', None],
    'bag__base_estimator':[None,DecisionTreeClassifier(max_depth=10),LogisticRegression(C=100)],
    'bag__n_estimators':[10,50,100,300],
    'bag__max_samples':[0.7,0.85,1],
    'bag__max_features':[0.5,0.75,1],
    'bag__bootstrap':[True,False]
}]
fit_time = fit_train_save(pipe, parameters, numpy.append(X_train, X_dev), numpy.append(y_train, y_dev), "bag")
logging.info("Bagging Model")
logging.info("dataset name: %s", args.train_data)
logging.info("Fit time : %s", fit_time)
'''





