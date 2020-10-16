"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################################
#                           Set A                                   #
#####################################################################
# In this set of experiments we will try logistic regression, svms, #
# ridge, decision trees, naive bayes and random forests classifiers #
# across a wide variety of parameters for each algorithm and test   #
# them via nested cross validation method.                          #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from utilities.preprocess import Preproccesor
from utilities.get_embedding import MeanEmbeddingTransformer
import time

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tn+fp)>0:
        speci = tn/(tn+fp)
        return speci
    return 0
def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tp+fn)>0:
        sensi = tp/(tp+fn)
        return sensi
    return 0

def nested_cross_val(pipe,parameters,X,y,name):
    scores = {}
    scores.setdefault('fit_time', [])
    scores.setdefault('score_time', [])
    scores.setdefault('test_F1', [])
    scores.setdefault('test_Precision', [])
    scores.setdefault('test_Recall', [])
    scores.setdefault('test_Accuracy', [])
    scores.setdefault('test_Specificity', [])
    scores.setdefault('test_Sensitivity', [])

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)
    splits = outer_cv.split(X)
    for train_index, test_index in splits:
        X_trainO, X_testO = X[train_index], X[test_index]
        y_trainO, y_testO = y[train_index], y[test_index]
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
        clf = GridSearchCV(estimator=pipe, param_grid=parameters, cv=inner_cv, n_jobs=54, verbose=1, scoring='f1')
        a = time.time()
        clf.fit(X_trainO,y_trainO)
        fit_time = time.time() - a
        a = time.time()
        y_preds = clf.predict(X_testO)
        score_time = time.time() - a
        scores['fit_time'].append(fit_time)
        scores['score_time'].append(score_time)
        scores['test_F1'].append(f1_score(y_testO,y_preds,average='macro'))
        scores['test_Precision'].append(precision_score(y_testO,y_preds,average='macro'))
        scores['test_Recall'].append(recall_score(y_testO,y_preds,average='macro'))
        scores['test_Accuracy'].append(accuracy_score(y_testO,y_preds))
        scores['test_Specificity'].append(specificity(y_testO,y_preds))
        scores['test_Sensitivity'].append(sensitivity(y_testO,y_preds))

    for k in scores:
        print(str(name)+" "+str(k)+": "+str(sum(scores[k])/10))
    f = open("../results/setA_twitter_fasttext_davidson.txt", "a+")
    f.write("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format(str(name)[:7],
        str('%.4f' % (sum(scores['fit_time'])/10)), str('%.4f' % (sum(scores['score_time'])/10)), str('%.4f' % (sum(scores['test_F1'])/10)),
        str('%.4f' % (sum(scores['test_Precision'])/10)), str('%.4f' % (sum(scores['test_Recall'])/10)), str('%.4f' % (sum(scores['test_Accuracy'])/10)),
        str('%.4f' % (sum(scores['test_Specificity'])/10)), str('%.4f' % (sum(scores['test_Sensitivity'])/10))))
    f.close()
    print ("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7}".format(str(name)[:7],
        str('%.4f' % (sum(scores['fit_time'])/10)), str('%.4f' % (sum(scores['score_time'])/10)), str('%.4f' % (sum(scores['test_F1'])/10)),
        str('%.4f' % (sum(scores['test_Precision'])/10)), str('%.4f' % (sum(scores['test_Recall'])/10)), str('%.4f' % (sum(scores['test_Accuracy'])/10)),
        str('%.4f' % (sum(scores['test_Specificity'])/10)), str('%.4f' % (sum(scores['test_Sensitivity'])/10))))


X, y = Preproccesor.load_data(True)
f = open("../results/setA_twitter_fasttext_davidson.txt", "w+")
f.write("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format('Method','fitTime','scoreTi','F1score','Precisi','Recall','Accurac','Specifi','Sensiti'))
f.write("=========================================================================\n")
f.close()
print("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7}".format('Method','fitTime','scoreTi','F1score','Precisi','Recall','Accurac','Specifi','Sensiti'))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Fasttext Embeddings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


EMBEDDING_PATH = "../twitter_ft_davidson.vec" #FastText
if not os.path.exists(EMBEDDING_PATH +'.tmp'):
	print('Embedding file pruning')
	MeanEmbeddingTransformer(EMBEDDING_PATH).generate_temp_embfile(X)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run GradBoost
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
grad = GradientBoostingClassifier()
#vec = TfidfVectorizer(analyzer='word')
vec = MeanEmbeddingTransformer(EMBEDDING_PATH +'.tmp')
pipe = Pipeline(steps=[('vec', vec), ('grad', grad)])
parameters = [{
    'grad__learning_rate':[0.0001,0.01,0.1,0.5,1],
    'grad__n_estimators':[10,50,100,300],
    'grad__subsample':[0.7,0.85,1],
    'grad__max_features':['sqrt','log2',None]
}]
nested_cross_val(pipe, parameters, X, y, "GradB")



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run Logistic Regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
log = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1)
#vec = TfidfVectorizer(analyzer='word')
vec = MeanEmbeddingTransformer(EMBEDDING_PATH +'.tmp')
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
nested_cross_val(pipe, parameters, X, y, "LogReg")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run SVM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
svm = SVC(random_state=0)
#vec = TfidfVectorizer(analyzer='word')
vec = MeanEmbeddingTransformer(EMBEDDING_PATH +'.tmp')
pipe = Pipeline(steps=[('vec', vec), ('svm', svm)])
parameters = [{
    'svm__kernel':['rbf'],
    'svm__C':[0.25, 0.5, 1, 3, 5, 10, 100, 1000],
    'svm__gamma':[0.05, 0.1, 0.5, 0.9, 1]
},{
    'svm__kernel':['linear'],
    'svm__C':[0.25, 0.5, 1, 3, 5, 10, 100, 1000]
}]
nested_cross_val(pipe, parameters, X, y, "SVM")

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
nested_cross_val(pipe, parameters, X, y, "RandomForest")
