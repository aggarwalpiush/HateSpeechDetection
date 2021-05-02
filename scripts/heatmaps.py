#! usr/bin/env python
# -*- coding : utf-8 -*-

import numpy as np
np.random.seed(0)
import os
import seaborn as sns
import matplotlib. pyplot as plt
sns.set(font_scale = 0.6)
#sns.set_theme()
#uniform_data = np.random.rand(10, 12)
#print(uniform_data)

#sns.heatmap(uniform_data)
#plt.show()

OBFUSCATED_SPAN = [ 'random', 'random_POS', 'all', 'dictionary', 'hierarchical']


OBFUSCATED_STRATEGY = [ 'camelcasing',
                        'snakecasing', 'spacing', 'voweldrop', 'random_masking', 'spelling', 'leetspeak', 'mathspeak',
                        'reversal', 'firstCharacter']

METHOD = ['bert', 'GradB', 'svm', 'LogReg', 'cnn_attention', 'cnn_lstm', 'txt_lstm',  'bilstm']

ORIGINAL_RESULT_DAVIDSON = ['0.4512820512820513', '0.2532299741602067', '0.3190883190883191', '0.1672473867595819', '0.32',
                   '0.12549019607843137', '0.2981366459627329', '0.26198083067092653']

ORIGINAL_RESULT_WASEEM = ['0.7869249394673123', '0.6398929049531459', '0.6702127659574468', '0.6193724420190997', '0.7178899082568807',
                   '0.7087628865979383', '0.726161369193154', '0.7387606318347508']

import codecs

def parse_evaluate_log(filename):
    evaluate_dict = {}
    for obs in OBFUSCATED_SPAN:
        evaluate_dict[obs] = []
    with codecs.open(filename, 'r', 'utf-8') as eval_obj:
        for i, line in enumerate(eval_obj):
            for obs in OBFUSCATED_SPAN:
                if obs in line:
                    if obs == 'random' and 'POS' in line:
                        continue
                    for st in OBFUSCATED_STRATEGY:
                        if st in line:
                            for clas in METHOD:
                                f = open(filename)
                                f1lines = f.readlines()
                                if clas in line or clas in f1lines[i-1]:
                                    f = open(filename)
                                    f1lines = f.readlines()
                                    f1 = float(f1lines[i+1].split(":")[-1])
                                    evaluate_dict[obs].append({st + '$$' + clas :  str(f1)})
    return evaluate_dict




for key, values in parse_evaluate_log('../logs/waseem/final_waseem.log').items():
    mydata = np.random.rand(len(OBFUSCATED_STRATEGY), len(METHOD))
    print(key)
    for i, st in enumerate(OBFUSCATED_STRATEGY):
        for j, clas in enumerate(METHOD):
            for  val in values:
                for k,v in val.items():
                    if st in k and clas in k:
                        mydata[i][j] = (float(ORIGINAL_RESULT_WASEEM[j]) - float(v) ) * 100
    g = sns.heatmap(mydata, xticklabels = METHOD, yticklabels=OBFUSCATED_STRATEGY)
    g.set_xticklabels(METHOD, rotation=30)
    plt.show()






    # sns.heatmap(uniform_data)
    # plt.show()


