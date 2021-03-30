# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.read_csv("thfcwageslgepos.csv")
# ordered_df = df.sort_values(by='TYPE')
# my_range=range(1,len(df.index)+1)
# plt.hlines(y=my_range, xmin=ordered_df['Obfuscated'], xmax=ordered_df['Original'], color='grey', alpha=0.4)
# plt.scatter(ordered_df['Obfuscated'], my_range, color='navy', alpha=1, label='Obfuscated')
# plt.scatter(ordered_df['Original'], my_range, color='gold', alpha=0.8 , label='Original')
# plt.legend()
# plt.yticks(my_range, ordered_df['TYPE'])
# plt.title("SVM performance on spelling obfuscation", loc='left')
# plt.xlabel('F1 Score')
# plt.ylabel('Obfuscation Type')
# plt.show()


#! usr/bin/env python
# -*- coding : utf-8 -*-

import numpy as np
np.random.seed(0)
import os
import string
import seaborn as sns
import matplotlib. pyplot as plt
import sys
sns.set(font_scale = 0.6)
#sns.set_theme()
#uniform_data = np.random.rand(10, 12)
#print(uniform_data)

#sns.heatmap(uniform_data)
#plt.show()

dataset_type = sys.argv[1]
model = sys.argv[2]
original_val = sys.argv[3]

OBFUSCATED_SPAN = [ 'random', 'random_POS', 'all', 'dictionary', 'hierarchical']


OBFUSCATED_STRATEGY = [ 'camelcasing',
                        'snakecasing', 'spacing', 'voweldrop', 'random_masking', 'spelling', 'leetspeak', 'mathspeak',
                        'reversal', 'firstCharacter']

#METHOD = ['bert', 'GradB', 'svm', 'LogReg', 'cnn_attention', 'cnn_lstm', 'txt_lstm',  'bilstm']

METHOD = [str(model)]

#ORIGINAL_RESULT_DAVIDSON = ['0.4512820512820513', '0.2532299741602067', '0.3190883190883191', '0.1672473867595819', '0.32',
   #                '0.12549019607843137', '0.2981366459627329', '0.26198083067092653']

#ORIGINAL_RESULT_DAVIDSON = ['0.26198083067092653']

#ORIGINAL_RESULT_WASEEM = ['0.7869249394673123', '0.6398929049531459', '0.6702127659574468', '0.6193724420190997', '0.7178899082568807',
        #           '0.7087628865979383', '0.726161369193154', '0.7387606318347508']

#ORIGINAL_RESULT_WASEEM = ['0.7387606318347508']

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
                    if obs == 'random' and 'random_masking' in line:
                        if not line.count('random') == 2:
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
import pandas as pd

for key, values in parse_evaluate_log('../logs/'+str(dataset_type)+'/final_'+str(dataset_type)+'.log').items():
    with codecs.open('../logs/'+str(dataset_type)+'_dumble_'+str(model)+'_results'+str(key)+'.csv', 'w', 'utf-8') as outfile:
        outfile.write('TYPE,Original,Obfuscated\n')
        mydata = np.random.rand(len(OBFUSCATED_STRATEGY), len(METHOD))
        print(key)
        for i, st in enumerate(OBFUSCATED_STRATEGY):
            for j, clas in enumerate(METHOD):
                for  val in values:
                    for k,v in val.items():
                        if st in k and clas in k:
                            outfile.write(','.join(['_'.join([st.replace('random_masking','rand_mask')]),
                                                        str(float(original_val)), str(float(v))]) + '\n')

    df = pd.read_csv('../logs/'+str(dataset_type)+'_dumble_'+str(model)+'_results'+str(key)+'.csv')
    ordered_df = df.sort_values(by='TYPE')
    my_range = range(1, len(df.index) + 1)
    plt.hlines(y=my_range, xmin=ordered_df['Obfuscated'], xmax=ordered_df['Original'], color='grey', alpha=0.4)
    plt.scatter(ordered_df['Obfuscated'], my_range, color='navy', alpha=1, label='Obfuscated')
    plt.scatter(ordered_df['Original'], my_range, color='gold', alpha=0.8, label='Original')
    plt.legend()
    plt.yticks(my_range, ordered_df['TYPE'], rotation=45)
    plt.xticks(np.arange(0, 1.2, step=0.2))
    plt.title(str(model).upper()+" performance on "+str(key)+" span selection", loc='center')
    plt.xlabel('F1 (Hate Label)')
    plt.ylabel('Obfuscation Type')
    #plt.show()
    plt.savefig('../visualizations/'+str(dataset_type)+'_dumble_results_'+str(model)+'_'+str(key)+'.png', dpi=400)
    plt.clf()


# commandline statements are
# python dumble.py davidson bert 0.4512820512820513
# python dumble.py davidson bilstm 0.26198083067092653
# python dumble.py davidson svm 0.3190883190883191
# python dumble.py waseem bert 0.7869249394673123
# python dumble.py waseem bilstm 0.7387606318347508
# python dumble.py waseem svm 0.6702127659574468

