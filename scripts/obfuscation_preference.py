#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


lemmatizer = WordNetLemmatizer()


score_mapper = {'First\\_Priority' : 0.5, 'Second\\_Priority' : 0.33 , 'Third\\_Priority' : 0.17}

def main():
    anno_files = []
    for i in range(len(sys.argv)-1):
        anno_files.append(sys.argv[i+1])
    lexicon_scores = {}
    for each_file in anno_files:
        phrase_id = 1
        phrase = []
        with codecs.open(each_file, 'r', 'utf-8') as infile:
            for line in infile:
                if any(pr in line for pr in score_mapper):
                    lexicon = ps.stem(lemmatizer.lemmatize(line.split('\t')[2].lower()))
                    if '[' + str(phrase_id) + ']' in line.split('\t')[3]:
                        phrase.append(lexicon)
                        phrase_score = score_mapper[line.split('\t')[3].split('[')[0]]
                        continue
                    if len(phrase) != 0:
                        if ' '.join(phrase) not in lexicon_scores.keys():
                            lexicon_scores[' '.join(phrase)] = phrase_score
                        else:
                            lexicon_scores[' '.join(phrase)] += phrase_score
                        phrase = []
                        phrase_id += 1
                    if '[' + str(phrase_id) + ']' in line.split('\t')[3]:
                        phrase.append(lexicon)
                        phrase_score = score_mapper[line.split('\t')[3].split('[')[0]]
                        continue
                    if lexicon not in lexicon_scores.keys():
                        lexicon_scores[lexicon] = score_mapper[line.split('\t')[3]]
                    else:
                        lexicon_scores[lexicon] += score_mapper[line.split('\t')[3]]
    output_file = '/Users/aggarwalpiush/github_repo/HateSpeechDetection/manual_profane_ranked_dictionary/manually_selected_lexicons_lemma_then_stem.tsv'
    with codecs.open(output_file, 'w', 'utf-8') as out_obj:
        for k, v in dict(sorted(lexicon_scores.items(), key=lambda item: item[1], reverse = True)).items():
            out_obj.write(k+'\t'+str(v)+'\n')




if __name__ == '__main__':
    main()


