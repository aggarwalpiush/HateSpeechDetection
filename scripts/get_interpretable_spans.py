#! /usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
from nltk.corpus import stopwords
from args import get_args


args = get_args()


if args.use_de_tokenizer:
    STOP_WORDS = set(stopwords.words('german'))
else:
    STOP_WORDS = set(stopwords.words('english'))


def get_span(filename, ngram = 2, threshold = -0.5):
    selected_terms = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            #print("\nline: %s\n".format(line))
            grams_score_pair = line.split("\t")
            for each_gspair in grams_score_pair:
                each_gspair = each_gspair.strip().replace("\r\n","").split(" ")
                gram = each_gspair[:-1]
                filter_gram = []
                for gr in gram:
                    if not gr == '<unk>' and not gr in STOP_WORDS:
                        filter_gram.append(gr)
                if len(filter_gram) <= ngram and len(filter_gram) > 0:
                    score = float(each_gspair[-1])
                    if score <= threshold:
                        selected_terms.append(" ".join(filter_gram).lower())
    return set(selected_terms)


if __name__ == "__main__":
    get_span("../hierarchical_dict/soc.davidson_demo.txt")



