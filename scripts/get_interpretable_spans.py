#! /usr/bin/env python
# -*- coding : utf-8 -*-

import codecs


def get_span(filename, ngram = 3, threshold = -0.5):
    selected_terms = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            grams_score_pair = line.split("\t")
            for each_gspair in grams_score_pair:
                each_gspair = each_gspair.strip().replace("\r\n","").split(" ")
                gram = each_gspair[:-1]
                if len(gram) <= ngram:
                    score = each_gspair[-1]
                    if score <= threshold:
                        selected_terms.append(" ".join(gram))
    return set(selected_terms)


if __name__ == "__main__":
    get_span("../hierarchical_dict/soc.davidson_demo.txt")



