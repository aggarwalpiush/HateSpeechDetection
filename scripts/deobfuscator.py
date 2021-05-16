#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import enchant
from happytransformer import HappyWordPrediction
import spacy
from get_velmo_features import visual_check
from spellchecker import SpellChecker
nlp = spacy.load('en')
spell = SpellChecker()



ENG_VOCAB = enchant.Dict("en_US")

# input sentences should be tokenized form of the input sentence


def counter_casing_attack(tok_sent_list):
    clean_sentlist = []
    for tok in tok_sent_list:
        tok = tok.lower()  # camelcasing
        if tok.count('-') > len(tok)/2:   # kebabcasing
            clean_sentlist.append(tok.lower().replace('-','').replace('_',''))
        elif tok.count('_') > len(tok)/2:  # snakecasing
            clean_sentlist.append(tok.lower().replace('_', ''))
        else:
            clean_sentlist.append(tok.lower())
    return clean_sentlist


def counter_adds(tok_sent_list):
    clean_sentlist = []
    incorrect_tokens = []
    for tok in tok_sent_list:
        if ENG_VOCAB.check(tok.lower()) and tok.lower() not in ['i', 'a']:
            if len(incorrect_tokens) > 0:
                clean_sentlist.append(''.join(incorrect_tokens))
                incorrect_tokens = []
            clean_sentlist.append(tok)
        else:
            incorrect_tokens.append(tok)
    if len(incorrect_tokens) > 0:
        clean_sentlist.append(''.join(incorrect_tokens))
    return clean_sentlist


def counter_drops(tok_sent_list):
    clean_sentlist = []
    happy_wp = HappyWordPrediction()
    for i, tok in enumerate(tok_sent_list):
        if ENG_VOCAB.check(tok.lower()):
            clean_sentlist.append(tok)
        else:
            rectified_token = tok
            most_probable_tokens = happy_wp.predict_mask("%s [MASK] %s".format(' '.join(tok_sent_list[:i])
                                                                              , ' '.join(tok_sent_list[i+1:])))
            most_probable_tokens = [str(nlp(tok.token)) for tok in  most_probable_tokens]
            for sug in ENG_VOCAB.suggest(tok.lower()):
                if str(nlp(sug.token)) in most_probable_tokens:
                    rectified_token = sug
                    break
                else:
                    rectified_token = ENG_VOCAB.suggest(tok.lower())[0]
            clean_sentlist.append(rectified_token)
    return clean_sentlist

def counter_updates(tok_sent_list):
    clean_sentlist = []
    happy_wp = HappyWordPrediction()
    for i, tok in enumerate(tok_sent_list):
        if ENG_VOCAB.check(tok.lower()):
            clean_sentlist.append(tok)
        else:
            rectified_token = tok
            if '*' in tok:
                most_probable_tokens = happy_wp.predict_mask("%s [MASK] %s".format(' '.join(tok_sent_list[:i])
                                                                               , ' '.join(tok_sent_list[i + 1:])))
                most_probable_tokens = [str(nlp(tok.token)[0].lemma_) for tok in most_probable_tokens]
            else:
                most_probable_tokens = []
            for sug in ENG_VOCAB.suggest(tok.lower()):
                if '*' in tok:
                    if str(nlp(sug)[0].lemma_) in most_probable_tokens:
                        rectified_token = sug
                        break
                else:
                    if visual_check((nlp(sug)[0].lemma_.lower(), nlp(tok)[0].lemma_.lower())):
                        rectified_token = sug
                        break
                    else:
                        rectified_token = ENG_VOCAB.suggest(tok.lower())[0]
            clean_sentlist.append(rectified_token)
    return clean_sentlist


def counter_Flips(tok_sent_list):
    clean_sentlist = []
    misspelled = spell.unknown(tok_sent_list)
    for tok in tok_sent_list:
        if tok in misspelled:
            clean_sentlist.append(spell.correction(tok))
        else:
            clean_sentlist.append(tok)
    return clean_sentlist



def main():
    pass


if __name__ == '__main__':
    main()

