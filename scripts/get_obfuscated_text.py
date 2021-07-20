#! usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import random
from nltk.tokenize import TweetTokenizer
import operator
import nltk
from get_hatebase_terms import extract_terms_from_json, get_lexicons
from get_interpretable_spans import get_span
from get_diacritics import get_diacritics_dict
from args import get_args
from somajo import SoMaJo
import os
import eng_to_ipa as ipa

de_tokenizer = SoMaJo("de_CMC")

args = get_args()



AVAILABLE_OBFUSCATION = ['camelcasing', 'snakecasing', 'spacing', 'voweldrop', 'masking', 'spelling', 'leetspeak', 'dicritics',
                         'mathspeak', 'reversal', 'firstcharacter']

LEXICON_DICT = get_span(args.hier_soc_file, args.hier_soc_ngram, args.hier_soc_thld)
SMALL_DIACRITIC_DICT, CAPS_DIACRITIC_DICT = get_diacritics_dict("../dictionaries/diacritics.txt")
 


class Obfuscation_strategies(object):
    def __init__(self, inputspan):
        self.inputspan = inputspan
        if isinstance(self.inputspan, str):
            self.inputspan = [self.inputspan]


    def apply_camelcasing(self):
        outputspan = []
        for each_span in self.inputspan:
            outspan = []
            for i in range(len(each_span)):
                if not i%2 == 0:
                    outspan.append(each_span[i].capitalize())
                else:
                    outspan.append(each_span[i])
            outputspan.append(''.join(outspan))
        return outputspan

    def apply_phonetic(self):
        outputspan = []
        for each_span in self.inputspan:
            phonetics = ipa.ipa_list(each_span)
            outspan = []
            for tok in phonetics:
                outspan.append(random.choice(tok))
            outputspan.append(' '.join(outspan))
        return outputspan

    def apply_kebabcasing(self):
        outputspan = []
        for each_span in self.inputspan:
            outspan = []
            for i in range(len(each_span)):
                if not i == 0:
                    outspan.append('-'+ str(each_span[i]))
                else:
                    outspan.append(each_span[i])
            outputspan.append(''.join(outspan))
        return outputspan

    def apply_diacritics(self):
        outputspan = []
        for each_span in self.inputspan:
            len_span = len(each_span)
            outspan = each_span
            getchar = lambda c: random.choice(list(SMALL_DIACRITIC_DICT[c])) if c in SMALL_DIACRITIC_DICT \
                else (random.choice(list(CAPS_DIACRITIC_DICT[c])) if c in CAPS_DIACRITIC_DICT else c)
            if len_span > 2:
                rep = random.choice(range(len_span))
                try:
                    outspan = outspan.replace(outspan[rep], getchar(outspan[rep]))
                except AttributeError:
                    outspan = outspan
            outputspan.append(outspan)
        return outputspan


    def apply_snakecasing(self):
        outputspan = []
        for each_span in self.inputspan:
            outspan = []
            for i in range(len(each_span)):
                if not i == 0:
                    outspan.append('_'+ str(each_span[i]))
                else:
                    outspan.append(each_span[i])
            outputspan.append(''.join(outspan))
        return outputspan


    def apply_spacing(self):
        outputspan = []
        for each_span in self.inputspan:
            outspan = []
            for i in range(len(each_span)):
                if not i == 0:
                    outspan.append(' '+ str(each_span[i]))
                else:
                    outspan.append(each_span[i])
            outputspan.append(''.join(outspan))
        return outputspan

    def apply_characterdrop(self):
        outputspan = []
        for each_span in self.inputspan:
            outspan = each_span
            len_span = len(outspan)
            if len_span > 2:
                rep = random.choice(range(1, len_span-1))
                try:
                    outspan = outspan.replace(outspan[rep], '')
                except AttributeError:
                    outspan = outspan
            outputspan.append(outspan)
        return outputspan

    def apply_voweldrop(self):
        vowels = ('a', 'e', 'i', 'o', 'u')
        outputspan = []
        for each_span in self.inputspan:
            outspan = []
            for i in each_span:
                if not i in vowels:
                    outspan.append(i)
                else:
                    continue
            outputspan.append(''.join(outspan))
        return outputspan

    def apply_random_masking(self):
        '''
        masking is done just by selecting random in-characters of the word having
        more than 2 character, such words are replaced by asterisk
        '''
        outputspan = []
        for each_span in self.inputspan:
            len_span = len(each_span)
            outspan = each_span
            if len_span > 2:
                random_rep_index = random.randint(1,len_span-2)
                try:
                    outspan = outspan.replace(outspan[random_rep_index], '*')
                except AttributeError:
                    outspan = outspan
            outputspan.append(outspan)
        return outputspan

    def apply_spelling(self):
        '''
        two random in-characters are chosen from the word having more than 3 characters
        and do the replacements.
        '''
        outputspan = []
        for each_span in self.inputspan:
            len_span = len(each_span)
            outspan = each_span
            if len_span > 3:
                random_rep_index1 = random.choice(range(1,len_span-1))
                random_rep_index2 = random.choice(range(1, len_span - 1))
                while random_rep_index1 == random_rep_index2:
                    random_rep_index2 = random.choice(range(1, len_span - 1))
                temp = list(outspan)
                pivot = temp[random_rep_index1]
                temp[random_rep_index1] = outspan[random_rep_index2]
                temp[random_rep_index2] = pivot
                outspan = "".join(temp)
            outputspan.append(outspan)
        return outputspan


    def apply_leetspeak(self):
        '''
        replacing character a,e,l,o,s  with integers
        '''
        outputspan = []
        for each_span in self.inputspan:
            getchar = lambda c: chars[c] if c in chars else c
            chars = {"a":"4","e":"3","l":"1","o":"0","s":"5"}
            outputspan.append(''.join(getchar(c) for c in each_span))
        return outputspan


    def apply_mathspeak(self):
        '''
        replacing ascii character with mathematical symbols
        '''
        outputspan = []
        for each_span in self.inputspan:
            getchar = lambda c: chars[c] if c in chars else c
            chars = {"C":"ℂ","N":"ℕ","Q":"ℚ","R":"ℝ","Z":"ℤ","M":"ℳ","L":"ℒ","l":"ℓ","E":"ℰ","a":"α",
                     "B":"β","y":"γ","p":"ρ"}
            outputspan.append(''.join(getchar(c) for c in each_span))
        return outputspan


    def apply_reversal(self):
        '''
        token reversal
        '''
        outputspan = []
        for each_span in self.inputspan:
            outputspan.append(each_span[::-1])
        return outputspan

    def apply_firstCharacter(self):
        '''
        take just first character of the word in capital case
        '''
        outputspan = []
        for each_span in self.inputspan:
            outputspan.append(each_span[0].upper())
        return outputspan

    def apply_original(self):
        '''
        return original input
        '''
        return self.inputspan

    function_mapping = {
        'phonetic' : apply_phonetic,
        'charcaterdrop' : apply_characterdrop,
        'kebabcasing' : apply_kebabcasing,
        'camelcasing': apply_camelcasing,
        'diacritics' : apply_diacritics,
        'snakecasing' : apply_snakecasing,
        'spacing': apply_spacing,
        'voweldrop': apply_voweldrop,
        'random_masking' : apply_random_masking,
        'spelling': apply_spelling,
        'leetspeak': apply_leetspeak,
        'mathspeak': apply_mathspeak,
        'reversal': apply_reversal,
        'firstCharacter': apply_firstCharacter,
        'original': apply_original
    }


class Select_span(object):
    def __init__(self, inputtext, random_ngram=1, dict_file='../dictionaries/hurtlex_lex.txt', is_hatebase = False,
                 hier_soc_file='../hierarchical_dict/soc.davidson_demo.txt', hier_soc_ngram=3, hier_soc_thld=-0.5):
        self.inputtext = inputtext
        self.random_ngram = random_ngram
        self.dict_file = dict_file
        self.is_hatebase = is_hatebase
        self.hier_soc_file = hier_soc_file
        self.hier_soc_ngram = hier_soc_ngram
        self.hier_soc_thld = hier_soc_thld
        if args.use_de_tokenizer:
            self.tokenized_input = []
            for sent in de_tokenizer.tokenize_text([inputtext]):
                for token in sent:
                    self.tokenized_input.append(token.text)
        else:
            self.tokenized_input = TweetTokenizer().tokenize(inputtext)
        self.input_length = len(self.tokenized_input)

    def apply_random(self):
        '''
        select random word from the sentence except first and last having
        sentence length (total number of tokens) greater than 4,
        by default one token is randomly selected but can apply for any gram
        '''
        filter_short_tokens = []
        for i, tok in enumerate(self.tokenized_input):
            if i != 0 and i != (self.input_length -1):
                if len(tok) > 3 and 'http' not in tok:
                    filter_short_tokens.append(tok)
        text_range = range(len(filter_short_tokens))
        random_rep_index = random.choice(text_range)
        if random_rep_index == 0:
            random_rep_index += 1
        return filter_short_tokens[random_rep_index - self.random_ngram: random_rep_index]

    def apply_random_POS(self):
        '''
        we believe most obfuscation strategy can be applied only for nouns,
        adjectives and verb, so all the nouns, adjective and verbs are selected from the sentence
        and apply any of the obfuscation strategy
        '''
        eligible_pos = []
        # apply tweet tokenizer
        # apply pos tagging
        pos_tags = nltk.pos_tag(self.tokenized_input)
        count = 0 
        for i, tag_pair in pos_tags:
            if count != 0 and count != len(pos_tags) -1:
                if tag_pair in ['VB', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBD', 'VBG', 'VBN',
                                         'VBP', 'VBZ']:
                    if len(i) > 3:
                        eligible_pos.append(i)
            count += 1
        #print([random.choice(eligible_pos)])
        return [random.choice(eligible_pos)]

    def apply_all(self):
        '''
        choose all the alpha words in the inputtext which are greater than 3 letters
        '''
        all_obfuscated_text = []
        for tok in self.tokenized_input:
            if len(tok) > 3:
                all_obfuscated_text.append(tok)
        return all_obfuscated_text


    def apply_dictionary(self):
        '''
        use hatebase lexicons to choose words from the input tweet
        '''
        dictionary_obfuscated_text = []
        if self.is_hatebase:
            lexicon_list = extract_terms_from_json(self.dict_file)
        else:
            lexicon_list = get_lexicons(self.dict_file)
        for lex in lexicon_list:
            if lex in self.inputtext.lower():
                dictionary_obfuscated_text.append(lex)
        return [random.choice(dictionary_obfuscated_text)]

    def apply_hierarchical(self):
        '''
        apply hierarchical based explanation strategy to choose terms for obfuscation
        '''
        hierarchical_obfuscated_text = {}

        for lex, val in LEXICON_DICT.items():
            if lex in self.tokenized_input:
                hierarchical_obfuscated_text[lex]= val
        #print(hierarchical_obfuscated_text)
        if len(hierarchical_obfuscated_text.keys()) >= 1:
            hier_out = [min(hierarchical_obfuscated_text.items(), key=operator.itemgetter(1))[0]]
        else:
            hier_out = []
        return hier_out
    
    def apply_original(self):
        '''
        return original input
        '''
        return self.inputtext

    function_mapping = {
        'original': apply_original,
        'random': apply_random,
        'random_POS': apply_random_POS,
        'all': apply_all,
        'dictionary': apply_dictionary,
        'hierarchical': apply_hierarchical
    }

