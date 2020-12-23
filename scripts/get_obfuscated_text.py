#! usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import random
from nltk.tokenize import TweetTokenizer
import nltk
from get_hatebase_terms import extract_terms_from_json, get_lexicons
from get_interpretable_spans import get_span


AVAILABLE_OBFUSCATION = ['camelcasing', 'snakecasing', 'spacing', 'voweldrop', 'masking', 'spelling', 'leetspeak',
                         'mathspeak', 'reversal', 'firstcharacter']



class Obfuscation_strategies(object):
    def __init__(self, inputspan):
        self.inputspan = inputspan


    def apply_camelcasing(self):
        outputspan = []
        for i in range(len(self.inputspan)):
            if not i%2 == 0:
                outputspan.append(self.inputspan[i].capitalize())
            else:
                outputspan.append(self.inputspan[i])
        return ''.join(outputspan)

    def apply_snakecasing(self):
        outputspan = []
        for i in range(len(self.inputspan)):
            if not i == 0:
                outputspan.append('_'+ str(self.inputspan[i]))
            else:
                outputspan.append(self.inputspan[i])
        return ''.join(outputspan)


    def apply_spacing(self):
        outputspan = []
        for i in range(len(self.inputspan)):
            if not i == 0:
                outputspan.append(' '+ str(self.inputspan[i]))
            else:
                outputspan.append(self.inputspan[i])
        return ''.join(outputspan)

    def apply_voweldrop(self):
        vowels = ('a', 'e', 'i', 'o', 'u')
        outputspan = []
        for i in self.inputspan:
            if not i in vowels:
                outputspan.append(i)
            else:
                continue
        return ''.join(outputspan)

    def apply_random_masking(self):
        '''
        masking is done just by selecting random in-characters of the word having
        more than 2 character, such words are replaced by asterisk
        '''
        len_span = len(self.inputspan)
        outputspan = self.inputspan
        if len_span > 2:
            random_rep_index = random.randint(1,len_span-2)
            outputspan = outputspan.replace(outputspan[random_rep_index], '*')
        return outputspan

    def apply_spelling(self):
        '''
        two random in-characters are chosen from the word having more than 3 characters
        and do the replacements.
        '''
        len_span = len(self.inputspan)
        outputspan = self.inputspan
        if len_span > 3:
            random_rep_index1 = random.choice(range(1,len_span-2))
            random_rep_index2 = random.choice(range(1, len_span - 2))
            exchange_char1 = outputspan[random_rep_index1]
            outputspan = outputspan.replace(outputspan[random_rep_index1], outputspan[random_rep_index2])
            outputspan = outputspan.replace(outputspan[random_rep_index2], exchange_char1)
        return outputspan


    def apply_leetspeak(self):
        '''
        replacing character a,e,l,o,s  with integers
        '''
        getchar = lambda c: chars[c] if c in chars else c
        chars = {"a":"4","e":"3","l":"1","o":"0","s":"5"}
        return ''.join(getchar(c) for c in self.inputspan)


    def apply_mathspeak(self):
        '''
        replacing ascii character with mathematical symbols
        '''
        getchar = lambda c: chars[c] if c in chars else c
        chars = {"C":"ℂ","N":"ℕ","Q":"ℚ","R":"ℝ","Z":"ℤ","M":"ℳ","L":"ℒ","l":"ℓ","E":"ℰ","a":"α","B":"β","y":"γ","p":"ρ"}
        return ''.join(getchar(c) for c in self.inputspan)

    def apply_reversal(self):
        '''
        token reversal
        '''
        return self.inputspan[::-1]

    def apply_firstCharacter(self):
        '''
        take just first character of the word in capital case
        '''
        return self.inputspan[0].upper()

    function_mapping = {
        'camelcasing': apply_camelcasing,
        'snakecasing' : apply_snakecasing,
        'spacing': apply_spacing,
        'voweldrop': apply_voweldrop,
        'random_masking' : apply_random_masking,
        'spelling': apply_spelling,
        'leetspeak': apply_leetspeak,
        'mathspeak': apply_mathspeak,
        'reversal': apply_reversal,
        'firstCharacter': apply_firstCharacter
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
                if len(tok) > 3:
                    filter_short_tokens.append(tok)
        text_range = range(len(filter_short_tokens))
        random_rep_index = random.choice(text_range)
        if random_rep_index == 0:
            random_rep_index += 1
        return ' '.join(filter_short_tokens[random_rep_index - self.random_ngram: random_rep_index])

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
        for i, tag_pair in pos_tags:
            if i != 0 and i != len(pos_tags) -1:
                if tag_pair in ['VB', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBD', 'VBG', 'VBN',
                                         'VBP', 'VBZ']:
                    if len(tag_pair[0]) > 3:
                        eligible_pos.append(tag_pair[0])
        return random.choice(eligible_pos)

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
        return dictionary_obfuscated_text

    def apply_hierarchical(self):
        '''
        apply hierarchical based explanation strategy to choose terms for obfuscation
        '''
        hierarchical_obfuscated_text = []

        lexicon_list = get_span(self.hier_soc_file, self.hier_soc_ngram, self.hier_soc_thld)
        for lex in lexicon_list:
            if lex in self.inputtext.lower():
                hierarchical_obfuscated_text.append(lex)
        return hierarchical_obfuscated_text

    function_mapping = {
        'random': apply_random,
        'random_POS': apply_random_POS,
        'all': apply_all,
        'dictionary': apply_dictionary,
        'hierarchical': apply_hierarchical
    }
