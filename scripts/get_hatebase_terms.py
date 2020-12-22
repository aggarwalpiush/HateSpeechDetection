#! usr/bin/env python
# -*- coding : utf-8 -*-

import json
import codecs


def extract_terms_from_json(jsonfile="../dictionaries/hatebase_lex.json"):
    terms = []
    with codecs.open(jsonfile, 'r', 'utf-8') as f:
        data = json.load(f)
        for vocab in data['result']:
            terms.append(vocab["term"])
    return set(terms)


def get_lexicons(inputfile="../dictionaries/hurtlex_lex.txt"):
    hate_lexicons = []
    with codecs.open(inputfile, 'r', 'utf-8') as dict_in:
        for line in dict_in:
            lexicon = line.strip().lower().replace('\r\n', '')
            hate_lexicons.append(lexicon)
    return set(hate_lexicons)


if __name__ == "__main__":
    extract_terms_from_json("../dictionaries/hatebase_lex.json")

