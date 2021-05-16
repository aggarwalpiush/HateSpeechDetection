# -*- coding: utf-8 -*-
"""
Created on Tue 23 Mar 13:43:00 2020

@author: aggarwal
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import requests
import json
from embeddings_velmo import get_embeddings, load_emb, embed, cos_sim

json_file = "../velmo/velmo_options.json"

hdf5_file = "../velmo/velmo_weights.hdf5"

loaded_vlmo = load_emb(json_file,hdf5_file)


def get_velmo_embeddings(sentence):
    sentence = [' '.join(sentence)]
    velmo_embedding = embed(sentence,loaded_vlmo)
    return velmo_embedding.mean(axis=1)


def visual_check(word_pair_tuple):
    word_pair = [[word_pair_tuple[0]], [word_pair_tuple[1]]]
    (w1, w2) = get_embeddings(word_pair, loaded_vlmo)
    if cos_sim(w1, w2) > 0.8:
        return True
    else:
        return False



if __name__ == '__main__':
    print(get_velmo_embeddings(['hello', 'how', 'are', 'you']))



