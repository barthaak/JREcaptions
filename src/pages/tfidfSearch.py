# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:24:24 2020

@author: barth
"""
import spacy
import numpy as np 
from scipy.spatial import distance

nlp = spacy.load('en_core_web_md')


def getClosestPod(example_word, tfidf_vectors, top=10):
    input_vec = nlp(example_word).vector

    dists = [(c,distance.euclidean(input_vec, iter_vec)) if type(iter_vec) != float else (c,999999) for c,iter_vec in enumerate(tfidf_vectors)]
    dists.sort(key=lambda x:x[1])
    closest_pods = [d[0] for d in dists[:top]]
    #closest_dists = [d[1] for d in dists[:top]]
    return closest_pods
