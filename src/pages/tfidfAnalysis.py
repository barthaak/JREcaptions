
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ast 


def identity_tokenizer(text):
  return text


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    return top_feats



def top_mean_feats(vecs, features, grp_ids=[], min_tfidf=0, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if len(grp_ids) > 0:
        D = vecs[grp_ids].toarray()
    else:
        D = vecs.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    df = pd.DataFrame(top_tfidf_feats(tfidf_means, features, top_n), 
                      columns=['feature','tfidf'])
    return df

def checkInName(names,word):
    all_names = list(names)
    ids = [c for c,i in enumerate(all_names) if word.lower() in i.replace(',','').lower().split()]
    return ids
    


def top_feats_by_class(names, vecs, y, features, min_tfidf=0, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.array(checkInName(names,label))
        feats_df = top_mean_feats(vecs, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df['label'] = label
        feats_df['num'] = len(ids)
        dfs.append(feats_df)
    return dfs
    

def getTFIDFs(captionsSeries):
    vectorizer = TfidfVectorizer(tokenizer = identity_tokenizer, lowercase=False)

    all_captions = [list(t) for t in captionsSeries if type(t) == set]

    vectors = vectorizer.fit_transform(all_captions)

    feature_names = vectorizer.get_feature_names()
    
    return vectors, feature_names


