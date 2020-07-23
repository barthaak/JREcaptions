# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:28:00 2020

@author: barth
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Import the wordcloud library
from wordcloud import WordCloud
import ast 



df_init = pd.read_pickle('JREdfWithTimeInfo.pkl')

def identity_tokenizer(text):
  return text

vectorizer = TfidfVectorizer(tokenizer = identity_tokenizer, lowercase=False)

df_filtered = df_init[pd.notnull(df_init['CaptionWords'])]
all_captions = [list(t) for t in df_filtered['CaptionWords'] if type(t) == set]

vectors = vectorizer.fit_transform(all_captions)

feature_names = vectorizer.get_feature_names()

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

def checkInName(df,word):
    all_names = list(df['Name'])
    ids = [c for c,i in enumerate(all_names) if word.lower() in i.replace(',','').lower().split()]
    return ids
    


def top_feats_by_class(df, vecs, y, features, min_tfidf=0, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.array(checkInName(df,label))
        feats_df = top_mean_feats(vecs, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df['label'] = label
        feats_df['num'] = len(ids)
        dfs.append(feats_df)
    return dfs



def plot_tfidf_classfeats_h(dfs, top_n = 25):
    

    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(20, 9), facecolor="w")
    x = np.arange(len(dfs[0][:top_n]))
    for i, df in enumerate(dfs):
        label = df.label[0]
        amount = df.num[0]
        df = df.drop(['label','num'],axis=1)
        df = df[:top_n]
        df = df.iloc[::-1]
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        if amount > 1:
            ax.set_xlabel("Mean Score - " + str(amount) + " podcasts", labelpad=16, fontsize=14)
        else:
            ax.set_xlabel("Mean Score - " + str(amount) + " podcast", labelpad=16, fontsize=14)

        ax.set_title(str(label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
    

input_selection_list =  "['Diaz','Patrick','Peterson','Hancock']"
selection_list = ast.literal_eval('['+input_selection_list+']')
tfidf_scores = top_feats_by_class(df_filtered, vectors, selection_list, feature_names, top_n = 100)
print(tfidf_scores[0].head())
plot_tfidf_classfeats_h(tfidf_scores, 25)

#plt.scatter(df['PodNum'],df['Views'])

pod_dict = df_init['TextIntervalDicts'][1400]

pod_df = pd.DataFrame(pod_dict)

def hourmin2short(hourmin):
    return hourmin[0][:5] + '-' + hourmin[1][:5]

pod_df['intervalShort'] = [hourmin2short(i) for i in pod_df['StartEnd']]

def getDFsInPod(startend, top10):
    df_tmp = pd.DataFrame(top10, columns=['feature','tfidf'])
    df_tmp['label'] = hourmin2short(startend)
    return df_tmp

list_of_dfs = [getDFsInPod(x, y) for x, y in zip(pod_df['StartEnd'], pod_df['Top10'])]

def printTop10features(time,df):
    print('Popular terms for',time,':')
    for word in [i[0] for i in df[df['intervalShort']==time]['Top10'].iloc[0]]:
        print(word)

# stamp = '00:50-00:55'
# printTop10features(stamp,pod_df)


























