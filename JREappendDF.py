# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:47:32 2020

@author: barth
"""

import pandas as pd
import re
from tqdm import tqdm
from nltk.corpus import stopwords


df = pd.read_pickle('JREdfUPDATED.pkl')

def onlyWords(captions):
    new_cap = re.sub('\d{1,6}\\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}','',captions)
    new_cap = re.sub('<.?font[^>]*>','',new_cap)
    new_cap = re.sub('\\n\\n\\n',' ',new_cap)
    words = new_cap.split()
    unique_words = set(words)
    unique_words = [w for w in list(unique_words) if w not in stopwords.words('english')]
    return set(unique_words)

only_words = []
for i in tqdm(df['Captions']):
    if type(i) == str:
        only_words.append(onlyWords(i))
    else:
        only_words.append(i)
        
#only_words = [onlyWords(i) if type(i) == str else i for i in df['Captions']]

df['CaptionWords'] = only_words

"""
def getBoWarray(item, uniqueWords):
    wordDict = dict.fromkeys(uniqueWords, 0)
    for word in item:
        wordDict[word] += 1
    return wordDict

all_words = []
for i in only_words:
    if type(i) == set:
        all_words.extend(list(i))

all_words = set(all_words)


word_dicts = []
for i in tqdm(df['CaptionWords']):
    if type(i) == set:
        word_dicts.append(getBoWarray(i,all_words))
    else:
        word_dicts.append(i)
        
df['CaptionWordsDict'] = word_dicts
"""

df.to_pickle('JREdfWithBoW.pkl')



