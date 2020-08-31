# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:26:00 2020

@author: barth
"""

import pandas as pd
import re
from deepsegment import DeepSegment
from tqdm import tqdm


updating = True

segmenter = DeepSegment('en')

df = pd.read_pickle('JREdataframeUPDATED.pkl',)


def getPodNum(vidtitle):
    num = vidtitle.split('#')[1].split()[0]
    num = num.replace('-','')
    
    return int(num)

vidnums = [getPodNum(i) for i in df['Title']]

df['PodNum'] = vidnums

df = df.sort_values(['PodNum','Title']).reset_index(drop=True)


def removeNoise(captions):
    new_cap = re.sub('\d{1,6}\\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}','',captions)
    new_cap = re.sub('<.?font[^>]*>','',new_cap)
    new_cap = re.sub('\\n\\n\\n',' ',new_cap)
    new_cap = segmenter.segment_long(new_cap,10)
    return new_cap

#filtertest = removeNoise(df['Captions'][1000])

new_captions = []

if updating:
    df_old = pd.read_pickle('JREdf.pkl')

for c,i in enumerate(tqdm(df['Captions'])):
    if type(i) == str:
        if updating:
            title = df['Title'][c]
            if title in list(df_old['Title']):
                new_captions.append(df_old[df_old['Title'] == title]['TextSegments'].iloc[0])
            else:
                new_captions.append(removeNoise(i))
        else:
            new_captions.append(removeNoise(i))

    else:
        new_captions.append(i)

#new_captions = [removeNoise(i) if type(i) == str else i for i in df['Captions']]
       
df['TextSegments'] = new_captions

df.to_pickle('JREdfUPDATED.pkl')



# with open('test.txt', 'w',encoding='utf-8') as f: 
#     for line in df[df['PodNum']==94]['Captions'][66]: 
#         f.write(line)