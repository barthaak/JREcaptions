# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:42:12 2020

@author: barth
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import urllib.request
from pytube import YouTube
import pytube
import json
import time

with open('JREshowsUPDATED.txt','r', encoding='utf-8') as f:
    vid_hrefs = []
    vid_titles = []
    for line in f:
        vid_hrefs.append(line.split('|||')[1].strip())
        vid_titles.append(line.split('|||')[0].strip())

#test_hrefs = vid_hrefs[907:911]
#vid_titles = vid_titles[907:911]

df_old = pd.read_pickle('JREdataframe.pkl')

df = pd.DataFrame(columns=['Title','Description', 'Views', 'Rating',
                 'Duration','Captions'])

for c,url in enumerate(tqdm(vid_hrefs)):
    
    title = vid_titles[c]
    old = title in list(df_old['Title'])
    
    try:
        yt = YouTube(url)
    except:
        print('\n Taking a pause ',c)
        time.sleep(7)
        try:
            yt = YouTube(url)
        except:
            print('Could not fetch new URL')
            continue
    
    if not old:
        try:
            description = yt.description
        except:
            descritpion = np.nan
        try:
            rating = yt.rating
        except:
            rating = np.nan
        try:
            length = yt.length
        except:
            length = np.nan
        try:
            views = yt.views
        except:
            views = np.nan
        try:
            caption = yt.captions['en']
            srt_captions = caption.generate_srt_captions()
        except:
            srt_captions = np.nan
    
    else:
        description = df_old[df_old['Title'] == title]['Description'].iloc[0]
        length = df_old[df_old['Title'] == title]['Duration'].iloc[0]
        srt_captions = df_old[df_old['Title'] == title]['Captions'].iloc[0]

        try:
            rating = yt.rating
        except:
            rating = np.nan
            
        try:
            views = yt.views
        except:
            views = np.nan
            
    
    new_entry = {'Title':title,'Description':description, 'Views':views, 'Rating': rating,
                 'Duration':length,'Captions':srt_captions}
    
    df = df.append(new_entry,ignore_index=True)


excluded = list(set(df_old['Title']) - set(df['Title']))

for item in excluded:
    df = df.append(df_old[df_old['Title']==item].iloc[0],ignore_index=True)

df.to_pickle('JREdataframeUPDATEDv2.pkl')










