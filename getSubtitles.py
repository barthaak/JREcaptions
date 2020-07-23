# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:12:37 2020

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

with open('JREshows.txt','r', encoding='utf-8') as f:
    vid_hrefs = []
    vid_titles = []
    for line in f:
        vid_hrefs.append(line.split('|||')[1].strip())
        vid_titles.append(line.split('|||')[0].strip())

#test_hrefs = vid_hrefs[907:911]
#vid_titles = vid_titles[907:911]
actual_nums = list(range(1,1492))
nums = []
for i in vid_titles:
    num = i.split('#')
    num = num[1].split()
    num = int(num[0].replace('-',''))
    nums.append(num)

excluded = list(set(actual_nums) - set(nums))

df = pd.DataFrame(columns=['Title','Description', 'Views', 'Rating',
                 'Duration','Captions'])

for c,url in enumerate(tqdm(vid_hrefs)):
    try:
        yt = YouTube(url)
    except:
        print('\n Taking a pause ',c)
        time.sleep(5)
        yt = YouTube(url)
  
    title = vid_titles[c]
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
    
    new_entry = {'Title':title,'Description':description, 'Views':views, 'Rating': rating,
                 'Duration':length,'Captions':srt_captions}
    
    df = df.append(new_entry,ignore_index=True)

df.to_pickle('JREdataframe.pkl')

