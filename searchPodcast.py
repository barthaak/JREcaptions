# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:42:17 2020

@author: barth
"""

import spacy
import pandas as pd
import numpy as np 
from scipy.spatial import distance
from tqdm import tqdm

nlp = spacy.load('en_core_web_md')

df_init = pd.read_pickle('JREdfWithTimeInfo.pkl')


def getTFIDFvector(tfidf_scores):
    words = [tfidf[0] for tfidf in tfidf_scores]
    scores = [tfidf[1] for tfidf in tfidf_scores]
    
    if min(scores) != max(scores):
        norm_scores = [(i-min(scores))/(max(scores)-min(scores)) for i in scores]
    else:
        norm_scores = [1.0]*len(scores)
        
    word_vecs = [nlp(word).vector for word in words]
    weighted_vecs = [weight*word_vecs[c] for c,weight in enumerate(norm_scores)]
    tfidf_vec = np.mean(weighted_vecs,axis=0)
    
    return tfidf_vec

tfidf_vectors = []
for i in tqdm(df_init['TfIdfAnalysis']):
    if type(i) != float:
        tfidf_vectors.append(getTFIDFvector(i))
    else:
        tfidf_vectors.append(i)

df_init['TFIDFvector'] = tfidf_vectors
df_init.to_pickle('JREdfTFIDFinfo.pkl')


####### TESTING WITH EXAMPLE ##############
example_word = nlp('health').vector

def getClosestPod(input_vec, tfidf_vectors, top=10):
    dists = [(c,distance.euclidean(input_vec, iter_vec)) if type(iter_vec) != float else (c,999999) for c,iter_vec in enumerate(tfidf_vectors)]
    dists.sort(key=lambda x:x[1])
    closest_pods = [d[0] for d in dists[:top]]
    closest_dists = [d[1] for d in dists[:top]]
    return closest_dists, closest_pods

closest_dists, closest_pods = getClosestPod(example_word,tfidf_vectors,20)

for c,cl in enumerate(closest_pods):
    print(df_init['Title'].iloc[cl], closest_dists[c])

"""
Example health weighted:
Joe Rogan Experience #1261 - Peter Hotez 6.367766857147217
Joe Rogan Experience #1066 - Mel Gibson & Dr. Neil Riordan 6.448888778686523
Joe Rogan Experience #984 - Yvette d'Entremont 6.4752583503723145
Joe Rogan Experience #1451 - Dr. Peter Hotez 6.4827117919921875
Joe Rogan Experience #1037 - Chris Kresser 6.503815174102783
Joe Rogan Experience #1454 - Dan Crenshaw 6.512412071228027
Joe Rogan Experience #1432 - Aubrey de Grey 6.524218559265137
Joe Rogan Experience #968 - Kelly Brogan 6.524788856506348
Joe Rogan Experience #1213 - Dr. Andrew Weil 6.530531883239746
Joe Rogan Experience #1442 - Shannon O'Loughlin 6.538110256195068
Joe Rogan Experience #1145 - Peter Schiff 6.539196968078613
Joe Rogan Experience #1120 - Ben Greenfield 6.539201259613037
Joe Rogan Experience #1108 - Peter Attia 6.5476393699646
Joe Rogan Experience #1478 - Joel Salatin 6.54788875579834
Joe Rogan Experience #994 - Dom D'Agostino 6.548111915588379
Joe Rogan Experience #1474 - Dr. Rhonda Patrick 6.548310279846191
Joe Rogan Experience #1114 - Matt Taibbi 6.553725242614746
Joe Rogan Experience #1500 - Barbara Freese 6.555604934692383
Joe Rogan Experience #1340 - John Nores 6.556639671325684
Joe Rogan Experience #1164 - Mikhaila Peterson 6.558343887329102


Example health unweighted:
Joe Rogan Experience #1330 - Bernie Sanders 6.016846179962158
Joe Rogan Experience #1066 - Mel Gibson & Dr. Neil Riordan 6.049488067626953
Joe Rogan Experience #1058 - Nina Teicholz 6.061325550079346
Joe Rogan Experience #1451 - Dr. Peter Hotez 6.113254547119141
Joe Rogan Experience #1454 - Dan Crenshaw 6.116639614105225
Joe Rogan Experience #1170 - Tulsi Gabbard 6.128044605255127
Joe Rogan Experience #1245 - Andrew Yang 6.132168769836426
Joe Rogan Experience #1261 - Peter Hotez 6.135093688964844
Joe Rogan Experience #1213 - Dr. Andrew Weil 6.144526481628418
Joe Rogan Experience #1500 - Barbara Freese 6.176733493804932
Joe Rogan Experience #136 - Daniel Pinchbeck (Part 2) 6.183254241943359
Joe Rogan Experience #1432 - Aubrey de Grey 6.214227676391602
Joe Rogan Experience #1439 - Michael Osterholm 6.215042591094971
Joe Rogan Experience #1037 - Chris Kresser 6.215671539306641
Joe Rogan Experience #984 - Yvette d'Entremont 6.2212910652160645
Joe Rogan Experience #1064 - Eddie Huang & Jessica Rosenworcel 6.232550621032715
Joe Rogan Experience #1498 - Jon Stewart 6.233044147491455
Joe Rogan Experience #1002 - Peter Schiff 6.2567338943481445
Joe Rogan Experience #134 - Kevin Smith (Part 1) 6.262784957885742
Joe Rogan Experience #1167 - Larry Sharpe 6.263704299926758
"""  
    
    
    
    
    
    