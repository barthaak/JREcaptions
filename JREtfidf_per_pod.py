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


def getName(vidtitle):
    name = vidtitle.split('-')[-1].strip()    
    return name


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    return top_feats



def getHourMin(mins):
    hours = int(mins/60)
    mins = mins % 60
    if mins < 10:
        mins = '0'+str(mins)
    if hours < 10:
        hours = '0'+str(hours)
    
    return str(hours)+':'+str(mins)

def getIntervals(text):
    first_stamp = re.search('\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}',text).group()
    first_min = 0
    interval = 5

    time_spans = []
    while first_stamp is not None:
        next_min = first_min + interval
        next_hourmin = getHourMin(next_min)
        try:
            next_stamp = re.search(
                str(next_hourmin)+':\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}',
                text).group()
        except:
            next_stamp = None
        
        if next_stamp is None:
            interval_stamp = (first_stamp,re.findall(
                '\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}',
                text)[-1])
        else:
            interval_stamp = (first_stamp,next_stamp)
        
        time_spans.append(interval_stamp)
        first_stamp = next_stamp
        first_min = next_min
    
    return time_spans



def intervalText(text):
    new_text = re.sub('(\d{1,6}\\n)?\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}','',text)
    new_text = re.sub('<.?font[^>]*>','',new_text)
    new_text = re.sub('\\n\\n\\n',' ',new_text)
    new_text = re.sub('\\n','', new_text)
    return new_text


def getIntervalText(item,text):
    start = item[0]
    end = item[1]
    startend = (start.split('-->')[0].strip(),end.split('-->')[1].strip())
    int_text = re.search(start+'(.|\n)*'+end,text).group()
    filtered_int_text = intervalText(int_text)
    
    return filtered_int_text, startend

def top_feats_in_doc(vecs, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(vecs[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def getPodDict(text, textSamp):
    
    intervals = getIntervals(text)

    startend_list = []
    interval_texts = []
    interval_segs = []
    sent_num = 0
    
    for c,i in enumerate(intervals):
        if i[0] != i[1]:
            int_text,startend = getIntervalText(i,text)
            if c > 0:
                prior_int_text, _ = getIntervalText(intervals[c-1],text)
                int_text = re.sub(sents, '', ' '.join([prior_int_text,int_text])).strip()
            
            sent_num_init = sent_num    
            sents = textSamp[sent_num]
            while sents in int_text:
                sent_num += 1
                sents = ' '.join([sents,textSamp[sent_num]])
        
            
            startend_list.append(startend)
            interval_texts.append(sents)
            interval_segs.append(textSamp[sent_num_init:sent_num+1])
            sent_num += 1
            if sent_num >= len(textSamp):
                break
        else:
            break
    
    
    
    vectorizer_interval = TfidfVectorizer(stop_words=stopwords.words('english'), lowercase=False)
    vectors_interval = vectorizer_interval.fit_transform(interval_texts)
    feature_names_interval = vectorizer_interval.get_feature_names()
    
    top_feats_intervals = [top_feats_in_doc(vectors_interval,feature_names_interval,i, 10) for i in range(len(interval_texts))] 
    dict_int = {'StartEnd':startend_list,'IntervalText':interval_texts,
                           'IntervalSegs':interval_segs,'Top10':top_feats_intervals}
    
    return dict_int


df_init = pd.read_pickle('JREdfWithBoW.pkl')
df_init['Name'] = [getName(i) for i in df_init['Title']]

# test_text = df_init['Captions'][146]
# test_textSamp = df_init['TextSegments'][146]
# df_test = pd.DataFrame(getPodDict(test_text,test_textSamp))

all_interval_dicts = []
for pod in tqdm(range(len(df_init))):
    if type(df_init['Captions'][pod]) == str:
        all_interval_dicts.append(getPodDict(df_init['Captions'][pod],df_init['TextSegments'][pod]))
    else:
        all_interval_dicts.append(np.nan)

df_init['TextIntervalDicts'] = all_interval_dicts

df_init.to_pickle('JREdfWithTimeInfo.pkl')






"""
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize

def getWords(text):
    new_text = re.sub('\\n','',text)
    words = word_tokenize(new_text)
    unique_words = set(words)
    unique_words = [w for w in list(unique_words) if w not in stopwords.words('english')]
    
    return unique_words

words_list = []
for i in tqdm(df_int['IntervalText']):
    words_list.append(getWords(i))
# Join the different processed titles together.
long_string = ','.join(list(df_int['IntervalText'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
plt.figure(figsize=(12, 9))
plt.imshow(wordcloud)
plt.axis("off")

plt.show()

from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
sns.set_style('whitegrid')
# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(df_int['IntervalText'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)


"""

















#plt.scatter(df['PodNum'],df['Views'])


