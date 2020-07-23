from django.shortcuts import render
from django.http import HttpResponse
from captionAnalysis.models import MyModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import urllib, base64
import ast

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


from pages.tfidfAnalysis import *
# Create your views here.

def home_view(request, *args, **kwargs):
    my_context = {
        'my_name': 'Bart'
        }
    return render(request, 'home.html', my_context)

def plot_view(request, *args, **kwargs):
    if request.method == 'POST':
        x_text = request.POST.get('x_textfield', None)
        y_text = request.POST.get('y_textfield', None)

        try:
            x_item = MyModel.objects.values_list(x_text, flat=True)
            y_item = MyModel.objects.values_list(y_text, flat=True)
            plt.figure(figsize=(20,10))
            plt.xlabel(x_text, fontsize=16)
            plt.ylabel(y_text, fontsize=16)
            plt.scatter(x_item,y_item,alpha=0.6)

            
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf,format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = urllib.parse.quote(string)
            
            # db_item = MyModel.objects.get(pod_id = search_id)
            # #do something with user

            # html = ("<H1>{}</H1>".format(db_item.Title))
            # return HttpResponse(html)
            
            return render(request, 'plot.html', {'data':uri})
            
        except:
            return HttpResponse("something went wrong")  
    else:
        return render(request, 'home.html')
    
    
    
def captions_view(request, *args, **kwargs):
    if request.method == 'POST':
        pod_num = request.POST.get('pod_num', None)

        try:
            if int(pod_num) not in list(MyModel.objects.values_list("PodNum", flat=True)):
                title = "Podcast not available"
                caps = ""
            else:
                item = MyModel.objects.get(PodNum = pod_num)
                caps = ast.literal_eval(item.TextSegments) if item.TextSegments != 'nan' else 'No captions available'
    
                title = item.Title
                
            return render(request, 'captions.html', {'title': title, 'caps':caps})
            
        except:
            if pod_num > 1506:
                return HttpResponse("Make sure the podcast number is at most 1506, later podcasts are not supported")
            else:
                return HttpResponse("something went wrong")
    else:
        return render(request, 'home.html')
    
    
def tfidf_view(request, *args, **kwargs):
    if request.method == 'POST':
        names = request.POST.get('names', None)
        if ',' in names:
            names = names.split(',')
            names = [i.strip() for i in names]
        else:
            names = [names]
            
        try:
            db_names = MyModel.objects.values_list("Name", flat=True)
            db_capwords = MyModel.objects.values_list("CaptionWords", flat=True)
            db_capwords = [ast.literal_eval(i) if i != 'nan' else np.nan for i in db_capwords]
            non_nan_list = [c for c,i in enumerate(db_capwords) if type(i) == set]
            names_filtered = [i for c,i in enumerate(db_names) if c in non_nan_list]
            vectors, feature_names = getTFIDFs(db_capwords)
            
            selection_list = names
            tfidf_scores = top_feats_by_class(names_filtered, vectors, selection_list, feature_names, top_n = 100)
            top_n = 25
            
            
            fig = plt.figure(figsize=(20,10), facecolor="w")
            x = np.arange(len(tfidf_scores[0][:top_n]))
            for i, df in enumerate(tfidf_scores):
                label = df.label[0]
                amount = df.num[0]
                df = df.drop(['label','num'],axis=1)
                df = df[:top_n]
                df = df.iloc[::-1]
                ax = fig.add_subplot(1, len(tfidf_scores), i+1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_frame_on(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                if amount > 0:

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
                else:
                    ax.set_title('No values for "'+str(label)+'"', fontsize=16)
                    ax.set_yticks([])
                    ax.set_xticks([])

                plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)


            
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf,format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = urllib.parse.quote(string)

            return render(request, 'tfidf.html', {'data':uri})
            
        except:
            return HttpResponse("something went wrong")  
    else:
        return render(request, 'home.html')


    
