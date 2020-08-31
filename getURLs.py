# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:53:44 2020

@author: barth
"""

import urllib.request
from pytube import YouTube
import pytube
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

browser = webdriver.Chrome("C:/chromedriver/chromedriver.exe")

browser.get("https://www.youtube.com/user/PowerfulJRE/videos")
time.sleep(1)

elem = browser.find_element_by_tag_name("body")

no_of_pagedowns = 300

while no_of_pagedowns:
    if no_of_pagedowns%10 == 0:
        print(no_of_pagedowns)
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    no_of_pagedowns-=1

# SEARCH FOR REVIEWS
vid_elems = browser.find_elements_by_id('video-title')

vid_hrefs = []
vid_titles = []
for vid in vid_elems:
    if vid.text.startswith('Joe Rogan Experience #') or vid.text.startswith('Joe Rogan Experience - #'):
        vid_hrefs.append(vid.get_attribute('href'))
        vid_titles.append(vid.text)

with open('JREshowsUPDATED.txt','w', encoding='utf-8') as f:
    for c,item in enumerate(vid_titles):
        f.write(item+' ||| '+vid_hrefs[c]+'\n')