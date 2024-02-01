# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:38:00 2023

@author: 20pw19
"""

import requests 
from bs4 import BeautifulSoup 
import pandas as pd

# Making a GET request 
r = requests.get('https://en.wikipedia.org/wiki/List_of_Nobel_laureates') 

# Parsing the HTML 
soup = BeautifulSoup(r.content, 'html.parser') 

# Getting the title tag 
print(soup.title) 

table= soup.find('tbody')

s=soup.find('span',id='List-of-laureates')
row=table.find_all('tr')
content=[]
href=[]
for i in range(2):
    content+=row[i].text.splitlines()
    href+=row[i].find_all('a')
href=[i['href'] for i in href][8:13]
while("" in content):
    content.remove("")


df = pd.DataFrame()
df['winner']=content[8:13]
df['subject']=content[1:6]
df['year']=[content[7] for i in range(5)] 
df['url']=href
print(df.to_string())