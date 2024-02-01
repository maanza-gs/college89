# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:48:58 2023

@author: 20PW19
"""

import requests 
from bs4 import BeautifulSoup 
import lxml
import pandas as pd

URL = "https://www.goodreads.com/shelf/show/2023-release"
r = requests.get(URL) 

soup = BeautifulSoup(r.text, 'html.parser') # If this line causes an error, run 'pip install html5lib' or install html5lib 
# print(soup.prettify()) 

bookTitles=[] 
authorNames=[]
bookLink=[] 
   
bookNameScraping = soup.findAll('a', attrs={'class': 'bookTitle'})
# for i in bookNameScraping:
#     print('B: ', i.text)
    
authorNameScraping = soup.findAll('a', attrs={'class': 'authorName'})
# for i in authorNameScraping:
#     print('Author: ', i.text)

# for i in bookNameScraping:
#     print('Book:',i.text)
#     print("link:: ", i.get('href'), '\n')

for i in range(0,len(bookNameScraping)):
    bookTitles.append(bookNameScraping[i].text)
    authorNames.append(authorNameScraping[i].text)
    bookLink.append(bookNameScraping[i].get('href'))
    # print( bookNameScraping[i].text, '\t',authorNameScraping[i].text ,'\t', bookNameScraping[i].get('href'), '\n')
    
books = {'name': bookTitles, 'author': authorNames, 'link': bookLink}    
df = pd.DataFrame(books)

print(df.to_string())
   