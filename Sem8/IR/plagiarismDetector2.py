# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:57:17 2024

@author: 20pw19
"""

inputData = ['2023 was not nice','2023 was fun','Unexpected things happened in 2023','Was 2023 nice?']

document = {
    'Life':'Unexpected things happened in 2023.',
    'Life Lessons': 'You may not get everything you wanted.',
    'Life Lessons': 'You will eventually get what you worked for.',
    '2023 Life': 'UNEXPECTED THINGS HAPPENED IN 2023.'}

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from functools import reduce
 
ps = PorterStemmer()

words = [word_tokenize(i) for i in inputData]
stemmed_sentence = [reduce(lambda x, y: x + " " + ps.stem(y), i, "") for i in words]
 
# print(stemmed_sentence)

tokenized = [ word_tokenize(c) for c in inputData ]
#print(tokenized)

stop_words = stopwords.words('english')

l = []

for i in words:
    s= ' '.join(i)
    token = word_tokenize(s)
    cleaned_token = []
    for word in token:
        if word not in stop_words:
            cleaned_token.append(word)
    print(cleaned_token)        
    l.append(cleaned_token)
    stemmed = [ps.stem(word) for word in cleaned_token]
    print(stemmed)


# class PlagiarismChecker:
#   def binary_distance(self, u, v):
#         # Binary distance between vectors u and v
#         return int(u != v)

#   def check_duplicate_title(self, new_title):
#         for existing_title in self.documents:
#             distance = self.binary_distance(new_title, existing_title)
#             if distance == 0:
#                 print("Document discarded as it has a duplicate title.")
#                 return True
#         return False
    
    
