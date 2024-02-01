# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:47:08 2024

@author: 20pw19
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read()
                 for _file in student_files]


def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(doc1, doc2): return cosine_similarity([doc1, doc2])


vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()


def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results


for data in check_plagiarism():
    print(data)
    
    
# from collections import Counter
# from tqdm import tqdm
# from scipy.sparse import csr_matrix
# import math
# import operator
# from sklearn.preprocessing import normalize
# import numpy

# def IDF(document, unique_words):
#     idf_vals = {}    
#     total_docs = len(document) 
#     for word in unique_words: 
#         cnt = 0
#         for row in document:
#             if word in row.split(" "): 
#                 cnt+=1 
#         idf_vals[word] = 1 + math.log((1+total_docs)/(1+cnt)) 
#     return idf_vals

# def fit(dataset):    
#     unique_words = set() 
#     if isinstance(dataset, (list,)):
#         print("Unique Words: ",unique_words)
#         for row in dataset:
#             for word in row.split(" "):
#                 if len(word) < 2:
#                     continue
#                 unique_words.add(word)
#         unique_words = sorted(list(unique_words))
#         print("Unique Words: ",unique_words)
#         vocab = {j:i for i,j in enumerate(unique_words)}
#     # idfs_ = IDF(dataset, unique_words)
#     # return vocab, idfs_

# # features, idfs_ = fit(document)
# fit(document)

# def transform(dataset):
#     sparse_matrix = csr_matrix((len(dataset), len(features)), dtype=float)
#     for row in range(0, len(dataset)):
#         word_count = Counter(dataset[row].split(' '))
#         for word in dataset[row].split(' '):
#             if word in list(features.keys()):
#                 tf = word_count[word] / len(dataset[row].split(' '))
#                 tfidf = tf * idfs_[word]
#                 sparse_matrix[row, features[word]] = tfidf
#     output = normalize(sparse_matrix, norm='l2', axis = 1, copy=True, return_norm=False)
#     return output

# tfidf_corpus = transform(document)

# idf_list = list(idfs_.values())
# print("Our idfs: ", idf_list) #This will give us idf vals calc by our custom implementation
# print(tfidf_corpus[0].toarray())  #This will give us tfidf vals for the first document calc by our custom implementation