# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:45:36 2023

@author: 20PW19
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import utils

data = pd.read_csv("Credit.csv")
eta = 0.5

#1.
features = list(data.columns)
features.remove(features[0])
features.remove('Income')
data = data.dropna()

genderData = pd.get_dummies(data['Gender'], drop_first = True)

data = pd.concat([data, genderData], axis=1)

data.drop([data.columns[0],'Gender','Ethnicity'],axis = 1, inplace = True)

data.Student.replace(('Yes', 'No'), (1, 0), inplace=True)
data.Married.replace(('Yes', 'No'), (1, 0), inplace=True)

x = data.drop('Income', axis = 1)
y = data['Income']

y = y.round(0)

# lab = preprocessing.LabelEncoder()
# y_transformed = lab.fit_transform(y)

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)

model = LogisticRegression(max_iter=100000)
model.fit(x_training_data, y_training_data)

predictions = model.predict(x_test_data)
classification_report(y_test_data, predictions)