# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:56:38 2023

@author: 20pw19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

data = pd.read_csv("Z:/collegeThings/Sem8/DataMining/archive/train.csv")

# print(data.head())
print("Health Insurance Data Analysis")
print("Descriptive Statistics Analysis")
#1
ageAverage = data['Age'].mean()
print('Average Age in the Data',ageAverage)
#2
premiumAverage = data['Annual_Premium'].mean()
print('Average Premium paid in a year in the Data',premiumAverage)
#3
genderCount = data.groupby('Gender').Gender.count()
print('Female Count: ', genderCount[0])
print('Male Count: ', genderCount[1])
#4
ageAvg = data.groupby('Gender').Age.mean()
print('Female Avg Age: ', ageAvg[0])
print('Male Avg Age: ', ageAvg[1], ageAvg)
#5
premiumAvgUsingInsurance = data.groupby('Previously_Insured').Annual_Premium.mean()
print('Avg Premium without vehicle insurance', premiumAvgUsingInsurance[0])
print('Avg Premium with vehicle insurance', premiumAvgUsingInsurance[1])
#6
premiumAvgUsingLicense = data.groupby('Driving_License').Annual_Premium.mean()
print('Avg Premium without driving license', premiumAvgUsingLicense[0])
print('Avg Premium with driving license', premiumAvgUsingLicense[1])
#7
premiumVarianceUsingInsurance = data.groupby('Previously_Insured').Annual_Premium.var()
print('Variance of Premium without vehicle insurance', premiumVarianceUsingInsurance[0])
print('Variance of Premium with vehicle insurance', premiumVarianceUsingInsurance[1])

print('--------------------------------------------')
print("Discrete Variable analysis")
#1
age = data["Age"]
pmf = age.value_counts().sort_index() / len(age)
pmf.plot(kind = "bar")
#2
genderProportion = data['Gender'].value_counts(normalize=True)
print("Proportion of the number of male vs female customers", genderProportion)
print ("MaleProportion:: ", data.groupby('Gender').Gender.count()/len(data['Gender']))
if genderProportion[0]<genderProportion[1]:
    print("Gender with higher proportion: Male")
else :
    print("Gender with higher proportion: Female")
#3
premiumAvgUsingGender = data.groupby('Gender').Annual_Premium.count()/len(data['Annual_Premium'])
print(premiumAvgUsingGender)
   


