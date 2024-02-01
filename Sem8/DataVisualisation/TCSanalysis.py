# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:09:24 2023

@author: 20PW19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv("2013TCSData.csv")

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by = 'Date')

plt.plot(data.Date, data.Open)
plt.plot(data.Date, data.High)
plt.plot(data.Date, data.Low)
plt.plot(data.Date, data.Close)

plt.xlabel('Year') 
plt.ylabel('Price') 
plt.title('TCS stock price over 2013') 
plt.legend(['Open','High','low','Close']) 
# plt.show()


# data2 = pd.read_csv("TCSNS.csv")
# plt.plot(data2.Date, data2.Close)
# plt.xlabel('Year') 
# plt.ylabel('Closing Price') 
# plt.title('Closing price over years') 
# plt.show()


data3 = pd.read_csv("CorrelatedResultModified.txt")
print(data3.head(5))
# print(data3.head(5)) 
# dataplot = sb.heatmap(data3.corr(), cmap="YlGnBu", annot=True)