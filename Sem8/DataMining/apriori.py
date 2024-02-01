# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 08:59:29 2024

@author: 20pw19
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules 

data = pd.read_csv('CovidDataset.csv')
print(data.head())
