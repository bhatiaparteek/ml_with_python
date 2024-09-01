# -*- coding: utf-8 -*-
"""
Created on Mon May  9 01:18:02 2022

@author: Parteek Bhatia
"""
#Importing the Libraries
import numpy as np
import pandas as pd

#Data loading
dataset = pd.read_csv('Dataset.csv', header = None)  
# Making transactions
transactions = []
for i in range (0, 25):
      transactions.append([str(dataset.values[i,j]) for j in range (0, 5)])
 
#intall apriori_python by running following command in Console.
#!pip install apriori_python
#Copy above command and run in console, you will get the message
#Successfully installed apriori_python-1.0.4

#Training Apriori on the dataset
from apriori_python import apriori
#Building the model
freq_Items, rules = apriori (transactions, minSup = 0.249, minConf = 0.699)
print(rules[0])
print(freq_Items)