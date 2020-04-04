#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:54:12 2020

@author: salman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("Datasets/50_Startups.csv")
x_real=data.iloc[:,[1]].values
x_real = x_real.reshape(-1,1)
print(x_real)
y_real = data.iloc[:,4].values
y_real = y_real.reshape(-1,1)
print(y_real)
train_x,test_x,train_y,test_y = train_test_split(x_real,y_real,test_size=0.25,random_state=0)
lin = LinearRegression()
lin.fit(train_x,train_y)
plt.scatter(train_x,train_y,color='red')
plt.plot(train_x,lin.predict(train_x),color='blue')
plt.show()