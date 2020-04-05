#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:04:29 2020

@author: salman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
data = pd.read_csv('svm.csv')
print(data.head(5))
real_x=data.iloc[:,[2,3]].values
real_y=data.iloc[:,4].values
train_x,test_x,train_y,test_y = train_test_split(real_x,real_y,test_size=0.25,random_state=0)
s_c = StandardScaler()#data ko -2 se 2 tak distribute krega
train_x=s_c.fit_transform(train_x)
test_x= s_c.transform(test_x)
cls_svc = SVC(kernel='linear',random_state=0)
cls_svc.fit(train_x,train_y)
y_pred= cls_svc.predict(test_x)
print(y_pred)#output  of predict
c_m = confusion_matrix(test_y,y_pred)
print(c_m)
x_set,y_set = train_x,train_y
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1,step = 0.01),
                    np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1,step = 0.01))
plt.contourf(x1,x2,cls_svc.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
alpha=0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('SVM implementation')
plt.legend()
plt.xlabel('Age')
plt.ylabel('estimate salery')
plt.show()
