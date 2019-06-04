# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:05:30 2019

@author: Administrator
"""

import numpy as np

# SML分别对应为123

X = np.array([[1, 1], [1, 2], [1, 2], [1, 1], [1, 1], [2, 1], [2, 2], [2, 2], [2, 3], [2, 3], [3, 3], [3,2], [3,2], [3, 3], [3, 3]])
Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#拟合数据
clf.fit(X, Y)

print(clf.predict([[2, 1]]))
