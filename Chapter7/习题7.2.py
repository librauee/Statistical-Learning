# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:07:14 2019

@author: Lee
"""

from sklearn import svm
x=[[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]]
y=[1, 1, 1, -1, -1]
# 给以极大的惩罚参数C，可以得到线性可分支持向量机
clf = svm.SVC(kernel='linear',C=100000)
clf.fit(x, y)
print(clf.coef_)
print(clf.intercept_)


import matplotlib.pyplot as plt
import numpy as np
plt.scatter([i[0] for i in x], [i[1] for i in x], c=y)
xaxis = np.linspace(0, 3.5)
w = clf.coef_[0]
a = -w[0] / w[1]
y_sep = a * xaxis - (clf.intercept_[0]) / w[1]
b1 = clf.support_vectors_[0]
yy_down = a * xaxis + (b1[1] - a * b1[0])
b2 = clf.support_vectors_[-1]
yy_up = a * xaxis + (b2[1] - a * b2[0])
plt.plot(xaxis, y_sep, 'k-')
plt.plot(xaxis, yy_down, 'k--')
plt.plot(xaxis, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150, facecolors='none', edgecolors='k')
plt.show()
