# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:28:22 2019

@author: Administrator
"""
import numpy as np

x=np.array([[3,3],[4,3],[1,1]])
y=np.array([1,1,-1])
a = np.zeros(len(x), np.float)
b = 0.0
Gram = None

#calculate the Gram matrix
def cal_gram():

    g = np.empty((len(x), len(x)), np.int)
    for i in range(len(x)):
        for j in range(len(x)):
            g[i][j] = np.dot(x[i],x[j])
    return g

#update parameters using stochastic gradient descent
def update(i):

    global a, b
    a[i] += 1
    b = b + y[i]
    print(a)
    print(b)


# calculate the judge condition
def cal(i):
    global a, b, x, y
    res = np.dot(a * y, Gram[i])
    res = (res + b) * y[i]
    return res

# check if the hyperplane can classify the examples correctly
def check():
    global a, b, x, y
    flag = False
    for i in range(len(x)):
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        w = np.dot(a * y, x)
        print ("RESULT: w: " + str(w) + " b: " + str(b))
        return False
    return True


if __name__ == "__main__":
    Gram = cal_gram()  # initialize the Gram matrix
    for i in range(1000):
        if not check(): 
            break