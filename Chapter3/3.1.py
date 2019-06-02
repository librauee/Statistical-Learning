# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:24:51 2019

@author: Administrator
"""
import math

x=[[1,1],[5,1],[4,4]]

def distance(x,y,p=2):
    if len(x)!=len(y):
        return 0
    elif len(x)>1:
        sum1=0
        for i in range(len(x)):
            sum1+=math.pow(abs(x[i]-y[i]),p)
        return math.pow(sum1,1/p)

nearest=min(distance(x[0],x[1]),distance(x[0],x[2]))
   
print(nearest)
        