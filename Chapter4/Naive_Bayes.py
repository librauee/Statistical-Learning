# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:11:22 2019

@author: Administrator
"""
# 高斯模型
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(x):
        return sum(x)/float(len(x))

    # 标准差
    def std(self,x):
        avg = self.mean(x)
        return math.sqrt(sum(math.pow(x_i-avg,2) for x_i in x)/float(len(x)))

    # 概率密度函数
    def gaussian_prob(self,x,mean,std):
        exp = math.pow(math.e, -1*(math.pow(x - mean,2))/(2*std))
        return (1/(math.sqrt(2*math.pi*std)))*exp

    # 计算训练的均值和方差
    def mean_and_std(self,x):
        mean_and_std=[(self.mean(i),self.std(i)) for i in zip(*x)]
        return mean_and_std

    # 分类别求出数学期望和标准差
    def fit(self,x,y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f,label in zip(x,y):
            data[label].append(f)
        self.model = {label:self.mean_and_std(value) for label,value in data.items()}
        print("GaussianNB train Done!")


    # 计算概率
    def prob(self,data):
        probabillity = {}
        for label,value in self.model.items():
            probabillity[label] = 1
            for i in range(len(value)):
                mean,std = value[i]
                probabillity[label] *= self.gaussian_prob(data[i],mean,std)
        return probabillity

    # 类别
    def predict(self,x_test):
        #print(self.prob(x_test))
        label = sorted(self.prob(x_test).items(),key=lambda x:x[-1])[-1][0]
        return label

    def score(self,x_test,y_test):
        right = 0
        for x,y in zip(x_test,y_test):
            label = self.predict(x)
            if  label == y:
                right+=1
        return right / float(len(x_test))

model = NaiveBayes()
model.fit(X_train,y_train)
#print(model.model)
print(model.predict([4.4,3.2,1.3,0.2]))
print("在测试集中的准确率：")
print(str(100*(model.score(X_test, y_test)))+"%")

# scikit-learn 实现

from sklearn.naive_bayes import  GaussianNB  # 高斯模型
from sklearn.naive_bayes import BernoulliNB # 伯努利模型
from sklearn.naive_bayes import MultinomialNB # 多项式模型

clf = GaussianNB()
# clf = BernoulliNB()
# clf = MultinomialNB()

clf.fit(X_train,y_train)
print(clf.predict([[4.4,3.2,1.3,0.2]]))
