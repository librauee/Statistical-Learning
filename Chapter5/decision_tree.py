# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:30:03 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
from math import log


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, labels

datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)


class Node:
    def __init__(self,root=True,label=None,feature_name = None, feature = None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:':self.label,'feature':self.feature,'tree':self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self,val,node):
        self.tree[val] = node

    def predict(self,features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self,epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 经验熵
    @staticmethod   #静态方法可以不实例化调用
    def entropy(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2)
                    for p in label_count.values()])
        return ent

    # 经验条件熵
    def condition_entropy(self,datasets,axis = 0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        condi_ent = sum([ (len(p) / data_length)*self.entropy(p) for p in feature_sets.values()])
        return condi_ent

    # 信息增益
    @staticmethod
    def info_gain(ent,condi_entropy):
        return ent - condi_entropy


    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.entropy(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent,self.condition_entropy(datasets,axis=c))
            best_feature.append((c,c_info_gain))
            print("特征（{}）的信息增益为： {:.3f}".format(labels[c],c_info_gain))
        best = max(best_feature, key=lambda x:x[-1])
        return best

    def train(self,train_data):
        _,y_train,features = train_data.iloc[:,:-1],train_data.iloc[:,-1],train_data.columns[:-1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if  len(y_train.value_counts()) == 1:
            return Node(root=True, label = y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root= True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = labels[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True,label= y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(root = False,feature_name= max_feature_name, feature= max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name],axis=1)
            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self,train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self,X_test):
        return self._tree.predict(X_test)

datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)
print(tree)

print(dt.predict(['老年', '否', '否', '一般']))



# scikit-learn 实现
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:,:4], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.fit(X_train, y_train))
print(clf.score(X_test, y_test))