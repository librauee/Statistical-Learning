# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:42:04 2019

@author: Administrator
"""

import pandas as pd
from math import log


# 例5.2
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

# 经验熵
def entropy(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label]+=1
    ent = -sum([( p/data_length) * log(p/data_length,2)
                for p in label_count.values()])
    return ent

# 经验条件熵
def condition_entropy(datasets,axis = 0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    condi_ent = sum([ (len(p) / data_length)*entropy(p) for p in feature_sets.values()])
    #print(feature_sets)
    return condi_ent

# 信息增益
def info_gain(ent,condi_entropy):
    return ent - condi_entropy

def info_gain_train(data_sets):
    count = len(datasets[0]) - 1
    ent = entropy(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent,condition_entropy(datasets,axis=c))
        best_feature.append((c,c_info_gain))
        print("特征（{}）的信息增益为： {:.3f}".format(labels[c],c_info_gain))
    best = max(best_feature, key=lambda x:x[-1])
    print( '特征({})的信息增益最大，选择为根节点特征'.format(labels[best[0]]))


info_gain_train(datasets)