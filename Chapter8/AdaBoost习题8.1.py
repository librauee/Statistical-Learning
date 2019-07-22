# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:22:56 2019

@author: Lee
"""
import numpy as np

class AdaBoost:
    def __init__(self,n_estimators=30,learning_rate=0.1):
        self.clf_num=n_estimators
        self.learning_rate=learning_rate
        
    def init_args(self,data,labels):
        self.X=data
        self.y=labels
        self.M,self.N=data.shape
        # 弱分类器数目和集合
        self.clf_sets=[]
        # 初始化weights,均分
        self.weights=[1.0/self.M]*self.M
        # 基本分类器的系数 alpha
        self.alpha=[]
        
    def _classify(self,features,labels,weights):
        m=len(features)
        error=1.0           # 初始化误差率
        best_v=0.0          # 初始化分类结点
        # 单维特征
        feature_min=min(features)
        feature_max=max(features)
        # 需要的步骤数目
        n_step=(feature_max-feature_min+self.learning_rate)/self.learning_rate        
        direct,compare_array=None,None   # 初始化方式和结果数组
        for i in range(1,int(n_step)):
            v=feature_min+self.learning_rate*i
            if v not in features:
                # 计算在训练集上的分类误差率
                # 不改变所给的训练数据，但是不断改变数据权值分布
                compare_array_positive=np.array([1 if features[k]>v else -1 for k in range(m)])
                weight_error_positive=sum([weights[k] for k in range(m) if compare_array_positive[k]!=labels[k]])
                compare_array_negative=np.array([-1 if features[k]>v else 1 for k in range(m)])
                weight_error_negative=sum([weights[k] for k in range(m) if compare_array_negative[k]!=labels[k]])

                if weight_error_positive<=weight_error_negative:
                    weight_error=weight_error_positive
                    _compare_array=compare_array_positive
                    direct='positive'
                else:
                    weight_error=weight_error_negative
                    _compare_array=compare_array_negative
                    direct='negative'
                
                if weight_error<error:
                    error=weight_error
                    compare_array=_compare_array
                    best_v=v
        # print(best_v)
        return best_v,direct,error,compare_array
    
    # 计算alpha
    def _alpha(self,error):
        return 0.5*np.log((1-error)/error)
    
    # 规范化因子
    def _Z(self,weights,a,clf):
        return sum([weights[i]*np.exp(-1*a*self.y[i]*clf[i]) for i in range(self.M)])
    
    # 权值更新
    def _w(self,a,clf,Z):
        for i in range(self.M):
            self.weights[i]=self.weights[i]*np.exp(-1*a*self.y[i]*clf[i])/Z
    
    # 生成G(x)
    def G(self,x,v,direct):
        if direct=='positive':
            return 1 if x>v else -1
        else:
            return -1 if x>v else 1
        

     
    def fit(self,X,y):
        self.init_args(X,y)
        
        for epoch in range(self.clf_num):
            best_clf_error,best_v,clf_result=1.0,None,None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features=self.X[:,j]
                # 分类阈值，分类误差，分类结果
                v,direct,error,compare_array=self._classify(
                    features,self.y,self.weights)

                if error<best_clf_error:
                    best_clf_error=error
                    best_v=v
                    final_direct=direct
                    clf_result=compare_array
                    axis=j
            if self.test()==1:
                break
            else:
                # 计算G(x)系数a
                a=self._alpha(best_clf_error)
                self.alpha.append(a)
                # 记录分类器
                self.clf_sets.append((axis,best_v,final_direct))
                Z=self._Z(self.weights,a,clf_result)
                self._w(a,clf_result,Z)            
            print("已经训练完第{}个弱分类器，分类点为{}，分类方式为{},分类误差为{:.4f}，权重为{}"
                  .format(epoch+1,best_v,final_direct,best_clf_error,self.weights)) 

        print(self.alpha)    
        print(self.clf_sets)
    
    def predict(self,feature):
        result=0.0
        for i in range(len(self.clf_sets)):
            axis,clf_v,direct=self.clf_sets[i]
            result+=self.alpha[i]*self.G(feature[axis],clf_v,direct)            
        return np.sign(result)

    def test(self):
        right_count = 0
        for i in range(len(self.X)):
            feature=self.X[i]
            if self.predict(feature)==self.y[i]:
                right_count += 1
        return right_count/len(self.X)
             
                
                
data=np.array([[0. , 1. , 3.],
                     [0. , 3. , 1.],
                     [1. , 2. , 2.],
                     [1. , 1. , 3.],
                     [1. , 2. , 3.],
                     [0. , 1. , 2.],
                     [1. , 1. , 2.],
                     [1. , 1. , 1.],
                     [1. , 3. , 1.],
                     [0. , 2. , 1.]])

labels=[-1.0, -1.0,-1.0, -1.0, -1.0, -1.0,1.0, 1.0, -1.0 , -1.0]   
clf=AdaBoost()
clf.fit(data,labels)           
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
