#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zy19950209
# @Date:   2019-05-31 11:31:12
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-05-31 12:00:38
import pandas as pd
import graphviz
import numpy as np


def read_dataset(fname):
        # 指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0)
    # 丢弃无用数据
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    labels = data['Sex'].unique().tolist()
    data['Sex'] = [*map(lambda x: labels.index(x), data['Sex'])]
    #     处理登船港口数据
    lables = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: lables.index(n))
#     处理缺失数据填充0
    data = data.fillna(0)
    return data
train = read_dataset('./kaggle_titanic/train.csv')
# print(train)
# 拆分数据集，将survived列提取出来作为标签
from sklearn.model_selection import train_test_split
y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train_shape:", X_train.shape, " y_train_shape:", y_train.shape)
print("X_test_shape:", X_test.shape, "  y_test_shape:", y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
clf = RandomForestClassifier()

#可以通过定义树的各种参数，限制树的大小，防止出现过拟合现象哦，也可以通过剪枝来限制，但sklearn中的决策树分类器目前不支持剪枝
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],        #分类标准用熵，基尼系数
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

#以下是用于比较参数好坏的评分，使用'make_scorer'将'accuracy_score'转换为评分函数
acc_scorer = make_scorer(accuracy_score)

#自动调参，GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数
#GridSearchCV用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。
grid_obj = GridSearchCV(clf,parameters,scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train,y_train)

#将clf设置为参数的最佳组合
clf = grid_obj.best_estimator_

#将最佳算法运用于数据中
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test,predictions))