#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-05-31 10:11:47
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-05-31 11:28:26
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
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(min_samples_split=22)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
with open("./tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
graph = graphviz.Source(f)
graph.render('tree')
# 调整参数
# def cv_score(d):
#     clf = DecisionTreeClassifier(max_depth=d)
#     clf.fit(X_train, y_train)
#     return clf.score(X_train, y_train), clf.score(X_test, y_test)
# depths = np.arange(1, 10)
# scores = [cv_score(i) for i in depths]
# tr_scores = [s[0] for s in scores]
# te_scores = [s[1] for s in scores]
# tr_best_index = np.argmax(tr_scores)
# te_best_index = np.argmax(te_scores)
# print("bestdepth:", te_best_index+1, "bestdepth_score:",
#       te_scores[te_best_index], '\n')
# 决策树分裂，其信息增益低于这个阈值则不在分裂
# from sklearn.model_selection import GridSearchCV
# thresholds = np.linspace(0, 0.2, 50)
# param_grid = {'min_impurity_decrease':thresholds}
# clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
# clf.fit(X, y)
# print("best_prams:{0},best_score:{1}".format(
#     clf.best_params_, clf.best_score_))
