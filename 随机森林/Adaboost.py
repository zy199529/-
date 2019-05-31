#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zy19950209
# @Date:   2019-05-31 12:01:31
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-05-31 14:45:42
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20,
                                                min_samples_leaf=5), algorithm='SAMME', n_estimators=200, learning_rate=0.8)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
