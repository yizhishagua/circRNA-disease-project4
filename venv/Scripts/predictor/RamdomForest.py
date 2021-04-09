'''
    date: 16. Dec. 2020
    des: RamdomForest classifier
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import math
from sklearn import svm, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import scipy as sp
import copy
class RamdomForest(object):
    def __init__(self):
        pass

    def classify(self, x, y, x_train, y_train, x_test, y_test, cv):
        '''
            对输入样本进行分类
        :param x: 原始所有样本
        :param y: 所有样本的标签
        :param x_train: 训练集
        :param y_train: 训练集标签
        :param x_test: 测试集
        :param y_test: 测试集标签
        :param cv: 交叉验证的折数
        :return:
        '''
        # n_estimators: 树的数目, oob_score: 袋外分数
        clf = RandomForestClassifier(n_estimators = 150, oob_score=True)
        y_train, y_test, y = y_train.ravel(), y_test.ravel(), y.ravel()
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)
        y_scores = clf.predict_proba(x_test)
        conf_mat = confusion_matrix(y_test, y_predict)

        # print("袋外精度：",clf.oob_score_)
        scores = cross_val_score(clf, x_train, y_train, cv=cv)
        test_accuracy = accuracy_score(y_test, y_predict)
        print('test accuracy is:', test_accuracy)  # 用测试集来评估模型
        print("Cross validation accuracy: {},{}(+/- {})".format(scores, scores.mean(), scores.std()))
        print("混淆矩阵:\n", conf_mat)
        print(classification_report(y_test, y_predict))
        print("\n\n")
        return y_predict, y_scores[:,1]

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import scipy as sp
import copy