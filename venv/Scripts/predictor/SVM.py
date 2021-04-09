'''
    author: Shi-Hao Li
    date: 16 Dec. 2020
    des: this script is a svm classifier import from sklearn
'''
from sklearn import svm, metrics
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import *
import scipy as sp
class SvmModel(object):

    # 默认参数
    para_c = 1
    para_gamma = 1
    kernel = 'rbf'

    def __init__(self):
        pass
    def cross_validation(self, x_train, y_train, cv, kernel_list = ['rbf'],
                         C = range(1, 20, 1), gamma = np.arange(0.01, 0.1, 0.01)):
        '''
        交叉验证对训练集进行参数寻优
        :param x_train: 训练集
        :param y_train: 训练集标签
        :param cv: 使用几折交叉验证
        :param kernel_list: 核函数列表
        :param C: 惩罚因子C的列表
        :param gamma: 支撑向量维度列表
        :return:
        '''
        #X_train标准化
        y_train = y_train.ravel()
        # svm模型的SVC
        model = svm.SVC(kernel='rbf', probability=True)

        # 网格搜索 C:[2^-15, 2^15, 2] gamma:[2^-15, 2^5, 2^-1]
        # 'C': [i for i in sp.arange(2 ** -15, 2 ** 15, 2)]
        param_grid = {}
        # param_grid['kernel'] = ['rbf', 'linear', 'poly', 'sigmoid']
        param_grid['kernel'] = kernel_list
        param_grid['C'] = C
        param_grid['gamma'] = gamma
        grid_search = GridSearchCV(model, param_grid, n_jobs=10, verbose=1, cv=cv)
        grid_search.fit(x_train, y_train)
        best_parameters = grid_search.best_estimator_.get_params()
        # print("best kernel: {}, best c: {}, best gamma: {}".format(best_parameters['kernel'], best_parameters['C'],
        #                                                            best_parameters['gamma']))
        self.para_c = best_parameters['C']
        self.para_gamma = best_parameters['gamma']
        self.kernel = best_parameters['kernel']
        print("最优的c是：{}, gamma:{}, kernel:{}".format(self.para_c, self.para_gamma, self.kernel))

    def get_results(self, x, y, x_train, y_train, x_test, y_test, cv):
        '''
        获取svm结果
        :param x: 总的样本特征
        :param y: 总的标签
        :param x_train: 训练集样本
        :param y_train: 训练集标签
        :param x_test: 测试集样本
        :param y_test: 测试集标签
        :param cv: 交叉验证折数
        :return:
        '''

        y = y.ravel()
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        #如果是线性核, 则不用设置gamma参数
        if self.kernel == 'linear':
            clf_best = svm.SVC(C = self.para_c, kernel = self.kernel, decision_function_shape='ovr', probability=True)
        else:
            clf_best = svm.SVC(C = self.para_c, kernel = self.kernel,
                           gamma = self.para_gamma, decision_function_shape='ovr', probability=True)
                           # class_weight='balanced', probability=True)
        clf_best.fit(x_train, y_train)

        # cv为迭代次数,交叉验证用的训练集
        scores = cross_val_score(clf_best, x_train, y_train, cv=cv)

        # 划分的测试集部分得到的预测精度
        # test_accuracy = clf_best.score(x_test, y_test)
        # 预测结果, 即输出x_test的预测标签
        y_predict = clf_best.predict(x_test)
        y_score = clf_best.predict_proba(x_test)
        test_accuracy = accuracy_score(y_predict, y_test)
        print('test accuracy is:', test_accuracy)  # 用测试集来评估模型
        print("Cross validation accuracy: {},{}(+/- {})".format(scores, scores.mean(), scores.std()))
        return y_predict, y_score[:,1]

