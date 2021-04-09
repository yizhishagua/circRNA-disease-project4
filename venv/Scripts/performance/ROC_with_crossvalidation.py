from typing import Optional, Callable, Any, Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve,roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import interp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._split import train_test_split
from scipy import *
import scipy as sp
import math
import numpy as np
import pickle
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from threading import Thread
import os
# #############################################################################
class SvmModel(object):
    # 默认参数
    def __init__(self, c, g):
        self.para_c = c
        self.para_gamma = g
        self.kernel = 'rbf'
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
        # scaler = MinMaxScaler()
        # scaler.fit(X_train)
        # # 测试集和独立集都按照训练集的规则化方式规范
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

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
    def get_results(self, x_train, y_train, x_test, y_test, cv):
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
        # scaler = MinMaxScaler()
        # scaler.fit(x_train)
        # # 测试集和独立集都按照训练集的规则化方式规范
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

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
        test_accuracy = clf_best.score(x_test, y_test)
        # 预测结果, 即输出x_test的预测标签
        y_predict = clf_best.predict(x_test)
        y_score = clf_best.predict_proba(x_test)
        test_accuracy = accuracy_score(y_predict, y_test)
        print('test accuracy is:', test_accuracy)  # 用测试集来评估模型
        print("Cross validation accuracy: {},{}(+/- {})".format(scores, scores.mean(), scores.std()))
        return scores, [scores.mean(), scores.std()], y_predict, y_score[:,1]
    def predict(self, X_to_predict, model):
        '''
            使用传入的模型对象对X_to_predict进行预测
        :param X_to_predict:
        :param model:使用的预测模型
        :return:判别结果，预测概率
        '''
        y_predict = model.predict(X_to_predict)
        y_predict_proba = model.predict_proba(X_to_predict)
        return y_predict, y_predict_proba
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
class Metrics(object):
    sn = -1
    sp = -1
    mcc = -1
    acc = -1
    def __init__(self, ori_label, predict_label):
        self.sn, self.sp, self.mcc, self.acc = self.compute_metrics(ori_label, predict_label)

    def compute_metrics(self, ori_label, predict_label):
        '''
        计算sn, sp, mcc
        :param ori_label: 数据的原始标签数组
        :param predict_label: 数据的预测标签数组
        :return:
        '''
        # ori_laber[i] is actual value, predict_label is predictive value
        metrics = {}
        metrics['tp'] = 0
        metrics['tn'] = 0
        metrics['fp'] = 0
        metrics['fn'] = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(ori_label)):
            if ori_label[i] == 1 and predict_label[i] == 1:
                tp += 1
                metrics['tp'] += 1
            if ori_label[i] == 1 and predict_label[i] == 0:
                fn += 1
                metrics['fn'] += 1
            if ori_label[i] == 0 and predict_label[i] == 1:
                fp += 1
                metrics['fp'] += 1
            if ori_label[i] == 0 and predict_label[i] == 0:
                tn += 1
                metrics['tn'] += 1
        # 计算sn敏感性
        if ((tp + fn) != 0):
            sn = tp / (tp + fn)
        else:
            sn = -1
            print("sn 分母为0")

        # 计算sp特异性
        if ((fp + tn) != 0):
            sp = tn / (fp + tn)
        else:
            sp = -1
            print("sp 分母为0")

        # 计算mcc马修相关系数
        if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0):
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        else:
            mcc = -1
            print('mcc分母为0')

        # 计算acc精度
        if tp != 0 or tn != 0 or fp != 0 or fn != 0:
            acc = (tp + tn)/(tp + tn + fp + fn)
        else:
            acc = -1
            print('acc 分母为0')
        return sn, sp, mcc, acc

    def get_metrics(self):
        return self.sn, self.sp, self.mcc, self.acc
class Serializaed_Model(object):
    '''
        the duration of predictive model
    '''
    def __init__(self):
        pass
    def save_model(self, model_name, model_path):
        '''
        :param model_name: the model you aim to durantion.
        :param model_path: the path of saved model.
        :return:
        '''
        with open(model_path, 'wb') as f:
            pickle.dump(model_name, f)
            f.close()
    def load_model(self):
        '''
            使用pickle.load() load the pointed model
        :return scaler: scaler
        :return model: predictive model
        '''
        with open(model_name,'rb') as f:
            model = pickle.load(f)
            f.close()
        with open(r"./train_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
            f.close()
        return scaler, model
class Thread_for_metrics_compute(Thread):

    def __init__(self, task, X, y, model, n_fold = 5, seed = 0):
        Thread.__init__(self)
        self.task = task
        self.X = X
        self.y = y
        self.model = model
        self.n_fold = n_fold
        self.seed = seed
        self.metrics = 0
    def get_n_fold_metrics(self, X, y, models_dic, n_fold=5, seed=0):
        '''
            get metics including Sn, Sp, Acc, Mcc with n_folds
        :param X: all samples nd-array like
        :param y: all labels 1d-array
        :param n_fold: fold numer, default is 5
        :param seed: random seed, default is 0
        :param model: the predictive model
        :return: dic containing average score of each metrics, key: 'sn', 'sp', 'acc', 'mcc'
        '''
        sss = StratifiedShuffleSplit(n_splits=5, train_size=0.8, random_state=seed)
        scaler = MinMaxScaler()
        metrics_dic_all_model = {}
        for key in models_dic:
            metrics_dic_all_model[key] = {
                'sn': [], 'sp': [], 'acc': [], 'mcc': []
            }
        # model = RandomForestClassifier(n_estimators=100, oob_score=False)
        for name, model in models_dic.items():
            for train_index, test_index in sss.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                model.fit(X_train, y_train)
                y_prd = model.predict(X_test)
                metrictor = Metrics(y_test, y_prd)
                sn, sp, mcc, acc = metrictor.get_metrics()
                # print("sn:{}\nsp:{}\nmcc:{}\nacc:{}".format(sn, sp, mcc, acc))
                metrics_dic_all_model[name]['sn'].append(sn)
                metrics_dic_all_model[name]['sp'].append(sp)
                metrics_dic_all_model[name]['acc'].append(acc)
                metrics_dic_all_model[name]['mcc'].append(mcc)

        # 在这启下一个线程去画数据的bar

        for name in metrics_dic_all_model.keys():
            for key, val in metrics_dic_all_model[name].items():
                metrics_dic_all_model[name][key] = round(np.mean(val), 2).astype(str) + "+/-" + round(np.std(val),2).astype(str)
        return metrics_dic_all_model
    def run(self):
        print("{} begin".format(self.task))
        self.metrics = self.get_n_fold_metrics(self.X, self.y, self.model, self.n_fold, self.seed)
class Thread_for_roc_draw(Thread):
    def __init__(self, X, y, cv, models, seed, sample_name):
        Thread.__init__(self)
        self.X = X
        self.y = y
        self.cv = cv
        self.models = models
        self.seed = seed
        self.sample_name = sample_name
    def roc_draw(self, X, y, cv, models, seed, sample_name):
        '''
        根据交叉验证折数绘制单模型或多模型的ROC曲线
        :param X: 样本特征
        :param y: 样本标签
        :param model: 预测模型, 单模型 or 多模型都可以
        :param cv: 交叉验证折数
        :param seed: 随机种子数
        :param 针对的哪个样本画的ROC曲线
        :return:
        '''
        scaler = MinMaxScaler()
        random_state = seed
        # data use to draw roc
        models_data_dic = {}
        for key in models.keys():
            models_data_dic[key] = {
                'tprs':[],
                'aucs':[],
                'mean_fpr':np.linspace(0, 1, 100),
                'mean_tpr':[]
            }

        # graph relatede obj
        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(StratifiedKFold(n_splits=cv).split(X, y)):
            print(i)
            scaler.fit(X[train])
            X[train] = scaler.transform(X[train])
            X[test] = scaler.transform(X[test])

            for name, classifier in models.items():

                classifier.fit(X[train], y[train])
                fpr, tpr, thresholds = roc_curve(y[test], classifier.predict_proba(X[test])[:,1])
                roc_auc = roc_auc_score(y[test], classifier.predict_proba(X[test])[:,1])
                interp_tpr = np.interp(models_data_dic[name]['mean_fpr'], fpr, tpr) #插值
                interp_tpr[0] = 0.0

                models_data_dic[name]['tprs'].append(interp_tpr)
                models_data_dic[name]['aucs'].append(roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
                label='', alpha=.6)
        for name in models_data_dic.keys():
            mean_tpr = np.mean(models_data_dic[name]['tprs'], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(models_data_dic[name]['mean_fpr'], mean_tpr)
            std_auc = np.std(models_data_dic[name]['aucs'])

            models_data_dic[name]['mean_tpr'] = mean_tpr
            ax.plot(models_data_dic[name]['mean_fpr'], models_data_dic[name]['mean_tpr'], color="deeppink",
                    label=r'%s (AUC = %0.2f $\pm$ %0.2f)' % (name, mean_auc, std_auc),lw=1, alpha=.8)

            std_tpr = np.std(models_data_dic[name]['tprs'], axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

            # ax.fill_between(models_data_dic[name]['mean_fpr'], tprs_lower, tprs_upper, color='deeppink', alpha=0.1, label='')
            ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01], title="")
            ax.legend(loc="lower right")
            save_path = r"./roc_"+sample_name
        for key in models_data_dic.keys():
            save_path = save_path+'_'+str(key)
        if not os.path.exists(save_path+r".svg"):
            plt.savefig(save_path+r".svg", dpi=300, format='svg')
        print("draw over")
        return plt
    def run(self):
        print("smaple_name begin_draw")
        self.roc_draw(self.X, self.y, self.cv, self.models, self.seed, self.sample_name)
class Thread_for_metrics_bar(Thread):
    def __init__(self):
        Thread.__init__(self)
    def bar_draw(self, df):
        pass
    def run(self):
        print("Bar metrics begin draw")
        pass
def load_samples(file, col_num, flag):
    '''

    :param file:
    :param col_num: 特征开始的地方
    :param flag: 是测试集使用：1，验证集：0
    :return:
    '''
    df = pd.read_csv(file, index_col=0)

    # df seed percent
    if flag:
        picked_for_test, picked_for_train = down_sampling(df, 0, 0.3)
        sample_name_train = df.sample(frac=0.7)
        sample_name = np.array(picked_for_train.iloc[:,:2])
        X = np.array(picked_for_train.iloc[:,col_num:])
        y = np.array(picked_for_train['label'].tolist())
        return X, y, sample_name, picked_for_test

    sample_name = np.array(df.iloc[:, :2])
    X = np.array(df.iloc[:, col_num:])
    y = np.array(df['label'].tolist())
    return X, y, sample_name, []
def find_optimal_cutoff(tpr, fpr, threshold):
    '''
    find optimal index of classification named Youdeng index
    :param tpr:
    :param fpr:
    :param threshold:
    :return:
    '''
    y = tpr - fpr
    youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point
def get_fpr_tpr_youdan(y_true, y_score):
        fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
        optimal_threshold, _ = self.find_optimal_cutoff(tpr, fpr, threshold)
        re_fpr = np.linspace(0, 1, 100)
        tpr = interp(re_fpr, fpr, tpr)
        fpr = re_fpr
        roc_auc = roc_auc_score(y_true, y_score)

        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
        re_recall = np.linspace(0, 1, 100)
        precision = interp(re_recall, recall[::-1], precision[::-1])
        recall = re_recall
        prc_average_precision_score = average_precision_score(y_true, y_score)

        y_pred = (np.array(y_score) >= optimal_threshold).astype(int)
        report_result = classification_report(y_true, y_pred, output_dict=True)

        return fpr, tpr, optimal_threshold, roc_auc, report_result, precision, recall, prc_average_precision_score
def find_optimal_cutoff(tpr, fpr, threshold):
    '''
    find the optimal cutoff named jordan index: 约登系数
    :param tpr:
    :param fpr:
    :param threshold:
    :return:
    '''
    y = tpr - fpr
    youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point
def dataset_split(X, y, seed=0, ratio=0.7):
    '''
    将原始样本随机划分为训练集和独立集
    :param X:原始X样本
    :param y:原始标签
    :param seed:随机数种子
    :param ratio:train:test
    :return:
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, train_size=ratio)
    return X_train, y_train, X_test, y_test
def test_set_MinMaxScale_by_train(scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled

def result_save(save_file, seed, cv_res_list, cv_res_avg, test_sn, test_sp, test_mcc, test_acc):
    '''
    保存SVM结果
    :param save_file: 保存位置
    :param seed:随机种子
    :param cv_res_list:训练集五折交叉结果每一折的acc
    :param cv_res_avg:训练集平均[acc, std]
    :param test_sn:独立集sn
    :param test_sp:独立集sp
    :param test_mcc:独立集mcc
    :param test_acc:独立集acc
    :return:
    '''
    f = open(save_file, 'a+')
    f.write('c = 32, g=0.5:\n')
    f.write('seed: {}\n'.format(str(seed)))

    #train
    f.write("\ttrain:\n")
    for i in range(len(cv_res_list)):
        f.write("\t\tfold{}: {}\n".format(i, cv_res_list[i]))
    f.write('\t\tavg: {}+/-{}\n'.format(cv_res_avg[0], cv_res_avg[1]))

    # test
    f.write('\ttest:\n')
    f.write('\t\tsn: {}\n\t\tsp: {}\n\t\tmcc: {}\n\t\tacc: {}'.format(test_sn, test_sp, test_mcc, test_acc))
    f.write('\n=================================这是分隔符=================================\n')
    f.close()
def pred_not_selected_nega(not_select_fea_path, pre_res_path, scaler, model, fea_start_col):
    '''
    预测没被选择负样本的样本的概率
    :param nagative_samplepath:带有特征的样本位置
    :param pre_res_path: 预测结果文件位置
    :param scaler: 归一化器
    :param svm_model: svm模型
    :param fea_start_col: 特征开始的列索引
    :return:
    '''
    X_test, y_test, sample_name_neg, _= load_samples(not_select_fea_path, fea_start_col, 0)
    X_test = scaler.transform(X_test)
    y_predict = model.predict(X_test)
    y_predict_proba = model.predict_proba(X_test)

    # {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data={'predic_res':y_predict,
                            'predic_nega_proba':y_predict_proba[:,0],
                            'predic_posi_proba':y_predict_proba[:,1],
                            'Disease Name': sample_name_neg[:,0],
                            'circBase ID': sample_name_neg[:,1]})
    df = df.sort_values(inplace=False,by='predic_posi_proba',ascending=False)
    df.to_csv(pre_res_path,index=False)
    return df
def pred_not_selected_nega_post_processor(pre_res_path):
    df = pd.read_csv(pre_res_df)
    return df
def down_sampling(df, seed, percent):
        '''
        给定一个dataFrame, 对其进行下采样
        :param df: 输入数据框
        :param percent: 正负样本比例
        :param seed: 随机种子
        :return:
        '''
        data0 = df[df['label'].astype(str) == '0']  # 将多数类别的样本放在data0
        data1 = df[df['label'].astype(str) == '1']  # 将少数类别的样本放在data1

        # 设置不同的随机种子, 随机获得多组负样本

        random.seed(seed)
        # 在[0, len(data0))随机取size个非重复的数
        index = random.sample(range(0, len(data0)), int(percent * (len(data1))))  # 随机给定下采样取出样本的序号

        lower_data0 = data0.iloc[list(index)]  # 按照随机给定的索引进行采样
        lower_data1 = data1.iloc[list(index)]

        # 未被选择的负样本: 在所有样本中删去被选的负样本
        not_selected_neg = data0[~data0.isin(lower_data0)]
        not_selected_pos = data1[~data1.isin(lower_data1)]
        picked_for_check = pd.concat([lower_data1, lower_data0])
        not_selected = pd.concat([not_selected_pos, not_selected_neg])
        not_selected.dropna(axis=0, how='all',inplace=True)
        return picked_for_check, not_selected


if __name__ == '__main__':
    base_path = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\feature_file\dataset_feature_file\svm_result'
    file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\pick_method{}\feature_file\dataset_feature_file\dataset_disease_sim_{}_mer.csv"\
        .format('1', 4)
    # file = r"./method{}_sample_picked.csv".format('1')
    # 样本文件  特征开始的列索引  是否训练集使用
    seed = 0
    X, y, sample_name_train, picked_for_test = load_samples(file, 4, 0)

    # 公共资源
    svm_classifier = svm.SVC(kernel='rbf', C = 32, gamma=0.5, decision_function_shape='ovr', probability=True)
    rf_classifier = RandomForestClassifier(n_estimators=200, oob_score=False)

    #异步画roc曲线
    #样本特征 样本标签 几折 分类器字典  随机种子 样本名字
    # roc = Thread_for_roc_draw(X, y, 5, {'SVM': svm_classifier, 'RF': rf_classifier}, seed, r"method1_4_mer")
    # roc.start()

    # 异步计算指标
    metrics_computor = Thread_for_metrics_compute('compute the four metrics', X, y, {'SVM':svm_classifier, 'RF':rf_classifier}, 5, 0)
    metrics_computor.start()

    metrics_computor.join()
    # 将要传入的model 封装成一个字典，标明模型名称
    print(metrics_computor.metrics)
    # ----------------------------------------------------------------------------
    # 加载预测模型进行预测
    # picked_for_test.to_csv("./_0.3_known_for_check.csv")
    # base_path = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset";
    # not_select_fea_path = base_path + r"\pick_method{}\feature_file\not_select_dataset_feature_file\dataset_disease_sim_{}_mer.csv".format('2','4')
    # # not_select_fea_path = r".\new_found_dataset_disease_sim_circRNA_fea_4_mer.csv"
    # pre_res_path = base_path+r"\pick_method{}\feature_file\not_select_dataset_feature_file\dataset_disease_sim_{}_mer_pred_res1.csv".format('2','4')
    # # pre_res_path = r".\new_found_pred_res.csv"
    # # pre_res_df = pred_not_selected_nega(not_select_fea_path, pre_res_path, scaler, model, 4)
