'''
    author: Shi-Hao Li
    date: 16 Dec. 2020
    des: this script is writen for transforming the csv fomaated feature file to the data format of sklearn
    responsibility mode: 责任链模式
    1. 数据转化为sklearn格式
    2. 数据标准化，归一化，正则化等
    3. SVM分类器寻找最优超参数
    4. SVM模型性能
    5. RF模型分类
'''

class SklearnFormator(object):
    '''
        将csv特征文件转化为sklearn直接load的txt文件
    '''
    base_file = r"D:/Study_Shihao_Li/circRNA_disease_2020_thesis/data/2020"
    def __init__(self, csv_file):
        self.csv_file = self.base_file + "/"  + csv_file + ".csv"
        self.sklearn_file = self.base_file + '/sklearn/'+ csv_file + ".txt"
        self.process()

    def csv_to_sklearn(self):
        '''
        将原始csv特征文件转化成sklearn可使用的数据格式
        :param csv_file:
        :param sklearn_file:
        :return:
        '''
        ch = ','
        f1 = open(self.sklearn_file, 'w+')
        with open(self.csv_file, 'r') as f2:
            f2_read = csv.reader(f2)
            for row in f2_read:
                if f2_read.line_num > 1:
                    temp = row[1:]
                    f1.write(ch.join(temp))
                    f1.write('\n')
            f2.close()
        f1.close()
    def get_sklearn_file(self):
        return self.sklearn_file
    def process(self):
        self.csv_to_sklearn()

class DataPreprocessor(object):
    '''
        加载数据并进行标准化, 采用sklearn.preprocessing.scale()
    '''
    def __init__(self):
        pass
    def preprocessing(self, file_name, column_num, ratio, random_state):
        '''
        对划分后的样本进行标准化
        按指定比例和指定随机状态进行样本划分
        :param file_name: 要进行处理的sklearn初始样本
        :param column_num: 要进行分割的列号
        :param ratio: 训练集和测试集的比例
        :param random_state: 划分训练集测试集的随机状态
        :return:
        '''
        data = np.loadtxt(file_name, dtype = float, delimiter = ',')
        # 将txt文件按第column_num, 进行列分割
        y, x = np.split(data, [column_num], axis=1)

        # 将样本先进行划分后再分别标准化, 避免信息泄露
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = random_state, train_size=ratio)
        x_train, x_test = self.min_max_scale(x_train, x_test)
        return x, x_train, x_test, y, y_train, y_test
    def scale(self, x):
        '''
        对传入的x数组进行列标准化
        采取的是sklearn.preprocessing.scale(x)
        :param x:
        :return:
        '''
        return preprocessing.scale(x)
    def standard_scale(self, x):
        '''
        对传入的x数组进行列标准化
        可以保存标准化参数（均值，方差等）
        :param x:
        :return: 标准化规则和标准化后的数据
        '''
        return StandardScaler.fit(x), StandardScaler.transform(x)
    def min_max_scale(self, x_train, x_test):
        '''
        对传入的x_train的属性缩放到指定的最小最大值之间
        对x_test的属性使用一样的缩放规则
        :param x_train:
        :param x_test:
        :return:
        '''
        min_max_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
        x_train_min_max = min_max_scaler.fit_transform(x_train)
        x_test_min_max = min_max_scaler.transform(x_test)
        return x_train_min_max, x_test_min_max
if __name__=='__main__':
    from Scripts.predictor.SVM import SvmModel
    from Scripts.predictor.RamdomForest import RamdomForest
    from Scripts.performance.Metrics import Metrics
    from Scripts.performance.ROC import ROC
    import csv
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split

    # 增加逻辑, 如果文件存在就不在生成新的文件了
    for i in ['60', '2000']:
        for feature_strategy in [1, 2]:
            for seed in [0, 1, 2, 3, 4]:
                sklearn_formator_60 = SklearnFormator("samples_{}_strategy{}_seed{}".format(i, feature_strategy, seed))

                # 数据准备
                data_processor = DataPreprocessor()
                x, x_train, x_test, y, y_train, y_test = data_processor.preprocessing(sklearn_formator_60.get_sklearn_file(), 1, 0.7, 2)

                # 模型训练
                # svm_model = SvmModel()
                # svm_model.cross_validation(x_train, y_train, 5, [ 'rbf'], [2**3], [2**1])
                # y_predict, y_scores =  svm_model.get_results(x, y, x_train, y_train, x_test, y_test, 5)

                rf = RamdomForest()
                y_predict, y_scores = rf.classify(x, y, x_train, y_train, x_test, y_test, 5)

                # 性能指标计算
                metrics = Metrics(y_test.ravel(), y_predict)
                metrics_list = metrics.get_metrics()
                print("sn = {}, sp = {}, mcc = {}, acc = {}".format(metrics_list[0], metrics_list[1], metrics_list[2], metrics_list[3]))

                # ROC曲线和auc
                roc = ROC(y_test.ravel(), y_scores)
