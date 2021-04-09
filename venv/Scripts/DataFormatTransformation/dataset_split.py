class DatasetSplit(object):
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
        data = np.loadtxt(file_name, dtype=str, delimiter=' ')
        # 将txt文件按第column_num, 进行列分割
        y, x = np.split(data, [column_num], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, train_size=ratio)
        return x, x_train, x_test, y, y_train, y_test

    def dataset_to_file(self, x, y, file_name):
        '''
        将划分训练测试集的样本写入到文件中
        :param x: 样本包含两个部分，x 特征数组
        :param y: 标签数组
        :param file_name:
        :return:
        '''
        f = open(file_name, 'w+', encoding='utf-8')
        for i in range(len(y)):
            temp = ' '.join(list(y[i]) + list(x[i]))
            f.write(temp)
            f.write('\n')
    def load_test(self, file_name):
        '''
        加载测试集中的标签
        :param file_name:
        :return:
        '''
        f = open(file_name, 'r')
        labels = [[line.split()[0]] for line in f]
        return np.array(labels)

if __name__ =='__main__':
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection._split import train_test_split

    base_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\libsvm"
    datasetSplitor  = DatasetSplit()

    for i in ['60', '2000']:
        for feature_strategy in [1, 2]:
            for seed in [0, 1, 2, 3, 4]:
                # 按照第一列分割np.arr, 正比负 = 7:3, 随机状态: 1
                x, x_train, x_test, y, y_train, y_test = datasetSplitor.preprocessing(
                    base_file + r"\libsvm_samples_{}_strategy{}_seed{}.txt".format(str(i), str(feature_strategy), str(seed)), 1, 0.7, 1)

                datasetSplitor.dataset_to_file(x_train, y_train,
                                               base_file + r"\libsvm_samples_{}_strategy{}_seed{}_train.txt".format(str(i), str(feature_strategy), str(seed)))

                datasetSplitor.dataset_to_file(x_test, y_test,
                                               base_file + r"\libsvm_samples_{}_strategy{}_seed{}_test.txt".format(str(i), str(feature_strategy), str(seed)))
