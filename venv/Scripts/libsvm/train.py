class Trainer(object):
    def __init__(self):
        pass
    def train(self, train_file, ori_file, kernel, c, gamma, b):
        '''
        对特征文件进行训练，训练出来的模型在当前文件夹下
        :param train_file: svm-train 的位置
        :param ori_file: 要训练的文件
        :param kernel: 核函数
        :param c: c参数
        :param gamma: gamma参数
        :param b: 是否进行概率估计，1：是, 0：否
        :return:
        '''
        # -t 0: 线性核 1: 多项式核 2: rbf核 3: sigmoid核(-d: 设置s核函数的度)
        # -g gamma参数
        # -c C惩罚参数
        command = train_file + r"/svm-train -b {} -t {} -c {} -g {} ".format(b, kernel, c, gamma) + ori_file
        return command
if __name__ == '__main__':
    import os
    trainer = Trainer()
    name_list = ['60', '2000']
    strategy_list = ['1', '2']
    seed_list = ['0']
    param_dic = {'kernel':['2', '2', '2', '2'], 'c':['2', '2', '8', '8'], 'gamma':['2', '8', '2', '8'], 'b':['1', '1', '1', '1']}
    for i in range(len(name_list)):
        for j in range(len(strategy_list)):
            for k in range(len(seed_list)):
                commmand = trainer.train("~/circRna-disease-association/libsvm-3.24/libsvm-3.24",
                           "~/circRna-disease-association/scripts/libsvm_samples_{}_strategy{}_seed{}_train_scale.txt"
                                         .format(name_list[i], strategy_list[j], seed_list[k]),
                                 param_dic['kernel'][i], param_dic['c'][i], param_dic['gamma'][i], param_dic['b'][i])
                os.system(commmand)