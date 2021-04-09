class Predict(object):
    def __init__(self):
        pass
    def predict(self, predict_file, ori_file, model_file, tar_file, b):
        '''
        对测试样本进行预测
        :param predict_file: svm-predict 的位置
        :param ori_file: 要预测的文件(.scale)
        :param model_file: 模型文件
        :param tar_file: 预测结果
        :param probability_estimates: 是否对预测结果进行概率估计，用0 or 1指定，默认为0
        :return:
        '''
        # -b 1 使用概率估计
        command = predict_file + r"/svm-predict -b {} ".format(b) + ori_file +" "+ model_file + " " + tar_file
        return command

if __name__ == '__main__':
    import os
    predictor = Predict()
    name_list = ['60', '2000']
    strategy_list = ['1', '2']
    seed_list = ['0']
    param_dic = {'kernel':['2', '2', '2', '2'], 'c':['2', '2', '8', '8'], 'gamma':['2', '8', '2', '8'], 'b':['1', '1', '1', '1']}

    for i in range(len(name_list)):
        for j in range(len(strategy_list)):
            for k in range(len(seed_list)):
                commmand1 = predictor.predict("~/circRna-disease-association/libsvm-3.24/libsvm-3.24",
                            r"~/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_test_scale.txt"
                                         .format(name_list[i], strategy_list[j], seed_list[k]),
                            r"~/circRna-disease-association/scripts/libsvm_samples_{}_strategy{}_seed{}_train_scale.txt.model"
                                        .format(name_list[i], strategy_list[j], seed_list[k]),
                            r"~/circRna-disease-association/results/libsvm_samples_{}_strategy{}_seed{}_test.predict"
                                        .format(name_list[i], strategy_list[j], seed_list[k]), '1')

                commmand2 = predictor.predict("~/circRna-disease-association/libsvm-3.24/libsvm-3.24",
                                             r"~/circRna-disease-association/data/libsvm/scale/not_selected_negative_samples_{}_strategy{}_seed{}_scale.txt"
                                             .format(name_list[i], strategy_list[j], seed_list[k]),
                                             r"~/circRna-disease-association/scripts/libsvm_samples_{}_strategy{}_seed{}_train_scale.txt.model"
                                             .format(name_list[i], strategy_list[j], seed_list[k]),
                                             r"~/circRna-disease-association/results/not_selected_negative_samples_{}_strategy{}_seed{}.predict"
                                             .format(name_list[i], strategy_list[j], seed_list[k]), '1')
                os.system(commmand1)
                os.system(commmand2)