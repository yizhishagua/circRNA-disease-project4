class ResultHandler(object):
    def __init__(self):
        pass

    def handle(self, ori_file, result_file, input_roc_file, y_test_file, y_predict_file):
        '''
        将预测结果和原始标签合并输出到input_roc_file中，用R做ROC曲线图
        :param ori_file:测试样本scale文件
        :param result_file: 预测结果文件
        :param input_roc_file:
        :return:
        '''
        f1 = open(ori_file, 'r', encoding='utf-8')
        f2 = open(result_file, 'r', encoding='utf-8')
        f3 = open(input_roc_file, 'w+', encoding='utf-8')
        f4 = open(y_test_file, 'w+', encoding='utf-8')
        f5 = open(y_predict_file, 'w+', encoding='utf-8')

        y_test = [line.strip().split()[0] for line in f1]
        f1.close()

        y_predict_list = [line.strip() for line in f2 if line.strip()[0] != 'l']
        y_predict = []
        f2.close()

        if len(y_test) == len(y_predict_list):
            for i in range(len(y_test)):
                temp = []
                temp.append(y_test[i])
                temp.append(y_predict_list[i])
                temp = ' '.join(temp)
                f3.write(temp)
                f3.write('\n')
                if y_test[i] == '-1':
                    y_test[i] = '0'
                if y_predict_list[i].split()[0] == '-1':
                    y_predict.append('0')
                else:
                    y_predict.append('1')
        else:
            print("handler exception")
        f3.close()

        f4.write(' '.join(y_test))
        f5.write(' '.join(y_predict))

        f4.close()
        f5.close()
        return y_test, y_predict
if __name__ == '__main__':
    result_handler = ResultHandler()
    name_list = ['60', '2000']
    strategy_list = ['1', '2']
    seed_list = ['0']

    for i in range(len(name_list)):
        for j in range(len(strategy_list)):
            for k in range(len(seed_list)):
                y_test, y_predict = result_handler.handle(r"/home/shli/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_test_scale.txt"
                                                          .format(name_list[i], strategy_list[j], seed_list[k]),
                                                         r"/home/shli/circRna-disease-association/results/libsvm_samples_{}_strategy{}_seed{}_test.predict"
                                                          .format(name_list[i], strategy_list[j], seed_list[k]),
                                                        r"/home/shli/circRna-disease-association/results/libsvm_samples_{}_strategy{}_seed{}_test.roc_input"
                                                          .format(name_list[i], strategy_list[j], seed_list[k]),
                                                        r"/home/shli/circRna-disease-association/results/libsvm_samples_{}_strategy{}_seed{}_y_test.txt"
                                                          .format(name_list[i], strategy_list[j], seed_list[k]),
                                                        r"/home/shli/circRna-disease-association/results/libsvm_samples_{}_strategy{}_seed{}_y_predict.txt"
                                                          .format(name_list[i], strategy_list[j], seed_list[k]))