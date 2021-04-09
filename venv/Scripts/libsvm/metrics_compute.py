'''
    des: 用来计算libsvm预测出的结果指标：sn, sp, acc, mcc
'''
class MetricsComputor(object):
    def __init__(self):
        pass
    def compute(self, y_test_file, y_predict_file):
        f1 = open(y_test_file, 'r', encoding='utf-8')
        f2 = open(y_predict_file, 'r', encoding='utf-8')

        y_test = [int(i) for i in f1.readline().strip().split()]

        y_predict = [ int(i) for i in f2.readline().strip().split()]

        f1.close()
        f2.close()
        print(y_test)
        print(y_predict)
        return y_test, y_predict

if __name__ == '__main__':
    from Scripts.performance.Metrics import Metrics

    metrics_computor =  MetricsComputor()
    # for i in ['60', '2000']:
    #     y_test, y_predict = metrics_computor.compute(
    #         r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\libsvm\predict_results\libsvm_samples_{}_y_test.txt".format(i),
    #         r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\libsvm\predict_results\libsvm_samples_{}_y_predict.txt".format(i))
    #     metrics = Metrics(y_test, y_predict)
    #     metrics_list = metrics.get_metrics()
    #     print("libsvm_samples_{}_y_predict.txt: sn = {}, sp = {}, mcc = {}, acc = {}".format(i, metrics_list[0], metrics_list[1], metrics_list[2], metrics_list[3]))
    name_list = ['60', '2000']
    strategy_list = ['1', '2']
    seed_list = ['0']

    for i in range(len(name_list)):
        for j in range(len(strategy_list)):
            for k in range(len(seed_list)):
                y_test, y_predict = metrics_computor.compute(
                    r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\libsvm\predict_results\libsvm_samples_{}_strategy{}_seed{}_y_test.txt"
                        .format(name_list[i], strategy_list[j], seed_list[k]),
                    r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\libsvm\predict_results\libsvm_samples_{}_strategy{}_seed{}_y_predict.txt"
                        .format(name_list[i], strategy_list[j], seed_list[k]))
                metrics = Metrics(y_test, y_predict)
                metrics_list = metrics.get_metrics()
                print("libsvm_samples_{}_y_predict.txt: sn = {}, sp = {}, mcc = {}, acc = {}".format(i, metrics_list[0], metrics_list[1], metrics_list[2], metrics_list[3]))
