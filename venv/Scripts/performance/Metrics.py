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
import math
import numpy as np
if __name__ == '__main__':
    y_test = np.array([[1],[0], [1], [0], [1], [0]]).ravel()
    y_predict = [1, 0, 0, 0, 1, 1]
    metrics = Metrics()
    result = metrics.compute_metrics(y_test, y_predict)
    print(result)