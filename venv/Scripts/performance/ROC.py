class ROC(object):
    auc = 0
    def __init__(self, y_test, y_scores):
        self.draw_roc(y_test, y_scores)
        pass

    def draw_roc(self, y_test, y_scores):
        '''
        画出ROC曲线
        :param y_test: 测试集的原始标签
        :param y_scores: 测试集的预测分数
        :return:
        '''
        # roc曲线
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)

        # area under ROC
        self.auc = auc(fpr, tpr)

        # 画图过程
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkred', lw=lw, label='ROC curve (area = %0.2f)' % self.auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Svm Receiver Operating Characteristic curve')
        plt.legend(loc='lower right')
        print("AUC is: ", self.auc)
        # plt.show()
    def get_auc(self):
        return self.auc

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc