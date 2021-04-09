'''
    Date: 2021 /4/5
    this is to predict the potiential edges by the inner product of circRNA and disease
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
def get_cosine(file):
    '''
        对输入的训练集计算余弦
    :param file:
    :return:
    '''
    df = pd.read_csv(file, index_col=0)
    print(df)
    inner_product = []
    for i in df.index:
        rna = df.loc[i]['0_x':'511_x']
        disease = df.loc[i]['0_y':]
        inner_product.append(np.dot(rna, disease))
    df['inner_product'] = inner_product
    df.to_csv(r"./cos_method_3_sample_picked.csv")
def draw_roc(y_test, y_scores):
    '''
    画出ROC曲线
    :param y_test: 测试集的原始标签
    :param y_scores: 测试集的预测分数
    :return:
    '''
    # roc曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    optimal_th, optimal_point = find_optimal_cutoff(tpr, fpr, thresholds)
    area = auc(fpr, tpr)
    # 画图过程
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkred', lw=lw, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Svm Receiver Operating Characteristic curve')
    plt.legend(loc='lower right')
    plt.show()
def typical_sampling(group, typical_n_dict):
    name = group.name
    n = typical_n_dict[name]
    return group.sample(n=n)

def diff_threshold(file):
    data = pd.read_csv(file, index_col=0)
    np.random.seed(seed=0)
    typical_n_dict = {1: 511, 0: 511}
    result = data.groupby('label').apply(typical_sampling, typical_n_dict)
    y_test = result['label'].to_list()
    y_prob = result['inner_product'].tolist()
    draw_roc(y_test, y_prob)

def find_optimal_cutoff(tpr, fpr, threshold):
    y = tpr - fpr
    youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point
if __name__ == '__main__':
    # for i in [1, 2, 3, 'random']:
    file = r"./method{}_sample_picked.csv".format('3')
    # get_cosine(file)
    diff_threshold(r"./cos_method_3_sample_picked.csv")