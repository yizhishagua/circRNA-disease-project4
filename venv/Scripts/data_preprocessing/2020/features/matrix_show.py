'''
    2021/4/4
    用图形化的方式展示矩阵中的值
'''
import matplotlib .pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import font_manager
def concat_df(df1, df2):
    '''

    :param df1: 疾病相似性df
    :param df2: 高斯互作df
    :return:
    '''
    a = set(df1.columns.tolist())
    b = set(df2.columns.tolist())
    dff_name_list = list(a.symmetric_difference(b))
    for i in dff_name_list:
        df2[i]=0
        df2.loc[i] = 0
    df_sim_gip = df1.copy()
    for i in df_sim_gip.index:
        for j in df_sim_gip.columns:
            if df_sim_gip.loc[i,j] == 0:
                df_sim_gip.loc[i,j] = df2.loc[i,j]
    df_sim_gip.sort_index(axis=0, inplace=True)
    df_sim_gip.sort_index(axis=1, inplace=True)
    return df_sim_gip
def show_sim_in_violin(df):
    sns.set(style="darkgrid")
    sns.violinplot(y=df[df.columns[0]], linewidth=1, color='skyblue')
    # plt.show()
    plt.savefig(r"./cirRna_feature_4_mer_tanimoto_gip.svg",dpi = 300, format='svg')
def df_to_sim(df, y_name):
    '''
    将一个疾病相似性矩阵转化为一列的相似性用于绘制violin graph
    去掉1
    :param y_name: 纵坐标的名字
    :param df:
    :return:
    '''
    data = {y_name:[]}
    for column in df.columns:
        for j in df[column]:
            data[y_name].append(j)
    df = pd.DataFrame(data)
    df = df[(df!=1).all(axis=1)]
    return df
def matrix_show(df):
    '''
        用热图表示矩阵中的相似性值
    :return:
    '''
    corr = df.corr()
    ax1 = sns.heatmap(df, cbar=1, linewidths=0.2, vmax=1, vmin=0, square=True, center=0.5, cmap='Blues', xticklabels=False, yticklabels=False)
    # plt.show()
    plt.savefig(r"./disease_sim_1_hotmap.svg",dpi = 300, format='svg')

if __name__ == '__main__':
    df = pd.read_csv(r"./cirRna_feature_4_mer_tanimoto_gip.csv", index_col=0)
    df = df.applymap((lambda x: ",".join(x.split()) if type(x) is str else x))
    # show_sim_in_violin(df_to_sim(df, 'Integrated similarity of tanimoto index and GIP between circRNAs'))
    for font in font_manager.fontManager.ttflist:
        1
        # print(font.name+'-'+font.fname)

    index = np.arange(4)
    values = [[0.93, 0.86, 0.89, 0.79], [0.93, 0.86, 0.89, 0.79]]
    SD = [[0.03, 0.04, 0.02, 0.04], [0.03, 0.04, 0.02, 0.04]]

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    x=np.arange(5)
    y1=[0.93, 0.86, 0.89, 0.79, 0.91]
    y2=[0.82, 0.98, 0.9, 0.81, 0.90]
    std_err1=[0.03, 0.04, 0.02, 0.04, 0.05]
    std_err2=[0.03, 0.01, 0.01, 0.02, 0.08]
    error_attri={"elinewidth":1.5,"ecolor":"#333333","capsize":5}
    bar_width=0.3
    tick_label=["Sn","Sp","Acc","Mcc", "Auc"]
    #创建图形
    plt.bar(x,y1, bar_width,color="#0099cc", align="center",yerr=std_err1,error_kw=error_attri,label="SVM",alpha=1)
    plt.bar(x+bar_width, y2, bar_width,color="#FF7F2A", align="center", yerr=std_err2,error_kw=error_attri,label="RF",alpha=1)
    plt.xticks(x+bar_width/2, tick_label)
    # plt.title("Metrics between different model", size=14)
    plt.legend(loc='upper center')
    plt.savefig(r"./method1_svm_rf_metrics.svg", dpi=300, format='svg')
    plt.show()