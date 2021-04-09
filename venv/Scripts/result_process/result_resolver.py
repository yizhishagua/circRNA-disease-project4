'''
    date: 2021/4/6
    解析判别结果
'''
import pandas as pd
import numpy as np
def load_file(file):
    return pd.read_csv(file)
def get_pointed_df(df, lower, upper):
    '''
        给定阈值上下界， 返回满足条件的df['Disease Name', 'circBase ID']
    :param lower: 大于等于下界
    :param upper: 小于上界
    :return:
    '''
    match_rule = (df['predic_posi_proba']>=lower) & (df['predic_posi_proba']<upper)
    df = df[match_rule][['Disease Name','circBase ID']]
    return df
if __name__ == '__main__':
    method1_4_mer_pred_file = r"./method1_dataset_disease_sim_4_mer_pred_res.csv"
    method2_4_mer_pred_file = r"./method2_dataset_disease_sim_4_mer_pred_res.csv"

    df1 = load_file(method1_4_mer_pred_file)[['predic_posi_proba','Disease Name','circBase ID']]
    df2 = load_file(method2_4_mer_pred_file)[['predic_posi_proba','Disease Name','circBase ID']]
    for i in np.arange(0.90, 1.0, 0.01):
        i = round(i, 2)
        _df1= get_pointed_df(df1, i, round(i+0.01, 2))
        _df2 = get_pointed_df(df2, i, round(i+0.01, 2))
        df_combine = pd.merge(_df1, _df2, how='inner')

        # print(_df1)
        # print(_df2)
        # print(df_combine)
        df_combine.to_csv(r"./threshold_{}_{}_combine_samples.csv".format(i, round(i+0.01, 2)))
