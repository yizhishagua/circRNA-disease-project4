'''
    利用joblib加载已保存的模型预测未被实验验证的circRNA疾病对
'''
import pandas as pd
import numpy as np
import joblib
import os
def load_samples(file, col_num):
    '''

    :param file:
    :param col_num: 特征开始的列索引
    :return:
    '''
    df = pd.read_csv(file)
    X = np.array(df.iloc[:,col_num:])
    y = np.array(df['label'].tolist())
    return X, y
def fea_match(sample_fea_file, tar_fea_file):
    '''
    将输入的疾病样本和circRNA与疾病的特征融合：merge
    :param sample_fea_file: 原始样本文件所在的位置
    :param tar_fea_file: 特征匹配后文件的位置
    :return:
    '''
    df = pd.read_csv(sample_fea_file)
    df_circRNA = pd.read_csv(circRNA_fea_path)
    df_disease = pd.read_csv(disease_fea_path)
    merge_fea = df.merge(df_circRNA, left_on='circBase ID', right_on='circBase ID', sort=False, how='left')\
        .merge(df_disease, left_on='Disease Name', right_on='fea', sort=False, how='left')
    merge_fea = merge_fea.drop(['sequence_y','fea'],axis=1)
    merge_fea.to_csv(tar_fea_file, index=False)
def circRna_gip_matcher(sample_fea_file, tar_fea_file):
    '''
        将circRna_gip特征融合到样本中
    :param sample_fea_file:
    :param tar_fea_file:
    :return:
    '''
    df = pd.read_csv(sample_fea_file)
    df_circRNA_gip = pd.read_csv(circRNA_gip_path)
    merge_fea = df.merge(df_circRNA_gip, left_on='circBase ID', right_on='GIP_value', sort=False, how='left')
    merge_fea = merge_fea.drop(['GIP_value'],axis=1)
    merge_fea.to_csv(tar_fea_file, index=False)
def pred_not_selected_nega_post_processor(pre_res_path):
    df = pd.read_csv(pre_res_path)
    return df
if __name__ == '__main__':
    base_path = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset";
    not_select_path = base_path+r"\not_selected_dataset_after_rank_method{}.csv".format('1');
    positive_path = base_path + r"\pick_method{}\feature_file\dataset_feature_file\dataset_disease_sim_{}_mer.csv".format('3','4')

    circRNA_gip_path = base_path + r"\circRna_gip_sim.csv"
    circRNA_fea_path = base_path + r"\feature_file\cirRna_feature_{}_mer.csv".format('4')
    disease_fea_path = base_path + r"\feature_file\disease_sim.csv"

    positive_fea_with_circRna_gip = r".\method_{}_dataset_disease_sim_{}_mer_circRna_gip.csv".format('3','4')
    not_select_fea_path = base_path + r"\pick_method{}\feature_file\not_select_dataset_feature_file\dataset_disease_sim_{}_mer.csv".format('1','4')
    pre_res_path = base_path + r"\pick_method{}\feature_file\not_select_dataset_feature_file\dataset_disease_sim_{}_mer_pred_res.csv".format('1','4')

    # fea_match(not_select_path, not_select_fea_path)
    # circRna_gip_matcher(positive_path, positive_fea_with_circRna_gip)

    df = pd.read_csv(pre_res_path)
    df.sort_values(inplace=True,by='predic_posi_proba',ascending=False)
    # colorectal cancer
    match_rule = (df['predic_res']==1) & (df['Disease Name'] == 'Lung cancer') & (df['predic_posi_proba']>0.98)
    print(df[match_rule])

