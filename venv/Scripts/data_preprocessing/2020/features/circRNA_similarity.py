'''
    date:2021/3/31
    Tanimoto_similarity
    d(x, y) = x·y/x^2 + y^2-x·y
'''
import numpy as np
import pandas as pd
import os
import Levenshtein
def tanimoto(x, y):
    '''
    :param x: arr_vector
    :param y: arr_vector
    :return:
    '''
    inner_product = x.dot(y)
    res = inner_product/(x.dot(x)+y.dot(y)-inner_product)
    return res
def levenshtein_sim(fea_file, sim_file):
    df = pd.read_csv(fea_file, index_col=0)
    df.sort_index(axis=0, inplace= True)
    df.sort_index(axis=1, inplace= True)
    arr = np.zeros((len(df.index), len(df.index)))
    for i in range(len(df.index)):
        for j in range(len(df.index)):
            arr[i][j] = Levenshtein.ratio(df.iloc[i]['sequence'], df.iloc[j]['sequence'])
    df_res = pd.DataFrame(arr, columns=df.index.tolist())
    df_res.to_csv(sim_file)
def circ_tanimoto_sim(fea_file, sim_file):
    if(os.path.exists(sim_file)):
       return pd.read_csv(sim_file,index_col=0)

    df = pd.read_csv(fea_file)
    arr = np.array(df)[:,2:]
    sim_res = np.zeros((arr.shape[0], arr.shape[0]))
    circ_name = df.iloc[:,0].tolist()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            sim_res[i][j] = tanimoto(arr[i], arr[j])

    df_sim_res = pd.DataFrame(sim_res, index=circ_name, columns=circ_name)
    df_sim_res.sort_index(axis=1,inplace=True)
    df_sim_res.sort_index(axis=0,inplace=True)
    df_sim_res.to_csv(sim_file)
    return df_sim_res
def circ_sim_tanimoto_gip(tanimoto_sim_file, circ_gip_file, int_sim_file):
    '''
    Tanimoto相似性和circRNA gip 整合相似性
    :param tanimoto_sim_file: Tanimoto相似性文件
    :param circ_gip_file: gip相似性文件
    :param int_sim_file: 整合后的相似性文件
    :return:
    '''
    tanimoto_sim = circ_tanimoto_sim(circ_fea_file, tanimoto_sim_file)
    gip_sim = pd.read_csv(circ_gip_file, index_col='GIP_value')
    int_sim = (gip_sim+tanimoto_sim )/2
    int_sim.index.name = 'circBase ID'
    int_sim.to_csv(int_sim_file, index = 1)
if __name__ == '__main__':
    circ_fea_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\feature_file\cirRna_feature_4_mer.csv"
    circ_gip_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\circRna_gip_sim.csv"
    tanimoto_sim_file = r".\cirRna_feature_4_mer_tanimoto.csv"
    int_sim_file = r".\cirRna_feature_4_mer_tanimoto_gip.csv"
    # circ_sim_tanimoto_gip(tanimoto_sim_file, circ_gip_file, int_sim_file)
    levenshtein_sim(circ_fea_file, "./circRNA_levenshtein.csv")