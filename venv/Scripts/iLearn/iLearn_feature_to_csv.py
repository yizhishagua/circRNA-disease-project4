'''
    将iLearn产生的csv特征文件, 加上列名和行标签
'''
import os
import threading
from tqdm import *
import pandas as pd
import numpy as np
class myThread(threading.Thread):
    def __init__(self, threadID, name, counter, rna_name_file, rna_feature_file, final_feature_file):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.rna_name_file = rna_name_file
        self.rna_feature_file = rna_feature_file
        self.final_feature_file = final_feature_file

    def run(self):
        self.write_feature_csv(self.rna_name_file, self.rna_name_file, self.final_feature_file)

def write_feature_csv(rna_name_file, rna_feature_file, final_feature_file):
    df_rna = pd.read_csv(rna_name_file, header=0, keep_default_na=False)
    rna_name_list = df_rna['circBase ID'].tolist()
    arr_fea = np.array(pd.read_csv(rna_feature_file, header=None))
    dic = {}
    dic['circBase ID'] = rna_name_list

    # 生成的特征从索引为1开始的
    for i in range(1, len(arr_fea[0])):
        dic['fea' + str(i - 1)] = arr_fea[:, i]
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(final_feature_file, header=True, index=False)
if __name__ == '__main__':

    base_file =  r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset"
    if os.getcwd() != base_file:
        os.chdir(base_file)
    rna_name_file = base_file + r"\circRna_name.csv"

    lamada_list = range(1, 11)
    k_mer_list = ['3', '4', '5', '6', '7']
    weight_list = ['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']

    # Autocorrelation method
    method_list = ['DAC', 'DCC', 'DACC']
    for method in method_list:
        for k in k_mer_list:
            rna_feature_file = base_file + r"\iLearn_file\circRna_fea_{}_{}.csv".format(method, k)
            final_feature_file = base_file + r"\feature_file\circRna_fea_{}_{}.csv".format(method, k)
            # if not os.path.exists(final_feature_file):
            write_feature_csv(rna_name_file, rna_feature_file, final_feature_file)
            print('completed')

    # PseKNC特征文件转换
    for lam in tqdm(lamada_list):
        for k in tqdm(k_mer_list):
            for weight in tqdm(weight_list):
                rna_feature_file = base_file +  r"\iLearn_file\circRna_fea_PseKNC_lamada{}_k{}_weight{}.csv".format(lam, k, weight)
                final_feature_file = base_file + r"\feature_file\circRna_fea_PseKNC_lamada{}_k{}_weight{}.csv".format(lam, k, weight)
                # if not  os.path.exists(final_feature_file):
                write_feature_csv(rna_name_file, rna_feature_file, final_feature_file)
                print('completed')




