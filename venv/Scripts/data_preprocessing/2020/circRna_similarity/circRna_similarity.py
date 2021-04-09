'''
    author: Shi-Hao Li
    date: 9, Nov. 2020
    des: calculate the similarity between circRNA including sequence similarity and functional similarity
'''
import pandas as pd
import numpy as np
import os
class  circRNA_similarity_calculator(object):
    def __init__(self, sequence_file, association_file, disease_similarity_file):
        self.df_circRna_sequence, \
        self.df_association, \
        self.df_disease_similarity = self.load_file(sequence_file, association_file, disease_similarity_file)
    def load_file(self, sequence_file, association_file, disease_similarity_file):
        '''
        以DataFrame的格式载入文件到内存中
        :param sequence_file: circRNA序列文件
        :param association_file: circRNA 和疾病的关联文件
        :param disease_similarity_file: 疾病相似性文件
        :return:
        '''
        return pd.read_csv(sequence_file, header=0, keep_default_na=False),\
               pd.read_csv(sequence_file, header=0, keep_default_na=False),\
               pd.read_csv(sequence_file, header=0, keep_default_na=False)

    def get_functional_similarity(self, ci, cj):
        '''
        计算两个Rna的功能相似性
        :param ci:
        :param cj:
        :return:
        '''
    def get_sequence_similarity(self, ci, cj):
        '''
        use leivenst edit distance to compute similarity between ci and cj
        计算两个Rna的序列相似性
        :param ci:
        :param cj:
        :return:
        '''
        pass
if __name__ == '__main__':
    base_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset"
    if os.getcwd() != base_file:
        os.chdir(base_file)
    sequence_file = r"\circRna_name.csv"
    association_file = r"\association_file.csv"
    disease_similarity_file = r"\feature_file\disease_sim.csv"