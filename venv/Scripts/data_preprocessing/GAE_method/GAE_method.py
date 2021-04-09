'''
    date: 2021/4/4 17:03
    this method take integrated disease similarity circRNA_4_mer_gip feature into GAE, respectively.
'''
import pandas as pd
import numpy as np
import os

class Feature_loader(object):
    def __init__(self, rna_file, disease_file):
        self.seq_df = pd.DataFrame(self.load_RNA_fea(rna_file)).transpose()
        self.dis_df = pd.DataFrame(self.load_disease_fea(disease_file)).transpose()
    def load_RNA_fea(self, file):
        '''
        加载circRNA的特征并存入字典中
        :param file: rna嵌入节点特征
        :return:
        '''
        dic={}
        with open(file, 'r') as f:
            for line in f:
                if ":" in line:
                    sample = line.split(":")
                    name = sample[0]
                    fea = [ eval(i) for i in sample[1].split(' ')]
                    dic[name] = fea
        return dic
    def load_disease_fea(self, file):
        return self.load_RNA_fea(file)
    def clear(self, file):
        '''
        清理原始的特征列
        :param file:
        :return:
        '''
        df = pd.read_csv(file)
        df = df[['label','Disease Name', 'circBase ID']]
        df.to_csv(file)
        return
    def fea_match(self, file):
        '''
        将样本特征匹配过去
        :param file:
        :return:
        '''
        df = pd.read_csv(file, index_col=0)
        self.dis_df.index.name='Disease Name'
        self.seq_df.index.name='circBase ID'
        df = df.merge(self.dis_df,how='left',left_on='Disease Name',right_on='Disease Name')
        df = df.merge(self.seq_df,how='left',left_on='circBase ID',right_on='circBase ID')
        df.to_csv(file)
if __name__ == '__main__':
    circRNA_fea_file = r"./cirRna_GAE_node_emb_ep5000_split.txt"
    disease_fea_file = r"./disease_GAE_node_emb_ep2000_split.txt"
    feaLoader = Feature_loader(circRNA_fea_file, disease_fea_file)
    for i in [1, 2, 3, 'random']:
        # feaLoader.clear(r"./method{}_sample_picked.csv".format(i))
        feaLoader.fea_match(r"./method{}_sample_picked.csv".format(i))

