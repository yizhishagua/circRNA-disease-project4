'''
    enconding: utf-8
    author: Shi-Hao Li
    date: 25, Dec 11
    des: 将特征和样本匹配起来
'''
import pandas as pd
import os
from tqdm import *
import csv
import re
class Feature_Matcher(object):
    def __init__(self):
        self.disease_mapper = self.load_disease_dic(disease_name_mapper_file)
    def load_disease_dic(self, disease_name_mapper_file):
        '''
        加载原始疾病名字和MeSH中的名字的mapper文件
        :param disease_name_mapper_file:
        :return:
        '''
        try:
            disease_mapper = {}
            df_disease_mapper = pd.read_csv(disease_name_mapper_file, header=0, keep_default_na=False)[
                ['Ori_Name', 'MeSH_Name']]
            for i in df_disease_mapper.index:
                ori_name = str(df_disease_mapper.loc[i]['Ori_Name'])
                mesh_name = str(df_disease_mapper.loc[i]['MeSH_Name'])
                disease_mapper[ori_name] = mesh_name
            return disease_mapper
        except FileNotFoundError:
            print('plz check your file exists or not')
    # 载入每个样本的疾病相似性特征
    def get_samples_with_feature(self, dataset_file, rna_fea_file, disease_fea_file, tar_file):
        '''
        将输入的dataset_file的文件打上特征后输入到tar_file中
        采用表的内外连接就可以了啊
        :param dataset: 样本文件
        :param rna_fea_file: rna的特征文件
        :param disease_fea_file: 疾病的特征文件
        :param tar_file: 写入的目标文件
        :return:
        '''
        df = pd.read_csv(dataset_file, header=0, keep_default_na=False)
        df_rna = pd.read_csv(rna_fea_file, header=0, keep_default_na=False)
        if 'sequence' in df_rna.keys():
            df_rna = df_rna.drop('sequence', axis = 1)
        df_disease = pd.read_csv(disease_fea_file, header=0, keep_default_na=False)
        df_disease.rename(columns = {'fea': 'Disease Name'}, inplace=True)

        merge_rna = pd.merge(df, df_rna, left_on='circBase ID', right_on='circBase ID', sort=False, how='left')
        merge_rna_disease = pd.merge(merge_rna, df_disease, left_on='Disease Name', right_on='Disease Name', sort=False, how='left')
        if not os.path.exists(tar_file):
            merge_rna_disease.to_csv(tar_file, header=True, index=False)
        return 1

    # 载入每个样本疾病的mesh2vec特征
    def load_mesh_deep_walk_feature(self, dataset_file, rna_fea_file ,mesh2vec_file, dataset_with_feature_file):
        '''
        产生样本集中的每一个样本特征, 为rna序列特征 + 疾病的mesh_fea_deep_walk_feature
        :param data_set_file:
        :param rna_feature_file: rna的特征文件
        :param mesh2vec_file: mesh_deep_walk特征文件
        :param dataset_with_feature_file: 输出的具有特征的样本文件
        :return:
        '''
        df = pd.read_csv(mesh2vec_file, header=None)
        df_dataset = pd.read_csv(dataset_file, header=0, keep_default_na=False)
        df_disease = df_dataset.drop_duplicates(['Disease Name'], keep='first').reset_index(drop=True)[['Disease Name']]
        df_rna = pd.read_csv(rna_fea_file, header=0, keep_default_na=False)
        if 'sequence' in df_rna.keys():
            df_rna = df_rna.drop('sequence', axis = 1)
        # 给疾病的df增加mesh_name列, 便于merge查特征
        df_disease['mesh_name']=None
        for i in df_disease.index:
            df_disease.loc[i, 'mesh_name'] = self.disease_mapper[df_disease.loc[i]['Disease Name']]

        # 更换列名
        df.rename(columns={0: 'mesh_name'}, inplace=True)
        df_disease_mesh_merge = pd.merge(df_disease, df, left_on='mesh_name', right_on='mesh_name', sort=False, how='left')

        df_dataset_disease_merge = pd.merge(df_dataset, df_disease_mesh_merge, left_on='Disease Name', right_on='Disease Name', sort=False, how='left')
        df_dataset_disease_rna_merge = pd.merge(df_dataset_disease_merge, df_rna, left_on='circBase ID', right_on='circBase ID', sort=False, how='left')

        df_dataset_disease_rna_merge = df_dataset_disease_rna_merge[df_dataset_disease_rna_merge['mesh_name'] != '0']
        if not os.path.exists(dataset_with_feature_file):
            df_dataset_disease_rna_merge.to_csv(dataset_with_feature_file, header=True, index=False)

    # csv_to_arff_transformation
    def trasform(self, csv_file, arff_file, label_index, fea_start_index):
        '''
        将特征文件转化为arff
        注意attribute名字中包含空格要进行替换
        :param csv_file:
        :param arff_file:
        :param label_index: label所在的列索引
        :param fea_start_index: 第一维特征的索引
        :return:
        '''
        data = []
        # if os.path.exists(arff_file):
        #     return
        with open(csv_file, 'r', encoding='utf-8') as f1:
            f1.read = csv.reader(f1)
            for row in f1.read:
                data.append(row)
            f1.close()
        f2 = open(arff_file, 'w+', encoding='utf-8')
        f2.write('@relation association\n\n')
        for item in data[0][int(fea_start_index):]:
            if ' ' in item:
                item = re.sub(' ', '_', item)
            f2.write('@attribute {} numeric\n'.format(item))
        f2.write('@attribute Label {0,1}\n\n@data\n')
        for i in range(1, len(data)):
            for j in data[i][int(fea_start_index):]:
                f2.write(j + ',')
            # 在每一行的后面跟上标签
            f2.write(data[i][int(label_index)])
            f2.write('\n')
        f2.close()
if __name__ == '__main__':
    base_file =  r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset'
    if not os.getcwd() == base_file:
        os.chdir(base_file)

    # 疾病的sim1+sim2+gip feature
    disease_name_mapper_file = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\disease_name\OriDisNam_to_MeSHNameID.csv'

    feature_matcher = Feature_Matcher()
    disease_fea_list = ['disease_sim', 'mesh2vec']

    lamada_list = range(1, 11)
    k_mer_list = ['3', '4', '5', '6']
    weight_list = ['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']

    for disease_fea in disease_fea_list:
        # kmer的疾病相似性  ||  kmer mesh2vec
        # for k in ['1', '2', '3', '4', '5', '6']:
        for k in [ '4']:
            # for m in ['1', '2', '3']:
            for m in ['random']:
                # dataset_file = base_file + r'\picked_dataset_after_rank_method{}.csv'.format(m)
                dataset_file = base_file + r'\picked_dataset_{}0.csv'.format(m)

                # RNAfeature记得改,格式要一样哟
                rna_fea_file = base_file + r'\feature_file\cirRna_feature_{}_mer_tanimoto_gip.csv'.format(k)
                disease_fea_file = base_file + r'\feature_file\{}.csv'.format(disease_fea)
                dataset_with_feature_file = base_file + r'\pick_method{}\feature_file\dataset_feature_file\dataset_{}_{}_mer_tanimoto_gip.csv'.format(m, disease_fea, k)
                arff_file = base_file + r'\pick_method{}\feature_file\dataset_feature_file\arff_file\dataset_{}_{}_mer_tanimoto_gip.csv.arff'.format(m, disease_fea, k)
                if disease_fea == 'disease_sim':
                    feature_matcher.get_samples_with_feature(dataset_file, rna_fea_file, disease_fea_file, dataset_with_feature_file)
                    feature_matcher.trasform(dataset_with_feature_file, arff_file, 3, 4)
                else:
                    feature_matcher.load_mesh_deep_walk_feature(dataset_file, rna_fea_file, disease_fea_file, dataset_with_feature_file)
                    feature_matcher.trasform(dataset_with_feature_file, arff_file, 3, 5)
                print("{}, {}_mer match completed".format(disease_fea, k))

        # acc + disease_sim || acc + mesh2vec
        # method_list = ['DAC', 'DCC', 'DACC']
        # for method in method_list:
        #     for k in k_mer_list:
        #         for m in ['1', '2', '3']:
        #             dataset_file = base_file + r'\picked_dataset_after_rank_method{}.csv'.format(m)
        #             rna_fea_file = base_file + r'\feature_file\circRna_fea_{}_{}.csv'.format(method,k)
        #             disease_fea_file = base_file + r'\feature_file\{}.csv'.format(disease_fea)
        #             dataset_with_feature_file = base_file + r'\pick_method{}\feature_file\dataset_feature_file\dataset_{}_{}_{}_mer.csv'.format(m, disease_fea, method, k)
        #             arff_file = base_file + r'\pick_method{}\feature_file\dataset_feature_file\arff_file\dataset_{}_{}_{}_mer.arff'.format(m, disease_fea, method, k)
        #             if disease_fea == 'disease_sim':
        #                 feature_matcher.get_samples_with_feature(dataset_file, rna_fea_file, disease_fea_file, dataset_with_feature_file)
        #                 feature_matcher.trasform(dataset_with_feature_file, arff_file, 3, 4)
        #             else:
        #                 feature_matcher.load_mesh_deep_walk_feature(dataset_file, rna_fea_file, disease_fea_file, dataset_with_feature_file)
        #                 feature_matcher.trasform(dataset_with_feature_file, arff_file, 3, 5)
        #             print('{}_{}_{}_completed'.format(disease_fea, method, k))
        #
        # # PseKNC + disease_sim || PseKNC + mesh2vec
        # for lam in tqdm(lamada_list):
        #     for k in tqdm(k_mer_list):
        #         for weight in tqdm(weight_list):
        #             for m in ['1', '2', '3']:
        #                 dataset_file = base_file + r'\picked_dataset_after_rank_method{}.csv'.format(m)
        #                 rna_fea_file = base_file + r'\feature_file\circRna_fea_PseKNC_lamada{}_k{}_weight{}.csv'.format(lam, k, weight)
        #                 disease_fea_file = base_file + r'\feature_file\{}.csv'.format(disease_fea)
        #                 dataset_with_feature_file = base_file + r'\pick_method{}\feature_file\dataset_feature_file\dataset_PseKNC_{}_lamada{}_k{}_weight{}.csv'.format(m, disease_fea, lam, k, weight)
        #                 arff_file = base_file + r'\pick_method{}\feature_file\dataset_feature_file\arff_file\dataset_PseKNC_{}_lamada{}_k{}_weight{}.arff'.format(m, disease_fea, lam, k, weight)
        #                 if disease_fea == 'disease_sim':
        #                     feature_matcher.get_samples_with_feature(dataset_file, rna_fea_file, disease_fea_file, dataset_with_feature_file)
        #                     feature_matcher.trasform(dataset_with_feature_file, arff_file, 3, 4)
        #                 else:
        #                     feature_matcher.load_mesh_deep_walk_feature(dataset_file, rna_fea_file, disease_fea_file, dataset_with_feature_file)
        #                     feature_matcher.trasform(dataset_with_feature_file, arff_file, 3, 5)
        #                 print('PseKNC_{}_lamada{}_k{}_weight{} completed'.format(disease_fea, lam, k, weight))


