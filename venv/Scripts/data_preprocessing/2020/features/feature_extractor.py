'''
        author: Shi-Hao Li
        date: 23. Dec 2020
'''
import re
import os
import pandas as pd
import numpy as np
from tqdm import *
from MeshTreeHelper import MeshTreeHelper

class Feature_extractor(object):
    disease_mapper = {}
    def __init__(self, disease_mapper_file):
        # Mesh树的查询词典
        self.mesh_tree_helper = MeshTreeHelper()
        # 疾病原始名字和Mesh名字的查询字典
        self.disease_mapper = self.load_disease_dic(disease_name_mapper_file)

    # 对象初始化时, 调用该方法, 自动加载mesh树的词典
    def load_disease_dic(self, disease_name_mapper_file):
        '''
        加载原始疾病名字和MeSH中的名字的mapper文件
        :param disease_name_mapper_file:
        :return:
        '''
        try:
            disease_mapper = {}
            df_disease_mapper = pd.read_csv(disease_name_mapper_file, header=0, keep_default_na=False)[['Ori_Name', 'MeSH_Name']]
            for i in df_disease_mapper.index:
                ori_name = str(df_disease_mapper.loc[i]['Ori_Name'])
                mesh_name = str(df_disease_mapper.loc[i]['MeSH_Name'])
                disease_mapper[ori_name] = mesh_name
            return disease_mapper
        except FileNotFoundError:
            print('plz check your file exists or not')

    # 提取序列特征的封装方法, 输入序列加提起特征的方式对象
    def extract(self, sequence, Feature_Strategy):
        '''
        输出一条序列，提取其特征
        :param sequence:
        :param feature_strategy: 特征提取策略的对象, 如ker, k-space等等
        :return:
        '''
        if not isinstance(Feature_Strategy, Feature_Strategy_base):
            raise TypeError('plz input the son of Feature_Strategy class')
        return Feature_Strategy.extract(sequence)

    def load_seq_file(self, seq_file, col_name):
        '''
        输入有序列的csv文件, 并指定序列所在的列的名字第一行必须是表头
        :param seq_file:
        :param col_name: 序列所在的列名
        :return:
        '''
        df = pd.read_csv(seq_file, head=0, keep_default_na=False)
        return df
    def get_fea_file(self, input_file, col_namem, feature_strategy, output_file):
        '''
        输入含有序列的文件, 将新生成的特征文件写入之前的序列对应的下一列
        :param input_file: 包含序列的样本文件
        :param col_namem: 序列所在列的列名
        :param feature_strategy:
        :param output_file: 有特征值的文件
        :return:
        '''
        df = self.load_seq_file(input_file, col_namem)
        for i in df.index:
           fea_matrix =  self.extract(df[col_namem], feature_strategy)
           for j in range(len(fea_matrix)):
               df.loc[i,'fea'+str(j)] = fea_matrix[j]
        if not os.path.exists(output_file):
            df.to_csv(out_put_file, header=True, index = False)

    # 基于疾病两种语义相似性+GIP的计算法方法，生成输入样本疾病的相似性特征文件
    def disease_feature_extract(self, disease_name_file, sim_file1, sim_file2, sim_gip_file, disease_feature_file):
        '''
        :param disease_name_file:
        :param sim_file:
        :return:
        '''
        try:
            graph_list = []
            df_disease = pd.read_csv(disease_name_file, header=0, keep_default_na=False)
            disease_name_list = []
            for i in df_disease.index:
                disease_name = self.disease_mapper[df_disease.loc[i]['Disease Name']]
                if str(disease_name) == '0':
                    continue
                disease_dic = {disease_name: self.mesh_tree_helper.meshDic[disease_name]}
                g = self.mesh_tree_helper.getDiGraph(disease_name, disease_dic)
                disease_name_list.append(df_disease.loc[i]['Disease Name'])
                graph_list.append(g)

            # 过滤第一遍, 给每个图加上第一种语义相似性
            for g in graph_list:
                self.mesh_tree_helper.getSemanticContribution(g)
            df1 = self.generate_disease_sim_file(sim_file1, graph_list, disease_name_list)

            # 过滤第二遍, 给每个图加上第二种语义相似性
            for g in graph_list:
                self.mesh_tree_helper.get_semantic_contribution2(graph_list, g)
            df2 = self.generate_disease_sim_file(sim_file2, graph_list, disease_name_list)

            # 第三遍加上GIP相似性
            # GIP用GIPCalculator计算完成
            df_gip = pd.read_csv(sim_gip_file, header=0, keep_default_na=False, index_col=0)

            # df_sim的值为两种语义相似性df的均值
            df_sim = (df1 + df2)/2
            disease_name_list = df_disease['Disease Name'].values.tolist()
            dic_disease_feature = {'fea': disease_name_list}
            fea_matrix = []

            for d1 in disease_name_list:
                fea_list = []
                for d2 in disease_name_list:
                    # 如果两种疾病中任意一种查不到MeSH ID的话, 则使用GIP相似性代替
                    if str(self.disease_mapper[d1]) == '0' or str(self.disease_mapper[d2]) == '0':
                        temp = float(df_gip.loc[d1][d2])
                    # 如果都能查到MeSH ID但是语义相似性为0
                    elif float(df_sim.loc[d1][d2]) == 0.0:
                        temp = float(df_gip.loc[d1][d2])
                    else:
                        temp = float(df_sim.loc[d1][d2])
                    fea_list.append(temp)
                fea_matrix.append(fea_list)
            for i in range(len(disease_name_list)):
                dic_disease_feature[disease_name_list[i]] = np.array(fea_matrix)[:,i]
            df_disease_feature = pd.DataFrame.from_dict(dic_disease_feature)
            if not os.path.exists(disease_feature_file):
                df_disease_feature.to_csv(disease_feature_file, header=True,index=False)
        except FileNotFoundError:
            print('plz check whether your file is existed')

    # 得到所有疾病的语义相似性+GIP相似性的相似性文件
    def generate_disease_sim_file(self, sim_file, graph_list, disease_name_list):
        '''
        根据输入的一个DAGs的列表, 计算他们的相似性，并写入到sim_file中
        :param sim_file:
        :param graph_list: 输入的图的列表
        :param disease_name_list: 原始的输入构图的疾病列表
        :return:
        '''
        sim_matrix = []
        for g1 in graph_list:
            sim_list = []
            for g2 in graph_list:
                sim = self.mesh_tree_helper.getSemanticSimilarity(g1, g2)
                sim_list.append(sim)
            sim_matrix.append(sim_list)
        dic = {'similarity': disease_name_list}

        for i in range(len(disease_name_list)):
            # 注意有重复的键,
            dic[str(disease_name_list[i])] = np.array(sim_matrix)[:,i]
        df = pd.DataFrame.from_dict(dic)
        if not os.path.exists(sim_file):
            df.to_csv(sim_file, header=True, index=False)
        df = df.set_index('similarity', drop=True, inplace=False, verify_integrity=False)
        return df
class Feature_Strategy_base(object):
        '''
            des: 特征提取的基类, 只用于被子类继承
        '''
        def __init__(self):
            pass
        def extract(self, sequence):
            '''
                子类必须实现的方法，对于输入的一条序列，提取特征, 返回特征组成的向量
            :param sequence:
            :return: 序列的特征表示
            '''
class K_mer(Feature_Strategy_base):
    k = 0
    def __init__(self, k):
        '''
        初始化对象的时候就输入提取k摩尔特征
        :param k: 取几mer
        '''
        self.k = k
    def extract(self, sequence):
        '''
        对父类extract方法的重写
        :param sequence:
        :return:
        '''
        # 检查输入是否合法
        try:
            self.check_sequence(sequence)
        except TypeError:
            raise TypeError('plz check your type of input')
        except ValueError:
            raise ValueError('plz check your sequence is valid or not')
        except ArithmeticError:
            raise ArithmeticError('')

        # AGCT序列转化成数字序列0123
        number = [0 for i in sequence]
        for i in range(len(sequence)):
            number[i] = self.trans_numer(sequence[i])

        feature_vec = [0] * 4 ** self.k

        for i in range(len(number)):
            index = self.cal_position(number[i: i + self.k])
            feature_vec[index] += 1

        for i in range(len(feature_vec)):
            try:
                feature_vec[i] /= float(len(number) - self.k + 1)
            except ArithmeticError:
                raise ArithmeticError('分母为0')
        return feature_vec
    def trans_numer(self, char):
        '''
        将输入的AGCT字符转换成0123
        :param char:
        :return:字符对应的数字
        '''
        if char == 'A':
            return 0
        elif char == 'G':
            return 1
        elif char =='C':
            return 2
        elif char =='T':
            return 3
        else:
            raise ValueError('plz check your input character is right or not')
    def cal_position(self, mer):
        '''
        输入一段0123的序列，计算这段k-mer特征所处的索引位置
        :param mer: 当前的这段小片段
        :return: index
        '''
        index = 0
        for i in range(len(mer)):
            # 例如2-mer 01 对应的2-mer是AG, 特征所在的索引是 4*(4*0+0) + 1 = 1
            # 11 ==> GG, 4*(4*0 + 1) + 1 = 5
            # 111 ==> GGG, 4*(4*(4*0 + 1)+1)+1 = 21
            index = 4 * index + mer[i]
        return index
    def check_sequence(self, sequence):
        '''
        检查输入序列是否符合该算法需要的格式
        :return:
        '''
        if isinstance(sequence, str):
            # 使用正则表达式匹配输入字符AGCT中任意字符出现多次其他字符均不合法
            # 从字符串开始匹配到结尾对大小写敏感的多次匹配A或G或C或T
            pattern = re.compile(r'^(A|G|C|T)+$')
            res = re.match(pattern, sequence)
            if res == None:
                raise ValueError('invalid inputs')
        else:
            raise TypeError('plz input the str-like variance')
    def get_feature_vec(self):
        return self.feature_vec
def circRna_feature_extract_kmer(circRna_name_file, m):
    '''
    根据Rna带有序列的名字文件提取kmer特征
    :param circRna_name_file:
    :param k: k的最大取值
    :return:
    '''
    df_circRna = pd.read_csv(circRna_name_file, header=0, keep_default_na=False)
    for k in tqdm(range(1, m+1)):
        path = base_file + r"\third_data\final_dataset\feature_file\cirRna_feature_{}_mer.csv".format(k)
        if  os.path.exists(path):
            continue
        for index in df_circRna.index:
            feature_list = feature_extractor.extract(df_circRna.loc[index]['sequence'], K_mer(k))
            for i  in range(len(feature_list)):
                df_circRna.loc[index, 'fea' + str(i)] = feature_list[i]
        df_circRna.to_csv(path, header=True, index=False)

if __name__ == '__main__':
    base_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association"
    # 改变工作目录
    if os.getcwd() != base_file:
        os.chdir(base_file)

    disease_name_mapper_file = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\disease_name\OriDisNam_to_MeSHNameID.csv'
    all_data_set_file = r'\third_data\final_dataset\all_dataset'

    # 总的样本文件(随机选择的负样本, 不包含特征向量的样本)
    sample_with_seq_file = r'\third_data\final_dataset\all_dataset'

    # Rna的名字文件
    circRna_name_file = base_file + r"\third_data\final_dataset\circRna_name.csv"

    # 疾病的相似性文件
    disease_similarity_file1 = base_file + r"\third_data\final_dataset\disease_sim1.csv"
    disease_similarity_file2 = base_file + r"\third_data\final_dataset\disease_sim2.csv"
    disease_name_file = base_file + r"\third_data\final_dataset\disease_name.csv"
    disease_gip_sim_file = base_file + r"\third_data\final_dataset\disease_gip_sim.csv"

    # 正样本中疾病的相似性特征, 综合两种语义特征和GIP特征, 如若不行, 分开试一下
    disease_feature_file = base_file + r"\third_data\final_dataset\feature_file\disease_sim.csv"

    # 包含特征向量的，随机选择负样本的所有样本的csv文件(circBase ID, Disease Name, sequence, feature_1, feature_2, ... , feature_n)
    sample_with_feature_file = base_file + r"\third_data\final_dataset\random_negtive_samples_with_feature.csv"

    # 初始化对象, 加载疾病名字mapper文件
    feature_extractor = Feature_extractor(disease_name_mapper_file)

    # 提取疾病特征
    feature_extractor.disease_feature_extract(disease_name_file,
                                              disease_similarity_file1,
                                              disease_similarity_file2,
                                              disease_gip_sim_file,
                                              disease_feature_file)

    # 提取circRna特征:
    # 1. k_mer特征 k取1-6
    circRna_feature_extract_kmer(circRna_name_file, 6)

