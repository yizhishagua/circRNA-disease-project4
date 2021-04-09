'''
    date: 2021/04/06
    match the score of association between circRNA and disease of Circ2Disease
'''
class Mapper(object):
    '''
        原始疾病名字的mapper对象
    '''
    def __init__(self, file):
        self.df_mapper = self.load_dic(file)
    def load_dic(self, file):
        return pd.read_csv(file, index_col=0)
def transform(adj_file, experimental_file, mapper_file):
    '''
        将adj_matrix中的1用MNDR中的Score字段的值表示
    :param file:
    :param experimental_file: MNDR数据库circRNA与疾病的子库
    :param mapper_file: 原始疾病名字与MeSH ID的互相转换
    :return:
    '''
    mapper = Mapper(mapper_file)
    df_mapper = mapper.df_mapper
    df_experimental = pd.read_csv(experimental_file, delimiter='\t', encoding='gb2312')
    df_adj = pd.read_csv(adj_file, index_col=0)
    for index in df_adj.index:
        for column in df_adj.columns:
            if df_adj.loc[index, column] == 1: #对原始存在关联的疾病进行分数加持
                # MNRD没查到对应的circRNA，则关系用2表示。如果查到了则去查有没有疾病的MeSH_ID
                if index not in df_experimental['ncRNA Symbol'].to_list():
                    df_adj.loc[index, column] = 2 #提前截断加快速度
                    continue
                if df_mapper.loc[column, 'MeSH_ID'] == 0: #否则取查疾病是否存在我自己找的MeSH_ID, 如果为0再按疾病的名字字段去查, 如果有则用MeSH_ID去查
                    match_rule = (df_experimental['ncRNA Symbol']==index) & (df_experimental['Disease'] == column)
                else:
                    match_rule = (df_experimental['ncRNA Symbol']==index) & (df_experimental['MeSH ID'] == df_mapper.loc[column, 'MeSH_ID'])
                score = df_experimental[match_rule]['Score']
                if len(score) == 0: #如果查到了空的serial怎么办? 说明疾病搜不到呀，则用3表示
                    score = 3
                    df_adj.loc[index, column] = 3
                    continue #提前下一跳
                #如果查到了多条记录则用score的平均值来表示
                if len(score) >1:
                    score = np.mean(score)
                df_adj.loc[index, column] = float(score)
    df_adj.to_csv(r"./adj_score_file.csv")
def process_score_not_found(file):
    df = pd.read_csv(file, index_col=0)
    for index in df.index:
        for column in df.columns:
            ele = df.loc[index, column]
            if ele >= 2:
                #那就设为特别小的score, 设为0.1? 表明是人为的收集误差? 或许应该由其邻居来决定，先就这样把。
                df.loc[index, column] = 0.1
    df.to_csv(file)
def merge_and_concat(rna_sim_file, disease_sim_file, adj_with_score_file):
    '''
        将rna和adj连接
        疾病和adj连接
        上述两个连接后的df再concat
    :param rna_sim_file:
    :param disease_sim_file:
    :param adj_with_score_file:
    :return:
    '''
    df_rna = pd.read_csv(rna_sim_file)
    df_dis = pd.read_csv(disease_sim_file)
    df_adj = pd.read_csv(adj_with_score_file)
    # rna右连adj
    df_rna_adj = pd.merge(df_rna, df_adj, left_on='circBase ID',right_on='label',how='inner')
    df_rna_adj.drop(columns='label', axis=1, inplace=True)
    # df_rna_adj.to_csv(r"./df_rna_adj.csv")
    # 转置之后疾病右连过去
    df_adj.index = df_adj['label']
    df_adj.drop(columns=['label'], axis=1, inplace=True)
    df_adj = df_adj.transpose()
    df_adj.columns.name=None
    df_adj.index.name = 'Disease Name'
    df_adj = df_adj.reset_index()

    df_adj_dis = pd.merge(df_adj, df_dis, left_on='Disease Name',right_on='fea',how='inner')
    df_adj_dis.drop(columns='fea', inplace = True, axis=1)

    df_rna_adj.index = df_rna_adj['circBase ID']
    df_rna_adj.index.name=None
    df_rna_adj.drop(columns=['circBase ID'], inplace=True, axis=1)

    df_adj_dis.index = df_adj_dis['Disease Name']
    df_adj_dis.index.name=None
    df_adj_dis.drop(columns=['Disease Name'], inplace=True, axis=1)
    df = pd.concat([df_rna_adj, df_adj_dis])
    df.to_csv(r"./circRNA_disease_whole_network.csv")
import pandas as pd
import numpy as np
if __name__ == '__main__':
    experimental_file = r"./Experimental circRNA-disease information.tsv"
    mapper_file = r"./OriDisNam_to_MeSHNameID.csv"
    rna_sim_file = r"./sorted_cirRna_feature_4_mer_tanimoto_gip.csv"
    disease_sim_file = r"./sorted_disease_sim.csv"
    adj_with_score_file = r"./adj_score_file.csv"
    # transform(r"./sorted_adj_matrix.csv", experimental_file, mapper_file)
    # process_score_not_found(r"./adj_score_file.csv")
    merge_and_concat(rna_sim_file, disease_sim_file, adj_with_score_file)