'''
    author: Shi-Hao Li
    date: 20, Dec. 2020
    des: 从原始的关联文件中得到包含人类circRNA和疾病且circRNA有gene symbol(便于查找序列)
'''
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import random
import copy
class Samples(object):
    # 非重复的cirRna的文件（原始名字, 疾病名字, circBase ID, 序列, 长度）
    base_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association"
    circRna_name_file = base_file + r"\third_data\final_dataset\circRna_name.csv"
    disease_name_file = base_file + r"\third_data\final_dataset\disease_name.csv"
    disease_sim_file = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\feature_file\disease_sim.csv'

    def __init__(self, hg19_file, ID_file):
        '''
        生成对象的时候就加载文件
        :param hg19_file:
        '''
        self.seq_dic = self.get_seq_dic(hg19_file)
        self.ID_trans_dic = self.load_trans_dic(ID_file)
        self.df_disease_sim = self.load_disease_sim(self.disease_sim_file)

    def load_disease_sim(self, disease_sim_file):
        '''
        加载疾病相似性特征为df, (sim1+sim2)/2 + gip
        :param disease_sim_file:
        :return:
        '''
        return pd.read_csv(disease_sim_file, header=0, keep_default_na=False, index_col=0)
    def load_file(self, file_name, target_file, gene_symbol_file, has_seq_circRna_file, non_repeative_disease_file):
        '''
        加载原始关联文件, 并且筛选出有Gene Symbol的, 有关联的Human样本
        :param file_name: base_file + ori_ass_file,
        :param target_file: 生成的文件
        :param gene_symbol_file: 需要去查的gene symbol文件
        :param has_seq_circRna_file: 能直接拿到序列的Rna文件
        :param non_repeative_disease_file: 非重复的疾病的名字文件
        :return: 总共需要查找的gene_symbol的df
        '''
        ori_data = pd.read_csv(file_name, sep=',', header=0, keep_default_na=False)

        # 规则同时满足才输出
        # 规则: Human
        # Unclear 表示有关联但是不清楚是上调还是下调
        # (ori_data['Gene Symbol'] != 'N/A') &

        match_rule = (ori_data['Species'] == 'Human') & (ori_data['Brief description'] != 'Unclear')
        ori_data = ori_data[match_rule]

        # 需要哪些列的信息
        need_info = ['circRNA Name', 'Gene Symbol', 'CircBase Link', 'Disease Name']
        data = ori_data[need_info]

        #找到包含疾病名字列包含'Alzhei'串的行号, 将其值修改为Alzheimers disease
        index_list = data.loc[data['Disease Name'].str.contains('Alzhei')].index
        for i in index_list:
            data.loc[i]['Disease Name'] = 'Alzheimers disease'

        # 去掉Gene Symbol中每一条的空格
        index_list = data.loc[data['Gene Symbol'].str.contains(' ')].index
        for i in index_list:
            data.loc[i]['Gene Symbol'] = data.loc[i]['Gene Symbol'].strip()

        # 去掉重复的行, 保留第一次出现的值
        data = data.drop_duplicates(keep='first')

        if not os.path.exists(target_file):
            data.to_csv(target_file, index=False, header=True)

        non_repetive_circRna_name = data.drop_duplicates(['circRNA Name'], keep = 'first')
        non_repetive_disease_name = data.drop_duplicates(['Disease Name'], keep = 'first')['Disease Name']

        # 需要查找进行ID转换的样本
        circRna_need_query = non_repetive_circRna_name[non_repetive_circRna_name['CircBase Link'] == 'N/A']

        # 有circBase链接的circRna
        non_repetive_circRna_name = non_repetive_circRna_name[non_repetive_circRna_name['CircBase Link'] != 'N/A']

        # if not os.path.exists(gene_symbol_file):
        #     circRna_need_query.to_csv(gene_symbol_file, index=False, header=False)

        if not os.path.exists(has_seq_circRna_file):
            non_repetive_circRna_name.to_csv(has_seq_circRna_file, index=False, header=True)

        if not os.path.exists(non_repeative_disease_file):
            non_repetive_disease_name.to_csv(non_repeative_disease_file, index=False, header=True)

        return circRna_need_query

    # 加载6位ID转7位ID的文件到内存中
    def load_trans_dic(self, ID_file):
        '''
        将6位hsa_circRNA_123456 按照ID转换成7位的circBaseID, 例如: hsa_circ_1234567
        :param need_find_file:
        :return:
        '''
        # 字典的键是6位id, 值是7位id
        ID_dic = {}
        f = open(ID_file, 'r', encoding='utf-8')
        for line in f:
            # 跳过第一行
            if line[0] != 'h':
                continue
            name_list = line.strip().split()
            if not ID_dic.__contains__(name_list[0]):
                ID_dic[name_list[0]] = name_list[1]
        return ID_dic
    def get_seq_dic(self, seq_file):
        '''
            获取circBase所有人类的circRna序列对应的字典, 键为名字, 值为序列
        :return:
        '''
        seq_keys = []
        seq_values = []
        seq_dic = {}
        with open(seq_file,'r+', encoding = 'utf-8') as f:
            for line in f:
                if('>' == line[0]):
                    # 不取大于符号
                    seq_keys.append(line.split('|')[0][1:])
                else:
                    seq_values.append(line.strip())
        for i in range(len(seq_keys)):
            seq_dic[seq_keys[i]] = seq_values[i]
        return seq_dic
    def id_trans(self, need_trans_file):
        '''
        将6位hsa_circRNA_123456 按照ID转换成7位的circBaseID, 例如: hsa_circ_1234567
        :param need_trans_file: 需要转换ID的文件
        :return:
        '''
        try:
            need_trans_df = pd.read_csv(need_trans_file, header=None, keep_default_na=False)
            colum_names = ['circRNA Name', 'Gene Symbol', 'CircBase Link', 'Disease', 'circBase ID']
            need_trans_df.columns = colum_names

            # 匹配模式
            p = r'hsa(.)circRNA(.)(\d){6}'
            index = need_trans_df.index
            for i in index:
                s = need_trans_df.loc[i]['circRNA']
                res = self.regex_match(s, p)
                # 如果能匹配到6位的ID, 且他的Gene Symbol标为N/A, 则直接将ID转换为7位ID
                if (res  != '0') and (need_trans_df.loc[i]['Gene Symbol'] == 'N/A'):
                    if self.ID_trans_dic.__contains__(res):
                        need_trans_df.loc[i,'circBase ID'] = self.ID_trans_dic[res]
            # 重新写文件就炸了兄弟
            # if not os.path.exists(need_trans_file):
            #     need_trans_df.to_csv(need_trans_file, header=True, index=False)
            return need_trans_df
        except FileNotFoundError:
            print('请检查文件是否存在')
            return 0
    # 重载一个针对于DataFrame的id_trans函数
    def id_trans(self, need_trans_df, is_dataFrame):
        '''
         将6位hsa_circRNA_123456 按照ID转换成7位的circBaseID, 例如: hsa_circ_1234567
        :param need_trans_df: 是否是DataFrame
        :param is_dataFrame: 标志位，是否是针对数据框的id_trans函数
        :return:
        '''
        try:
            # 匹配模式
            p = r'hsa(.)circRNA(.)(\d){6}'
            index = need_trans_df.index
            for i in index:
                s = need_trans_df.loc[i]['circRNA Name']
                res = self.regex_match(s, p)
                # 如果能匹配到6位的ID, 且他的Gene Symbol标为NA, 则直接将ID转换为7位ID
                if (res  != '0') and (need_trans_df.loc[i]['Gene Symbol'] == 'N/A'):
                    if self.ID_trans_dic.__contains__(res):
                        need_trans_df.loc[i,'circBase ID'] = self.ID_trans_dic[res]
                else:
                    need_trans_df.loc[i, 'circBase ID'] = '0'
            return need_trans_df
        except FileNotFoundError:
            print('请检查文件是否存在')
            return 0
    def load_circRna_seq(self, has_seq_file):
        '''
        查找能直接查到的circRna的序列
        :param has_seq_file:
        :return:
        '''
        df = pd.read_csv(has_seq_file, header=0, keep_default_na=True)
        df['Sequence'] = None
        index = df.index
        for i in index:
            circRna = df.loc[i]['circRNA Name']
            circRna = self.regex_match(circRna, r'hsa(.)circ(.)(\d){7}')
            if self.seq_dic.__contains__(circRna):
                df.loc[i,'Sequence'] = self.seq_dic[circRna]
        if not os.path.exists(has_seq_file):
            df.to_csv(has_seq_file, header=True, index = False)
    def regex_match(self, s, pattern = r'hsa(.)circ(.)(\d){7}'):
        '''
            根据传入的匹配模式, 进行匹配验证
        :param pattern: 要匹配的模式
        :param s: 需要进行匹配的字串
        :return: 匹配成功返回第一个搜索到的字符串，没搜索到则返回'0'
        '''
        p = re.compile(pattern)
        res = re.search(p, s, 0)
        if res != None:
            res = res.group()
        else:
            res = '0'
        return res

    # 处理为unclear的样本
    def load_unclear_data(self, ori_ass_file, unclear_file):
        '''
        处理brief_description为Unclear的样本
        :param ori_ass_file: 原始关联文件
        :param unclear_file: 处理完后，带有circBase ID的文件
        :return: 处理完毕后, 带有circBase ID的df
        '''
        ori_data = pd.read_csv(ori_ass_file, sep=',', header=0, keep_default_na=False)
        match_rule = (ori_data['Species'] == 'Human') & (ori_data['Brief description'] == 'Unclear')
        ori_data = ori_data[match_rule]
        need_info = ['circRNA Name', 'Gene Symbol', 'CircBase Link', 'Disease Name']

        ori_data = ori_data[need_info]
        ori_data['circBase ID'] = None

        # id转换, 是数据框
        df = self.id_trans(ori_data, 1)
        df_has_link = df[df['CircBase Link'] != 'N/A']

        p = r'hsa(.)circ(.){1,4}(\d){7}'
        for i in df_has_link.index:
            ori_name = df_has_link.loc[i]['circRNA Name']
            res = self.regex_match(ori_name, p)
            # 去掉空格
            if ' ' in res:
                res = re.sub(' ', '_', res)
            # 异常字符串
            res_abnormal = re.match(r'hsa(.)circRNA(.)(\d){7}', res)
            if res_abnormal != None:
                res_abnormal = re.sub(r'circRNA', 'circ', res_abnormal.group())
                df_has_link.loc[i, 'circBase ID'] = res_abnormal
                continue
            df_has_link.loc[i]['circBase ID'] = res

        # 合并原始有circBase链接的和6位id转换后的df
        df = pd.concat([df_has_link, df]).drop_duplicates(['circRNA Name'], keep='first')
        df = df[df['circBase ID'] != '0']
        if os.path.exists(unclear_file):
            df.to_csv(unclear_file, header=True, index=False)
        return df

    # 将Unclear的样本和非Unclear的样本组合起来
    def form_final_dataset(self, ori_ass_file, has_seq_file, need_find_file, unclear_file, positve_dataset_file):
        '''
        形成最后的正样本文件，并把circRNA样本补充
        :param ori_ass_file: 原始关联文件：用来补充Unclear文件的
        :param has_seq_file: 有circBase链接的样本文件
        :param need_find_file: 需要手动check的circRna样本文件
        :param unclear_file: 保存的unclear文件
        :param positve_dataset_file: 最后形成含有cirRNA序列的样本文件
        :return:
        '''
        try:
            # 1. 数据预处理
            has_seq_df = pd.read_csv(has_seq_file, header=0, keep_default_na=False)
            need_find_df = pd.read_csv(need_find_file, header=0, keep_default_na=False)

            # 生成一个空列
            has_seq_df['circBase ID'] = None

            # 将原始circRNA id 转换到circBase ID, 通过正则匹配
            has_seq_index = has_seq_df.index
            for i in has_seq_index:
                ori_name = has_seq_df.loc[i]['circRNA Name']
                # 最少匹配1次，最多匹配4次
                res = self.regex_match(ori_name, r'hsa(.)circ(.){1,4}(\d){7}')
                # 空格替换
                if ' ' in res:
                    res = re.sub(' ', '_', res)
                # 异常字符串
                res_abnormal = re.match(r'hsa(.)circRNA(.)(\d){7}', res)
                if res_abnormal != None:
                    res_abnormal = re.sub(r'circRNA', 'circ',res_abnormal.group())
                    has_seq_df.loc[i, 'circBase ID'] = res_abnormal
                    continue
                has_seq_df.loc[i,'circBase ID'] = res

            # 直接拼接has_seq文件和need_find_df
            df = pd.concat([has_seq_df, need_find_df])

            df = df[(df['circBase ID'] != '0') & (df['circBase ID'] != '1')]
            df = df.drop_duplicates(keep='first').reset_index(drop=True)
            unclear_df = self.load_unclear_data(ori_ass_file, unclear_file)

            # 2. 合并 非unclear样本 和 unclear样本
            df = pd.concat([df, unclear_df]).drop_duplicates(keep='first').reset_index(drop=True)

            # 3. 序列爬取
            df['sequence'] = None
            df['length'] = None
            for i in df.index:
                key = df.loc[i]['circBase ID']
                if key:
                    try:
                        seq = self. seq_dic[key]
                        if len(seq) < 32767:
                            df.loc[i, 'sequence'] = seq
                            df.loc[i, 'length'] = len(seq)
                        else:
                            df.loc[i, 'sequence'] = '0'
                            df.loc[i, 'length'] = 0
                    except KeyError:
                        print("plz check the keys existed or not")
            df = df[df['sequence'] != '0']
            df = df[['circRNA Name', 'Disease Name', 'circBase ID', 'sequence', 'length']]
            if not os.path.exists(positve_dataset_file):
                df.to_csv(positve_dataset_file, header=True, index=False)
            return df
        except FileNotFoundError:
            print("plz check your file existed or not")
    def get_all_samples(self, positve_dataset_file, picked_dataset_file, all_dataset_file, not_selected_file, association_file, pick_trick):
        '''
        选择负样本
        :param positve_dataset_file: 正样本文件
        :param picked_dataset_file: 正负样本合起来的文件
        :param all_dataset_file: 正样本和所有负样本的文件
        :param not_selected_file: 未被选择的负样本文件
        :param pick_trick: 选择负样本的方式的对象: 如Negative_pick_random(seed, percent)
        :return:
        '''
        if not isinstance(pick_trick, Negative_pick_base):
            raise TypeError('pick_trick must be a object of Negative_pick_base')
        try:
            # 读进来的正样本先去重, 并且只需要取Disease Name, circBase ID, sequence 列
            df_positive = pd.read_csv(positve_dataset_file, header=0, keep_default_na=False).drop_duplicates(['Disease Name', 'circBase ID', 'sequence'], keep='first')[['Disease Name', 'circBase ID', 'sequence']]

            # 产生正样本的关联文件, 用于产生邻接矩阵, 从而计算GIP
            df_association = copy.deepcopy(df_positive)
            df_association['label'] = '1'
            df_association = df_association[['circBase ID', 'Disease Name', 'label']].reset_index(drop=True)
            if not os.path.exists(association_file):
                df_association.to_csv(association_file, header=True, index=True)

            # RNA按circBase ID去重, 只需要circBase ID, sequence列
            df_circRna_name = df_positive.drop_duplicates(['circBase ID'], keep='first')\
                            .reset_index(drop=True)[['circBase ID', 'sequence']]
            if not os.path.exists(self.circRna_name_file):
                df_circRna_name.to_csv(self.circRna_name_file, header=True, index=False)

            # 疾病按疾病名字去重后只取Diseaes Name 列
            df_disease_name = df_positive.drop_duplicates(['Disease Name'], keep='first').reset_index(drop=True)['Disease Name']
            if not os.path.exists(self.disease_name_file):
                df_disease_name.to_csv(self.disease_name_file, header=True, index=False)

            # 如果没有所有文件才生成
            if not os.path.exists(all_dataset_file):
                # 两两组合起来
                disease_name = []
                rna_id = []
                sequence = []
                label_list = []
                score_list = []
                for disease in tqdm(df_disease_name):
                    for i, row in df_circRna_name.iterrows():
                        disease_name.append(disease)
                        rna_id.append(row['circBase ID'])
                        sequence.append(row['sequence'])
                        if [disease, row['circBase ID'], row['sequence']] in df_positive.values.tolist():
                            label_list.append('1')
                            score_list.append(-1)
                        else:
                            label_list.append('0')
                            score = 0
                            for dis in df_disease_name:
                                if [dis, row['circBase ID'], row['sequence']] in df_positive.values.tolist():
                                    delta = 1
                                else:
                                    delta = 0
                                score += delta* self.df_disease_sim.loc[disease][dis]
                            score_list.append(score)
                df_all_dataset = pd.DataFrame({'Disease Name': disease_name, 'circBase ID': rna_id, 'sequence': sequence, 'label': label_list, 'score': score_list}).drop_duplicates(keep='first')
                df_all_dataset.to_csv(all_dataset_file, header=True, index=False)

            else: #否则从文件中查
                df_all_dataset = pd.read_csv(all_dataset_file, header = 0,  keep_default_na=False)
            # 负样本挑选方式的对象
            negative_picker = pick_trick
            df_picked_dataset, df_not_selected = negative_picker.pick(df_all_dataset)

            # 将选出来的正负样本写入文件
            if not os.path.exists(picked_dataset_file):
                df_picked_dataset.to_csv(picked_dataset_file, header=True, index = False)

            # 将未被选择的负样本写入文件
            if not os.path.exists(not_selected_file):
                df_not_selected.to_csv(not_selected_file, header=True, index = False)

        except FileNotFoundError:
            print('plz check your file is existed or not')
class Negative_pick_base(object):
    def __init__(self):
        pass
    def pick(self):
        pass
class Negative_pick_random(Negative_pick_base):
    '''
    挑选负样本的方式
    '''
    seed = 0
    percent = 1
    def __init__(self, seed, percent = 1):
        '''
        使用该负样本选择方式的时候, 初始化对象的时候必须传入随机种子数, 和负样本是正样本的几倍
        :param seed: 种子数
        :param percent: 比例
        '''
        self.seed = seed
        self.percent = percent
    def pick(self, df):
        return self.down_sampling(df, self.seed, self.percent)
    def down_sampling(self, df, seed, percent=1):
        '''
        给定一个dataFrame, 对其进行下采样
        :param df: 输入数据框
        :param percent: 正负样本比例
        :param seed: 随机种子
        :return:
        '''
        data0 = df[df['label'].astype(str) == '0']  # 将多数类别的样本放在data0
        data1 = df[df['label'].astype(str) == '1']  # 将少数类别的样本放在data1

        # 设置不同的随机种子, 随机获得多组负样本
        random.seed(seed)
        # 在[0, len(data0))随机取size个非重复的数
        index = random.sample(
            range(0, len(data0)), int(percent * (len(data1))))  # 随机给定下采样取出样本的序号

        lower_data0 = data0.iloc[list(index)]  # 按照随机给定的索引进行采样
        # 未被选择的负样本: 在所有样本中删去被选的负样本
        not_selected = data0[~data0.isin(lower_data0)]
        picked = (pd.concat([data1, lower_data0]))
        picked = picked.drop(columns='score',axis=1)
        not_selected = not_selected.drop(columns='score', axis=1)
        return picked, not_selected

# 负样本选择策略的一个子类, 在该类实现其pick方法即可
class Negative_pick_according_to_similarity(Negative_pick_base):
    def __init__(self, method):
        # 初始化对象的时候传入选择哪种排名的负样本
        self.method = method
    def pick(self, df_all_dataset):
        return self.pick_top_511(df_all_dataset, self.method)
    def rank(self,  df_all_data):
        '''
        计算每个负样本的得分
        :param pos_set:
        :param sample:
        :param disease_name:
        :param disease_sim:
        :return:
        '''
        # df_all_data = pd.read_csv(df_all_data, header=0, keep_default_na=False)
        df_all_pos = df_all_data[df_all_data['label'].astype(str) == '1'].drop(columns='score', axis=1)
        df_all_neg = df_all_data[df_all_data['label'].astype(str) == '0']

        for i in df_all_neg[df_all_neg['score']==-1].index:
            df_all_neg.loc[i,'score'] = 1
        df_all_neg = df_all_neg.sort_values(by=['score']).reset_index(drop=True)
        df_all_neg = df_all_neg.drop(columns='score', axis=1)
        return df_all_neg, df_all_pos

    # 挑选得分最低的前511个样本
    # 挑选包含71种疾病的511个样本
    # 挑选包含71种疾病且包含473个circRNA的前511个样本
    def pick_top_511(self, all_dataset_file, method):
        '''
        挑选与正样本数目相同的511个负样本
        method: 方式:
            1. 得分最低的511个样本: 125 x 12
            2. 包含所有疾病的511个样本: 124 x 71
            3. 包含所有疾病和所有circRNA的511个样本: 473 x 71
        :param all_dataset_file:
        :param method: 1：得分前511； 2: 包含所有疾病的得分前511; 3：包含所有疾病和circRna的前511个样本
        :return: 挑选出来的负样本已经未被选中的负样本 
        '''
        df_all_neg, df_all_pos = self.rank(all_dataset_file)

        # 构造一个负样本dic, 便于转化为DataFrame
        picked_sample_dic = {'Disease Name': [], 'circBase ID': [], 'sequence':[], 'label':[]}
        disease_name_list = df_all_neg.drop_duplicates(['Disease Name'], keep = 'first')['Disease Name'].tolist()

        if method == 1:
            for i in df_all_neg.index:
                if len(picked_sample_dic['Disease Name']) < 511:
                    picked_sample_dic['Disease Name'].append(df_all_neg.loc[i]['Disease Name'])
                    picked_sample_dic['circBase ID'].append(df_all_neg.loc[i]['circBase ID'])
                    picked_sample_dic['label'].append(df_all_neg.loc[i]['label'])
                    picked_sample_dic['sequence'].append(df_all_neg.loc[i]['sequence'])
                    df_all_neg.drop(index=i, inplace=True)

        elif method == 2:
            for i in df_all_neg.index:
                if df_all_neg.loc[i]['Disease Name'] not in picked_sample_dic['Disease Name']:
                    picked_sample_dic['Disease Name'].append(df_all_neg.loc[i]['Disease Name'])
                    picked_sample_dic['circBase ID'].append(df_all_neg.loc[i]['circBase ID'])
                    picked_sample_dic['label'].append(df_all_neg.loc[i]['label'])
                    picked_sample_dic['sequence'].append(df_all_neg.loc[i]['sequence'])
                    df_all_neg.drop(index=i, inplace=True)
            for i in df_all_neg.index:
                if len(picked_sample_dic['Disease Name']) < 511:
                    picked_sample_dic['Disease Name'].append(df_all_neg.loc[i]['Disease Name'])
                    picked_sample_dic['circBase ID'].append(df_all_neg.loc[i]['circBase ID'])
                    picked_sample_dic['label'].append(df_all_neg.loc[i]['label'])
                    picked_sample_dic['sequence'].append(df_all_neg.loc[i]['sequence'])
                    df_all_neg.drop(index=i, inplace=True)

        elif method == 3:
            for i in df_all_neg.index:
                if df_all_neg.loc[i]['Disease Name'] not in picked_sample_dic['Disease Name']:
                    picked_sample_dic['Disease Name'].append(df_all_neg.loc[i]['Disease Name'])
                    picked_sample_dic['circBase ID'].append(df_all_neg.loc[i]['circBase ID'])
                    picked_sample_dic['label'].append(df_all_neg.loc[i]['label'])
                    picked_sample_dic['sequence'].append(df_all_neg.loc[i]['sequence'])
                    df_all_neg.drop(index=i, inplace=True)

            for i in df_all_neg.index:
                if df_all_neg.loc[i]['circBase ID'] not in picked_sample_dic['circBase ID']:
                    picked_sample_dic['Disease Name'].append(df_all_neg.loc[i]['Disease Name'])
                    picked_sample_dic['circBase ID'].append(df_all_neg.loc[i]['circBase ID'])
                    picked_sample_dic['label'].append(df_all_neg.loc[i]['label'])
                    picked_sample_dic['sequence'].append(df_all_neg.loc[i]['sequence'])
                    df_all_neg.drop(index=i, inplace=True)

            for i in df_all_neg.index:
                if len(picked_sample_dic['Disease Name']) < 511:
                    picked_sample_dic['Disease Name'].append(df_all_neg.loc[i]['Disease Name'])
                    picked_sample_dic['circBase ID'].append(df_all_neg.loc[i]['circBase ID'])
                    picked_sample_dic['label'].append(df_all_neg.loc[i]['label'])
                    picked_sample_dic['sequence'].append(df_all_neg.loc[i]['sequence'])
                    df_all_neg.drop(index=i, inplace=True)

        df_picked_sample = pd.DataFrame(picked_sample_dic)
        df_not_selected = df_all_neg #inplace drop the picked negative samples, the left is the non-picked samples.
        df_picked_dataset = pd.concat([df_all_pos, df_picked_sample]).reset_index(drop=True)
        return df_picked_dataset, df_not_selected
if __name__ == '__main__':
    base_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association"

    # 改变工作目录
    if os.getcwd() != base_file:
        os.chdir(base_file)

    ori_ass_file = r"\prototype_CircR2Disease_circRNA-disease associations.csv"
    target_ass_file = r"\third_data\gene_symbol_set.csv"
    gene_symbol_file = r"\third_data\gene_symbol_needed_find.csv"
    has_seq_circRna_file =  r"\third_data\has_seq_circRna.csv"
    non_repeative_disease_file = r"\third_data\non_repetive_disease.csv"
    hg19_file = r"\human_hg19_circRNAs_putative_spliced_sequence.fa"
    ID_file = r"\third_data\ID.txt"

    # 这个是最终的正样本文件(未去重呢)
    positve_dataset_file = r"\third_data\final_positive_dataset\final_dataset.csv"
    unclear_file = r"\third_data\unclear_file.csv"

    # 关联文件
    association_file = r"\third_data\final_dataset\association_file.csv"

    # 包含所有负样本的数据集(疾病打分)
    all_dataset_with_score_file =  r"\third_data\final_dataset\all_dataset_with_score.csv"

    # 包含所有负样本的数据集, 未打分
    all_dataset_file =  r"\third_data\final_dataset\all_dataset.csv"

    samples = Samples(base_file + hg19_file, base_file + ID_file)
    # samples.load_file(base_file + ori_ass_file,
    #                   base_file + target_ass_file,
    #                   base_file + gene_symbol_file,
    #                   base_file + has_seq_circRna_file,
    #                   base_file + non_repeative_disease_file)

    # samples.form_final_dataset(base_file + ori_ass_file,
    #                            base_file + has_seq_circRna_file,
    #                            base_file + gene_symbol_file,
    #                            base_file + unclear_file,
    #                            base_file + positve_dataset_file)

    # 根据打分之后的总样本选择负样本
    seed=0
    for dataset in [all_dataset_with_score_file]:
        # 未选择的负样本的数据集
        not_selected_file = r"\third_data\final_dataset\not_selected_dataset_random{}.csv".format(seed)
        # 选择的最终的数据集
        picked_dataset_file = r"\third_data\final_dataset\picked_dataset_random{}.csv".format(seed)
        samples.get_all_samples(base_file + positve_dataset_file,
                                 base_file + picked_dataset_file,
                                 base_file + dataset,
                                 base_file + not_selected_file,
                                 base_file + association_file,
                                 Negative_pick_random(seed, 1))
