'''
    author: Shi-Hao Li
    date: 16 Dec. 2020
    description: this script is writen for extract the positive and random negative samples from emb_data that is
                 a feature file extracted with GAE algorithm.
'''
class Dataset(object):
    base_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020"

    #GAE训练60次生成的特征文件
    emb_data_60 = base_file + r"\emb_data\emb\GAE_node_emb_ep60.emd"

    #2000次
    emb_data_2000 = base_file + r"\emb_data\emb\GAE_node_emb_ep2000.emd"


    #正样本的名字文件
    positive_dataset = base_file + r'\Prototype_CircRNA_Disease_Association\third_data\final_positive_dataset\final_dataset.csv'

    def __init__(self):
        self.utils = Utils()

    def load_emb_data(self, emb_file):
        '''
            加载embedding向量
        '''
        node_fea_dic = {}
        f = open(emb_file, 'r', encoding='utf-8')
        for line in f:
            if ':' in line:
                node_list = line.strip().split(':')
                if len(node_list) >2:
                    node_fea_dic[node_list[0] + ':' + node_list[1]] = node_list[2].split()
                node_fea_dic[node_list[0]] = node_list[1].split()
        return node_fea_dic

    def load_positive_dataset(self):
        '''
            加载正样本
        '''
        ps = self.utils.readCsv(self.positive_dataset)
        return (ps[:,1:3])

    def get_datasets(self, target_file, not_selected_file,emb_file, feature_strategy, seed, percent):
        '''
            随机选择负样本：
            1. 组建所有的负样本
            2. 随机下采样
            3. 确定负样本的数目与正样本相同651个样本
        :param target_file: 特征文件的输出位置
        :param not_selected_file: 未被选择的负样本的文件输出位置
        :param emb_file: 嵌入特征文件所在位置
        :param feature_strategy: 特征组合方式：1. 简单融合, 2. 对应位置相加
        :param seed: 生成负样本的随机种子
        :param percent: 正负样本比例, 1 表示 正: 负 == 1:1; 2表示 正:负 = 1:2
        :return:
        '''
        if feature_strategy not in [1, 2, 3, 4]:
            print("请选择合理的特征组合方式")
            return

        # 加载所有正样本
        ps = self.load_positive_dataset()

        # 所有的负样本
        all_negative_samples = []

        # 正样本
        positive_samples = []

        # circRNA与疾病的分隔符
        ch = "=||="

        # circRna_name, disease_name, features; 正样本去重前659种, 去重后651种
        positive_samples = self.utils.listAndSet([str(item[0])+ch+str(item[1]) for item in ps])

        label1 = [1 for i in range(len(positive_samples))]

        # 用hashmap去重, 保证去重后还有顺序, 还能顺便统计各个元素出现的次数
        # 604种circRNA
        circRna_name0 = ps[:, 0]
        circRna_name_dic = self.remove_duplicates_by_dic(circRna_name0)
        circRna_name = circRna_name_dic.keys()

        # 88种disease
        disease_name0 = ps[:,1]
        disease_name_dic = self.remove_duplicates_by_dic(disease_name0)
        disease_name = disease_name_dic.keys()

        # 包含正样本和所有负样本以及标签的字典
        all_samples = {}

        count = 0
        in_list = []
        # circRNA和疾病两两组合，去掉正样本的部分
        for rna in circRna_name:
            for disease in disease_name:
                sample = str(rna) + ch + str(disease)
                if sample not in positive_samples:
                    all_negative_samples.append(sample)
        label0 = [0 for i in range(len(all_negative_samples))]

        samples = positive_samples + all_negative_samples

        # 标签不要整反了 毛哥
        label = label1 + label0

        all_samples['sample_name'] = samples
        all_samples['label'] = label

        # 加入特征步骤
        node_fea_dic = self.load_emb_data(emb_file)
        fea_list = []
        count = 0

        for item in samples:
            ele = item.split(ch)

            # 加入特征策略
            fea_list.append(self.feature_method(node_fea_dic[ele[0]], node_fea_dic[ele[1]], feature_strategy))
        fea_list = np.array(fea_list)

        if len(fea_list) > 0:
            for i in range(len(fea_list[0])):
                all_samples['fea' + str(i)] = list(fea_list[:, i])
                
        # (对正样本+所有负样本)的df, 随机种子, 正负样本比例
        df, not_selected  = self.down_sampling(pd.DataFrame(all_samples), seed, percent)

        # 将每个样本的特征加入到df中并写入文件
        df.to_csv(self.base_file+target_file, header=True,index=0)
        not_selected.to_csv(self.base_file + not_selected_file, header = True, index = 0)
        return circRna_name, disease_name

    def feature_method(self, fea_list1, fea_list2, strategy_num):
        '''
        对两种特征的融合方式
        :param fea_list1: 第一种节点的特征
        :param fea_list2: 第二种节点的特征
        :param strategy_num: 融合方式: strategy_num == 1, 直接融合; strategy_num == 2,对应索引相加
        :return: 融合后的特征
        '''
        # 列表中的元素若是字符串, 直接加和只是对应位置拼接
        if isinstance(fea_list1, list) and isinstance(fea_list2, list):
            if strategy_num == 1:
                # 列表直接相加, 返回的是列表元素拼接后的列表
                return fea_list1 + fea_list2
            elif strategy_num == 2:
                # 列表中的元素若是字符串, 直接加和只是对应位置拼接
                return [ str(float(fea_list1[i]) + float(fea_list2[i])) for i in range(len(fea_list1))]
                # 第三种策略是交换疾病和rna的顺序, 是否效果还是依然这么好
            elif strategy_num == 3:
                return fea_list2 + fea_list1
        else:
            print("输入的不是列表")
            return 0
    def remove_duplicates_by_dic(self, target_list):
        '''
        使用字典对target_list的元素去重, 并且计算每个元素出现的次数
        :param target_list:
        :return: 该字典
        '''
        dic = {}
        for item in target_list:
            # 如果包含这个键, 则计数
            if dic.__contains__(item):
                dic[item] += 1
            else: #否则生成这个键, 将其值设置为1
                dic[item] = 1
        return dic
    
    def down_sampling(self, df, seed, percent=1):
        '''
        给定一个dataFrame, 对其进行下采样
        :param df: 输入数据框
        :param percent: 正负样本比例
        :param seed: 随机种子
        :return:
        '''
        data0 = df[df['label'] == 0]  # 将多数类别的样本放在data0
        data1 = df[df['label'] == 1]  # 将少数类别的样本放在data1

        # 设置不同的随机种子, 随机获得多组负样本
        random.seed(seed)

        # 在[0, len(data0))随机取size个非重复的数
        index = random.sample(
            range(0, len(data0)), int(percent * (len(data1))))  # 随机给定下采样取出样本的序号
        lower_data0 = data0.iloc[list(index)]  # 按照随机给定的索引进行采样

        # 未被选择的负样本: 在所有样本中删去被选的负样本
        keys = df.keys()
        not_select = data0[~data0[keys[0]].isin(lower_data0[keys[0]].values)]

        return (pd.concat([data1, lower_data0])), not_select
    def process(self):
        pass
if __name__=="__main__":
    from Utils import *
    from pandas import *
    from numpy import *
    import pandas as pd
    import numpy as np
    import random
    dataset = Dataset()
    percent = 1
    # 3种特征融合方式, 五个随机种子
    df = dataset.load_positive_dataset()
    print(df)
    # for feature_strategy in [3]:
    #     for seed in [0, 1, 2, 3, 4]:
    #         # 60
    #         dataset.get_datasets(r"/samples_60_strategy{}_seed{}.csv".format(str(feature_strategy), str(seed)),
    #                              r"/not_selected_negative/not_selected_negative_samples_60_strategy{}_seed{}.csv".format(str(feature_strategy), str(seed)),
    #                              dataset.emb_data_60, feature_strategy, seed, percent)
    #         print(r"samples_60_strategy{}_seed{}.csv completed".format(str(feature_strategy), str(seed)))
    #
    #         # 2000
    #         dataset.get_datasets(r"/samples_2000_strategy{}_seed{}.csv".format(str(feature_strategy), str(seed)),
    #                              r"/not_selected_negative/not_selected_negative_samples_2000_strategy{}_seed{}.csv".format(str(feature_strategy), str(seed)),
    #                              dataset.emb_data_2000, feature_strategy, seed, percent)
    #         print(r"samples_2000_strategy{}_seed{}.csv completed".format(str(feature_strategy), str(seed)))
