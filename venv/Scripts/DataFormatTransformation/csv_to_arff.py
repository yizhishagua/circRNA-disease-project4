'''
date: 17. Dec. 2020
des: csv特征文件转化为arff文件
'''
import os
import re
class CSV_To_Arff(object):
    def __init__(self):
        print("fuck you")
    def trasform(self, csv_file, arff_file, label_index, fea_start_index):
        '''
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
    import csv
    transformer = CSV_To_Arff()
    base_file = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\feature_file'
    # for i in ['60', '2000']:
    #     for feature_strategy in [3]:
    #         for seed in [0, 1, 2, 3, 4]:
    #             csv_file = base_file + r"\samples_{}_strategy{}_seed{}.csv".format(str(i), str(feature_strategy), str(seed))
    #             arff_file = base_file + r"\weka\samples_{}_strategy{}_seed{}.arff".format(str(i), str(feature_strategy), str(seed))
    #             transformer.trasform(csv_file, arff_file)
    # for k in [1, 2, 3, 4, 5, 6]:
    #     csv_file = base_file + "\dataset_mesh_deep_walk_feature_{}_mer.csv".format(str(k))
    #     arff_file = base_file + r"\arff_file\dataset_mesh_deep_walk_feature_{}_mer.arff".format(str(k))
    #     transformer.trasform(csv_file, arff_file, 3, 5)
    csv_file = './whole_network_method1_sample_picked.csv'
    arrff_file = './whole_network_method1_sample_picked.arff'

    # label_col, fea_begin_col
    transformer.trasform(csv_file, arrff_file, 1, 4)