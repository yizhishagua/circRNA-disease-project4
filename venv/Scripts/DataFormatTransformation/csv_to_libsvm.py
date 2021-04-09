class Formator(object):
    def __init__(self, csv_file, libsvm_file, label_col_index, fea_begin_col):
        self.csv_to_sklearn(csv_file, libsvm_file, label_col_index, fea_begin_col)
    def csv_to_sklearn(self, csv_file, libsvm_file, label_col_index, fea_begin_col):
        '''
        将原始csv特征文件转化成libsvm可使用的数据格式
        :param csv_file:
        :param libsvm_file:
        :return:
        '''
        ch = ' '
        f1 = open(libsvm_file, 'w+')
        with open(csv_file, 'r') as f2:
            f2_read = csv.reader(f2)
            for row in f2_read:
                temp = []
                if f2_read.line_num > 1:
                    if str(row[label_col_index]) == '1':
                        temp.append('+1')
                    else:
                        temp.append('-1')
                    for i in range(fea_begin_col, len(row)):
                        temp.append(str(i-fea_begin_col+1)+':'+str(row[i]))
                    f1.write(ch.join(temp))
                    f1.write('\n')
            f2.close()
        f1.close()
if __name__ == '__main__':
    import csv
    path = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020'

    # for i in ['60', '2000']:
    #     for feature_strategy in [1, 2]:
    #         for seed in [0, 1, 2, 3, 4]:
    #             formator1 = Formator(path + r"\samples_{}_strategy{}_seed{}.csv".format(str(i), str(feature_strategy), str(seed)),
    #                                 path + r"\libsvm\libsvm_samples_{}_strategy{}_seed{}.txt".format(str(i), str(feature_strategy), str(seed)))
    #
    #             formator2 = Formator(path + r"\not_selected_negative\not_selected_negative_samples_{}_strategy{}_seed{}.csv".format(str(i), str(feature_strategy), str(seed)),
    #                                 path + r"\libsvm\not_selected_negative_samples_{}_strategy{}_seed{}.txt".format(str(i), str(feature_strategy),
    #                                                                                  str(seed)))
    for i in ['1','2', '3', 'random']:
        # r"./method{}_sample_picked.csv".format(i)
        # r"./libsvm_method{}_sample_picked.txt".fomat(i)
        # label_col fea_begin_col
        Formator(r"./method{}_sample_picked.csv".format(i), r"./libsvm_method{}_sample_picked.txt".format(i), 1, 4)