'''
    2021/04/03
    检查circRNA与疾病是否有关联
    1. 拿到circRNA的序列
    2. 提取4-mer特征
    3. 提取疾病特征
    4. 用已经训练好的模型来验证
'''
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
class File_loader(object):
    def __init__(self, seq_file, sample_file):
        '''
        初始化对象的时候加载字典和待预测样本
        :param seq_file:
        :param sample_file:
        '''
        self.seq_dic = self.get_seq_dic(seq_file)
        self.sample_df = self.load_samples(sample_file)

    def load_samples(self,sample_file):
        '''
        导入样本文件
        :param sample_file:
        :return: df
        '''
        df = pd.read_csv("./breast_cancer_new_found.csv")[['circBase ID', 'Disease Name']]
        return df
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
    def form_file_has_sequence(self, file):
        '''
        得到序列后持久化
        :param file: 持久化文件到什么位置
        :return:
        '''
        sequence = []
        for i in self.sample_df.index:
            if(str(self.sample_df.iloc[i]['Disease Name'])=="breast cancer"):
                self.sample_df.iloc[i]['Disease Name'] = "Breast cancer"
            if(str(self.sample_df.iloc[i]['Disease Name']) == "triple-negative breast cancer"):
                self.sample_df.iloc[i]['Disease Name'] = "Triple negative breast cancer"

            sequence.append(self.seq_dic[self.sample_df.iloc[i]['circBase ID']])
        self.sample_df['sequence'] = sequence
        self.sample_df.to_csv("./independent_with_sequence")

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
def circRna_feature_extract_kmer(circRna_name_file, min, max):
    '''
    根据Rna带有序列的名字文件提取kmer特征
    :param circRna_name_file:
    :param min: k的最小取值
    :param max: k的最大值
    :return:
    '''
    df_circRna = pd.read_csv(circRna_name_file, header=0, keep_default_na=False)
    for k in tqdm(range(min, max+1)):
        path = base_file + r".\cirRna_feature_{}_mer.csv".format(k)
        for index in df_circRna.index:
            feature_list = K_mer(k).extract(df_circRna.loc[index]['sequence'])
            for i  in range(len(feature_list)):
                df_circRna.loc[index, 'fea' + str(i)] = feature_list[i]
        df_circRna.to_csv(path, header=True, index=False)
if __name__ == '__main__':
    base_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association"
    hg19_file =base_file+r"\human_hg19_circRNAs_putative_spliced_sequence.fa"
    independent_sample_file = r"./breast_cancer_new_found.csv"
    independent_sample_has_seq_file = r"./independent_with_sequence"
    file_loader = File_loader(hg19_file, independent_sample_file)
    # file_loader.form_file_has_sequence(independent_sample_has_seq_file)
    min = 1
    max = 4
    # circRna_feature_extract_kmer(independent_sample_has_seq_file,min,max)
    df_disease_sim = pd.read_csv("./disease_sim.csv")
    for i in range(min, max+1):
        df_circRNA = pd.read_csv("./cirRna_feature_{}_mer.csv".format(i))
        label = [1]*len(df_circRNA)
        merge_rna = pd.merge(df_circRNA, df_disease_sim, left_on='Disease Name', right_on='fea')
        merge_rna.drop(columns=['fea'],axis=1,inplace=True)
        merge_rna.insert(3,'label',label)
        merge_rna.to_csv(r"./dataset_disease_sim_circRNA_fea_{}_mer.csv".format(i))

