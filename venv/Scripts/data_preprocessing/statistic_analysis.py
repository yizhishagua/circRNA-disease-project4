import pandas as pd
from pandas import DataFrame as df
def cal_num_disease_related_circRNA(file, res_file):
        '''
        统计关联文件中每个疾病关联的circRNA的数目
        :param file:关联文件adj_matrix
        :param res_file: 统计结果文件
        :return:
        '''
        df = pd.read_csv(file)
        res = {}
        for field in df.columns[2:]:
            ser = df[field]
            res[field] = ser.value_counts()[1]
        res = sorted(res.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
        with open(res_file,'w+') as f:
            for entry in res:
                f.write(entry[0]+'\t'+str(entry[1]))
                f.write("\n")
            f.close()
        return res
def test(file):
    df = pd.read_csv(file)
    return df
if __name__ == '__main__':
    adj_matrix_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\adj_matrix.csv"
    res_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\disease_related_circRNA_number.txt"

    file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\picked_dataset_after_rank_method2.csv"
    df = test(file)
    match_rule = df['label']==0
    df_disease = df[match_rule]['Disease Name']
    df_rna = df[match_rule]['circBase ID']

    df_disease = df_disease.drop_duplicates(keep='first')
    df_rna = df_rna.drop_duplicates(keep='first')
    print(df_disease)
    print(df_rna)