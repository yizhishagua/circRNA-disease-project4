import pandas as pd
from pandas import DataFrame as df
def cal_diseases_in_method1(file):
    df = pd.read_csv(file)
    df = df[df['label']==0]['Disease Name']
    df.drop_duplicates(inplace=True)
    df.sort_values(ascending=True,inplace=True)
    print(df)
if __name__ == '__main__':
    file = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\picked_dataset_after_rank_method1.csv'
    cal_diseases_in_method1(file)