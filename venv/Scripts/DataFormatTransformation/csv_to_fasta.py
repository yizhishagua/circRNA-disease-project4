'''
    author: Shi-Hao Li
    date: 26, Dec. 2020
    des: this script is for transforming the sequence file csv formatted to fasta formatted file
'''
import os
import pandas as pd
import numpy as np
def transform(csv_file, fasta_file):
    try:
        df_csv = pd.read_csv(csv_file, header=0, keep_default_na=False)
        print(df_csv)
        sequence = []
        for index in df_csv.index:
            sequence.append(( df_csv.loc[index]['circBase ID'], df_csv.loc[index]['sequence']))
        print(sequence)
        if os.path.exists(fasta_file):
            return
        with open(fasta_file, 'w+', encoding='utf-8') as f:
            for item in sequence:
                f.write('>' + item[0] + '\n')
                if item == sequence[-1]:
                    f.write(item[1])
                else:
                    f.write(item[1] + '\n')
            f.close()
    except FileNotFoundError:
        print('plz check your file existed or not')
if __name__ == '__main__':
    csv_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\circRna_name.csv"
    fasta_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\circRna_seuqence.fasta"
    transform(csv_file, fasta_file)