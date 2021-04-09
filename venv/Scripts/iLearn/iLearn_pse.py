'''
    Author: Shi-Hao Li
    date: 27, Dec 2020
    des: Run ilearn-nucleotide-pse.py scripts to get RNA feature
'''
import os
import numpy as np
import pandas as pd
from tqdm import *
import sys
def run_ilearn(input_file, method, type, lamada, weight, k_mer, out):
    if method == 'PseDNC':
        command = r"python iLearn-nucleotide-pse.py --file {} --method PseDNC --type {} --lamada {} --weight {} --out {} --format csv".format(
            input_file, type, lamada, weight, out, format1
        )
    else:
        command = r"python iLearn-nucleotide-pse.py --file {} --method {} --type {} --lamada {} --kmer {} --weight {} --out {} --format csv".format(
            input_file, method, type, lamada,k_mer, weight, out
        )
    return command

if __name__ == '__main__':
    iLearn_path = r"D:\Study_Shihao_Li\circRNA-disease association prediction\iLearn"
    if os.getcwd() != iLearn_path:
        os.chdir(iLearn_path)
    rna_name_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\circRna_name.csv"
    input_file  = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\circRna_sequence.fasta"
    output_file_base = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset"
    final_feature_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\feature_file"
    for method in ['PseKNC']:
        for lam in tqdm(range(1,11)):
            for k in tqdm(['3', '4', '5', '6', '7']):
                for weight in tqdm(['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0']):
                    command = run_ilearn(input_file, method, 'RNA', lam, weight, k, output_file_base + r"\iLearn_file\circRna_fea_{}_lamada{}_k{}_weight{}.csv".format(method, lam, k, weight))
                    os.system(command)

