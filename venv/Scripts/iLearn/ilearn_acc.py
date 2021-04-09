'''
    Author: Shi-Hao Li
    date: 26, Dec 2020
    des: Run ilearn-nucleotide-acc.py scripts to get RNA feature
'''
import os
def run_ilearn(input_file, method, type, lag, out, format1):
    command = r"python iLearn-nucleotide-acc.py --file {} --method {} --type {} --lag {} --out {} --format {}".format(
        input_file,  method, type, lag, out, format1
    )
    return command
if __name__ == '__main__':
    iLearn_path = r"D:\Study_Shihao_Li\circRNA-disease association prediction\iLearn"
    if os.getcwd() != iLearn_path:
        os.chdir(iLearn_path)
    print(os.getcwd())
    input_file  = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\circRna_sequence.fasta"
    output_file_base = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\iLearn_file"
    for method in ['DAC', 'DCC', 'DACC']:
        for lag in ['3', '4', '5', '6', '7']:
            try:
                command = run_ilearn(input_file, method, 'RNA', lag, output_file_base + r"\circRna_fea_{}_{}.csv".format(method, lag),
                                     'csv')
                os.system(command)
            except:
                raise ValueError('param error')