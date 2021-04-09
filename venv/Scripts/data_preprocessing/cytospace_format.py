import pandas as pd
import numpy as np
def cytospace_format(file, t_file):
   df = pd.read_csv(file)
   circRNA = df['circBase ID'].tolist()
   disease = df['Disease Name'].tolist()
   label = [1]*len(circRNA)
   print(label)
   df1 = pd.DataFrame({'circRNA':circRNA, 'label':label, 'disease':disease})
   df1.to_csv(t_file)
if __name__ == '__main__':
    file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_positive_dataset\final_dataset.csv"
    t_file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_positive_dataset\cytospace_file.csv"
    cytospace_format(file, t_file)