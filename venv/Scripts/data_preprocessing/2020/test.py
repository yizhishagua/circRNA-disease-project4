import numpy as np
import pandas as pd
if __name__ == '__main__':
    file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\association_file.csv"
    a = np.array(pd.read_csv(file))
    print(a[:,1]+a[:,2])