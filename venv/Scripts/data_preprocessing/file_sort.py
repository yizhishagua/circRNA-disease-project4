'''
    sort the certain file according to the index
'''
import pandas as pd
import numpy as np
def sort_file(file_name, index_col):
    df = pd.read_csv(file_name,index_col='label')
    df.sort_index(axis = 0, inplace = True)
    df.sort_index(axis = 1, inplace = True)
    df.to_csv(file_name);
    return df;
if __name__ == '__main__':
    file_name = r"./adj_matrix.csv"
    # ,r"./adj_matrix.csv"]
    # for file in file_name:
    print(sort_file(file_name, 1))