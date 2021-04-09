import pandas as pd
import numpy as np

if __name__ == '__main__':
    df1 = pd.DataFrame({'d1':[1, 0.5, 0.3], 'd2':[0,1,0]})
    df2 = pd.DataFrame({'d2':[1, 0.5, 0.3], 'd2':[0,1,0], 'd3':[0.5, 0.6, 1]})

    print(df1+df2)