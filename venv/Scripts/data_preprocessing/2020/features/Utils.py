from numpy import *
from pandas import *
import numpy as np
import pandas as pd
class Utils(object):
    def __init__(self):
        pass
    def countAndSet(self, oriData):
        return len(set(oriData))
    def arrSetAndList(self, oriData):
        return np.array(list(set(oriData)))
    def listAndSet(self, list_):
        '''
        对输入为字符串或者基本数据类型为元素的列表去重
        :param list:
        :return: 去重后的list
        '''
        return list(set(list_))
    def readCsv(self, source):
        '''
            # 使用numpy+pandas读文件，返回一个array
        :param source: 源文件位置
        :return: array
        '''
        return  np.array(pd.read_csv(source, sep = ',', header=0, keep_default_na=False))
    def readCsvDf(self, source):
        '''
            # 使用numpy+pandas读文件，返回一个dataFrame
        :param source: 源文件位置
        :return: array
        '''
        return pd.read_csv(source, sep = ',', header=0, keep_default_na=False)
    def writeCsv(self, target, dic = {}):
        '''
            #用pandas的DataFrame写入csv
        :param target: 目标文件
        :param keyValue: 字典格式的数据
        :return: void
        '''
        if(len(dic) == 0):
            print("传入键值对为0")
            return 0
        df = DataFrame(dic, columns= dic.keys())
        df.to_csv(target, index=True, header=True)
        return "写入成功"
    def getSequence(self, source):
        '''
            读取txt文件，读取方式为r
        :param source: 源文件
        :return:
        '''
        temp = {}
        key = ""
        f = open(source, 'r', encoding='utf-8')
        for line in f:
            if '>' in line:
                key = line.split('|')[0]
                continue
            temp[key] = line
        f.close()
        return temp
    def arrToDic(self, arr, indexNum = 0):
        '''
            将数组的第indexNum列作为键，后面的作为值
        :param arr:原始数组
        :param indexNum: 指定要作为键的列号
        :return: a dictionary
        '''
        d = {}
        for i in range(len(arr)):
            d[arr[i,indexNum]] = list(arr[i][indexNum + 1:])
        return d
    def dicToDf(self, ori_dic = {}, param = []):
        '''
        将一个字典转化为dataFrame
        :param oriDic: 输入的字典
        :param param: 键组成的列表
        :return:
        '''
        temp = {}
        if(len(ori_dic) != 0):
            temp[param[0]] = ori_dic.keys()
            temp[param[1]] = ori_dic.values()
        return temp
    def bubbleSort(self, list = [[]]):
        '''
            针对传入的二维数组，且每个元素是string类型，按照每个元素的长度进行排序
        :param list:
        :return:
        '''
        i = 0
        flag = 1
        while( (i<len(list)) and (flag == 1)):
            flag = 0
            j = len(list) - 2
            while( j >= i):
                list[j] = str(list[j])
                list[j + 1] = str(list[j+1])
                if(len(list[j]) > len(list[j+1])):
                    temp = list[j]
                    list[j] = list[j+1]
                    list[j+1] = temp
                    flag = 1
                j -= 1
            i += 1
        return list