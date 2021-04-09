from numpy import *
from pandas import *
from networkx import *
from matplotlib.pyplot import *
from Utils import *
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class MeshTreeHelper(object):
    def __init__(self):
        self.meshPath = r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\2020_mesh_tree\mtrees2020.bin'
        self.utils = Utils()
        # 全局词典
        self.meshDic, self.meshDicNodesAndNames = self.readFile()
        # 语义衰减因子
        self.decayFactor = 0.5
        # 保存图片的数目
        self.count = 0
    def readFile(self):
        '''
            读取meshTree的bin文件，返回两个字典类型的数据
            第一个字典的键是疾病名字，值是节点组成的列表
            第二个字典的键是节点，值是节点对应的疾病名字
        :return:
            meshDic: 疾病名字对应其节点列表
            meshDicNodesAndNames: 节点对应疾病名字
        '''
        meshDic = {}
        meshDicNodesAndNames = {}
        meshDic = {}
        meshDicNodesAndNames = {}
        with open(self.meshPath, 'r', encoding='utf-8') as f:
            for line in f:
                temp = line.split(";")
                meshDicNodesAndNames[temp[1][:-1]] = temp[0]
                if(meshDic.get(temp[0]) == None):
                    meshDic[temp[0]] = temp[1][:-1].split()
                else:
                    meshDic[temp[0]].append(temp[1][:-1])
        f.close()
        return meshDic, meshDicNodesAndNames
    def getAncestorList(self, nodeList):
        '''
            根据输入的节点的列表，求该列表对应的所有祖先节点, 并且按长度升序排列
            注意：包含节点本身
        :param nodeList: 多个节点组成的列表
        :return:
            ancestorNodesList: 所有的祖先节点
            ancestorList: 祖先节点中可能有的节点指的是同一种疾病，这个返回值就是所有祖先疾病的集合
        '''
        ancestorNodesList = [] # 疾病的所有祖先节点列表
        ancestorList = [] # 疾病的所有祖先名字列表
        ancestorsHelper = []
        # 将每个节点按'.'分开成列表
        for item in nodeList:
            ancestorsHelper.append(item.split('.'))
        # 将拆分的列表拼接成为有用的节点
        for item in ancestorsHelper:
            t = ''
            for i in range(len(item)):
                if (t!=''):
                    t = t + '.' + item[i]
                else: #第一个节点不需要加'.'
                    t += item[i]
                # 去除注释则不包含节点本身
                # if(t in nodeList):
                #     break
                ancestorNodesList.append(t)
                ancestorList.append(self.meshDicNodesAndNames[t])
        ancestorNodesList = self.utils.bubbleSort(list(set(ancestorNodesList)))
        ancestorList = list(set(ancestorList))
        return ancestorNodesList, ancestorList
    def getEdgeList(self, ancestorNodeList):
        '''
            根据包含本身节点的祖先节点List, 求所有的边
        :param ansestorList:
        :return:
        '''
        edgeList = []
        for item1 in ancestorNodeList:
            for item2 in ancestorNodeList:
                if(item1 in item2): #这个逻辑就保证了不会出现反向边，一定由节点小的指向节点大的
                    if( len(item2) - len(item1) == 4):
                        edgeList.append((self.meshDicNodesAndNames[item1], self.meshDicNodesAndNames[item2]))
        edgeList = list(set(edgeList))
        return edgeList
    def getDiGraph(self, diseaseName = 'Lupus Nephritis', diseaseNodesDic ={'Lupus Nephritis': ['C12.777.419.570.363.680', 'C13.351.968.419.570.363.680', 'C17.300.480.680',
                             'C20.111.590.560']}):
        '''
            获取每个疾病节点构成的有向图DAG
        :param : diseaseNames: 需要构图的节点的名字
                 diseaseNodesDic: 疾病名字为key，其节点列表为value的字典
        :return: 该疾病的有向无环图
        '''
        # 新建一个图对象
        g = nx.DiGraph(graphName = diseaseName)

        # 获得每个疾病的祖先节点列表, 以及祖先疾病包含疾病本身
        ancestorNodesList, ancestorList = self.getAncestorList(diseaseNodesDic[diseaseName])

        # 给每个节点加一个节点名字的属性方便操作, 并且将节点加入到图中去
        for item in ancestorList:
            g.add_node(item, nodeName = item)

        # 得到边, 无重复, 考虑了一种病有多个节点的情况
        edgesList = self.getEdgeList(ancestorNodesList)

        # 给图加入边
        g.add_edges_from(edgesList)

        # self.graphSave(g,diseaseName)
        return g
    def getDiGraph1(self, diseaseName):
        '''
        重载以下,便于参数的输入
        :param diseaseName:
        :return:
        '''
        g = self.getDiGraph(diseaseName, {diseaseName: self.meshDic[diseaseName]})
        return g
    def graphShow(self, G):
        nx.draw(G, with_labels=True, pos=nx.planar_layout(G))
        plt.show()
        plt.close()
    def getSemanticContribution(self, DAG):
        '''
            给输入图中的每个节点加入属性: 语义贡献值
            并且给计算DAG的语义值
        :param G:
        :return: G 加了语义贡献值后的图
        '''
        # 叶子节点的语义贡献值为1
        diseaseName = DAG.graph['graphName']
        DAG.nodes[diseaseName]['contribution'] = 1

        # 将diseaseName节点的祖先以列表的形式返回
        ancestorsOfDAG = list(nx.ancestors(DAG, diseaseName))

        semanticValue = 0 # DAG图的语义值
        # 如果祖先存在
        if (ancestorsOfDAG != None) and (len(ancestorsOfDAG) != 0):
            for node in ancestorsOfDAG:
                # 按最短路径来更新贡献值
                path = nx.shortest_path(DAG, node, diseaseName)
                # 路径越短，计算得来的贡献值就越大，符合逻辑
                DAG.nodes[node]['contribution'] = pow(self.decayFactor, (len(path) - 1))
                semanticValue += DAG.nodes[node]['contribution']

        semanticValue += DAG.nodes[diseaseName]['contribution']
        DAG.graph['semanticValue'] = semanticValue
        return DAG
    def get_semantic_contribution2(self, DAGs, DAG):
        '''
        第二种计算DAG图中每个节点的语义贡献值的方法
        n(disease(s))/n(diseases): 包含疾病s的DAG的总数目/疾病的总数目
        :param DAGs: 所有DAG组成的一个列表
        :param DAG: 要计算每个节点贡献值的DAG
        :return: 每个节点带有贡献值, 且图带有语义值的一个DAG对象
        '''
        # 总的疾病数目
        disease_num = 0
        disease_list = []
        for g in DAGs:
            disease_list += g.nodes
        disease_num = len(set(disease_list))
        if disease_num == 0:
            raise ValueError('疾病总数为0, 请检查输入')
        # DAG的所有节点
        nodes = DAG.nodes
        # DAG图的语义值
        semantic_value = 0

        for node in nodes:
            count = self.count_DAG_numer_contains_disease(DAGs, node)
            DAG.nodes[node]['contribution'] = float(-log(count/disease_num))
            semantic_value += DAG.nodes[node]['contribution']
        DAG.graph['semanticValue'] = semantic_value
        return DAG
    def count_DAG_numer_contains_disease(self, DAGs, s):
        '''
        计算包含疾病s的所有DAG的数目
        :param DAGs: DAG组成的一个列表
        :param s: 要查找的某个疾病
        :return:
        '''
        count = [ 1 for DAG in DAGs if s in DAG.nodes]
        return len(count)
    def getSemanticSimilarity(self, DAG1, DAG2):
        # 节点的信息[(节点名,{ k1:val1, k2:val2 ... })]
        nodesInfo1 = list(DAG1.nodes(data=True))
        nodesInfo2 = list(DAG2.nodes(data=True))

        # 两个图公共节点的贡献值之和
        comContriSum = 0
        similarity = 0

        for nodeInfo1  in nodesInfo1:
            for nodeInfo2 in nodesInfo2:
                # 如果节点名字相同
                if (nodeInfo1[0] == nodeInfo2[0]):
                    # 加算公共贡献值
                    comContriSum += (nodeInfo1[1]['contribution'] + nodeInfo2[1]['contribution'])

        # 相似性为公共节点的语义值之和除以两个图的语义值之和
        similarity = comContriSum / (DAG1.graph['semanticValue'] + DAG2.graph['semanticValue'])
        return similarity
    def graphSave(self, G, diseaseName):
         # 设定一个图片保存数目的计数器
        self.count += 1
        tempName = diseaseName
        nx.draw(G, with_labels=True, pos=nx.planar_layout(G))
        # plt.savefig(r'D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\DAG\{}.png'.format(tempName))
        # plt.show()
        plt.close()
        print(G.graph['graphName'], 'DAG已保存', self.count)
        return 1
    def process(self):

       g1 = self.getDiGraph1('Smith-Magenis Syndrome')
       g2 = self.getDiGraph1('Jet Lag Syndrome')
       # g1 = self.get_semantic_contribution2([g1, g2], g1)
       # g2 = self.get_semantic_contribution2([g1, g2], g2)
       g1 = self.getSemanticContribution(g1)
       g2 = self.getSemanticContribution(g2)
       sim = self.getSemanticSimilarity(g1, g2)
       # print("intersection is: {}" .format([x for x in g1.nodes if x in g2.nodes]))
       # print(g1.nodes(data=True))
       print(g1.graph)
       print(g2.graph)
       print(sim)
if __name__ == '__main__':
    MeshTreeHelper().process()