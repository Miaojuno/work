import csv
import numpy as np
import pandas as pd

def get_pred(dataSet, inputSimple):
    p0classData = []#初始化类别矩阵
    p1classData = []
    classLabels = dataSet[dataSet.columns[-1]]#选取类别列

    for i in range(len(dataSet.columns) - 1):
        columnLabels = dataSet[dataSet.columns[i]]#特征列
        pData = pd.concat([columnLabels, classLabels], axis = 1)#拼接特征列和类别列
        classSet = list(set(classLabels))
        for pclass in classSet:
            filterClass = pData[pData[pData.columns[-1]] == pclass]#根据类别划分数据集
            filterClass = filterClass[pData.columns[-2]]
            # if isinstance(inputSimple[i], float):#判断是否是连续变量
            if i==1 or i==4 or i==13:  # 判断是否是连续变量
                classVar = np.var(filterClass)#方差
                classMean = np.mean(filterClass)#均值
                pro_l = 1/(np.sqrt(2*np.pi) * np.sqrt(classVar))
                pro_r = np.exp(-(inputSimple[i] - classMean)**2/(2 * classVar))
                pro = pro_l * pro_r#概率
                if pclass == 1:
                    p0classData.append(pro)
                else:
                    p1classData.append(pro)
            else:
                classNum = np.count_nonzero(filterClass == inputSimple[i])#计算属于样本特征的数量
                pro = (classNum + 1)/(len(filterClass) + len(set(filterClass)))#此处进行了拉普拉斯修正
                if pclass == 1:
                    p0classData.append(pro)
                else:
                    p1classData.append(pro)
    return p0classData, p1classData



filename = r"C:\Users\lenovo\Desktop\german_clean.csv"
dataSet=pd.read_csv(filename)
train_data = np.array(dataSet)#np.ndarray()
ds=train_data.tolist()#list
acc=0
for i in range(10):
    p0classData, p1classData = get_pred(dataSet, ds[i])
    if np.prod(p0classData) > np.prod(p1classData):  # 计算条件概率的累积
        if ds[i][-1]==1:
            acc+=1
    else:
        if ds[i][-1]==2:
            acc+=1
print(acc/1000)
