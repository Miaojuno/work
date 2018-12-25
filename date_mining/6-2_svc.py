import csv
import numpy as np
import sklearn
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


path = r'C:\Users\lenovo\Desktop\adult/adult.data'  # 数据文件路径
f=open(path,"r")
x=[]
y=[]
i=0
for row in f.readlines():
    row.replace("\"","")
    lst=row.strip().split(',')
    if i==32561:
        break
    i+=1
    lst[0]=eval(lst[0][1:])
    lst[2] = eval(lst[2])
    lst[4] = eval(lst[4])
    lst[10] = eval(lst[10])
    lst[11] = eval(lst[11])
    lst[12] = eval(lst[12])
    x.append(lst[:-1])
    if lst[-1]==" <=50K":
        y.append(0)
    else:
        y.append(1)
df=pd.DataFrame(x)
date=pd.get_dummies(df)
date=date.values.tolist()
print(date)
print(y)
date=np.array(date)
y=np.array(y)
date=date.astype(np.float32)
y=y.astype(np.float32)

train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(date,y, random_state=1, train_size=0.6,test_size=0.4)
classifier=svm.SVC(gamma="auto")

classifier.fit(train_data,train_label.ravel())

print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))