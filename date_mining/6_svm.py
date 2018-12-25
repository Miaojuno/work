import csv
import numpy as np
import sklearn
from sklearn import svm

filename = r"C:\Users\lenovo\Desktop\telco.csv"
csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)

# 求均值
k=0
for row in readcsv:
    if k==0 :
        arv = [0] * (len(row) - 1)
        count = [0] * (len(row) - 1)
        k=1
        continue
    j=0
    for item in row[:-1]:
        if item != "":
            arv[j]+=eval(item)
            count[j]+=1
            j+=1
csvHand.close()

j=0
for item in arv:
    arv[j]/=count[j]
    j+=1



csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)
x= []
y=[]
k=0
for row in readcsv:
    if k==0 :
        k=1
        continue
    j=0
    for item in row[:-1]:
        if item=="" :

            row[j]=str(arv[j])
        j+=1
    x.append(list(map(eval, row[:-1])))
    y.append(list(map(eval, row[-1])))


x=np.array(x)
y=np.array(y)
x=x.astype(np.float32)
y=y.astype(np.float32)

train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4)
classifier=svm.SVC(gamma="auto")

classifier.fit(train_data,train_label.ravel())

print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))