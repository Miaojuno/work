import csv
import numpy as np

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

data = np.array(list(csv.reader(open(r"C:\Users\lenovo\Desktop\X_data.csv","r"))))
data = data.astype(float)

label = np.array(list(csv.reader(open(r"C:\Users\lenovo\Desktop\y_label.csv","r"))))
label = label.astype(float)

t1 = np.array(list(csv.reader(open(r"C:\Users\lenovo\Desktop\Theta1.csv","r"))))
t1 = t1.astype(float)
w1=t1[:]
w1=w1[:,1:]
b1=t1[:]
b1=b1[:,1]

t2 = np.array(list(csv.reader(open(r"C:\Users\lenovo\Desktop\Theta2.csv","r"))))
t2 = t2.astype(float)
w2=t2[:]
w2=w2[:,1:]
b2=t2[:]
b2=b2[:,1]

y1=sigmoid(np.matmul(data,w1.T)+b1)
y2=sigmoid(np.matmul(y1,w2.T)+b2)


acc_count=0
for i,line in enumerate(list(y2)):
    max = 0
    max_x = -100
    for j,x in enumerate(line):
        if x>max_x:
            max=j
            max_x=x
    if max+1 == label[i]:
        acc_count+=1
print("精确率:",acc_count/(i+1))