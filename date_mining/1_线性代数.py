import csv

from numpy.linalg import solve
from sklearn.linear_model import LinearRegression
import numpy as np
def j2(A,B):
    L=len(A[0])
    H=len(A)
    X0=[]
    X1=[]
    for i in range(L):
        X0.append(0)
        X1.append(0)
    time = 0
    while True:
        for i in range(H):
            y = 0.
            for j in range(L):
                if i != j:
                    y -= X0[j] * float(A[i][j])
            X1[i] = (float(B[i][0]) + y) / float(A[i][j])
        s = 0.
        for i in range(L):
            print(X0[1])
            s=s+pow(X1[i] - X0[i], 2)
            X0[i] = X1[i]

        time = time + 1
        if (time > 1000):
            break
        if s < pow(0.001, 2):
            break
    if (time > 1000):
        return 0
    return X1, time


filename = r"C:\Users\lenovo\Desktop\randn_data_regression_A.csv"
csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)
buffer= []
count=0
for row in readcsv:
    row = list(map(eval, row))
    buffer.append(row)
    count+=1
ll=[0.]
ll=ll*500
for i in range(len(row)-count):
    buffer.append(ll)

filename = r"C:\Users\lenovo\Desktop\randn_data_regression_b.csv"
csvHand= open(filename, "r")
readcsv= csv.reader(csvHand)
buffer2= []
for row in readcsv:
    row = list(map(eval, row))
    buffer2.append(row)
ll = [0.]
for i in range(450):
    buffer2.append(ll)
print(buffer2)


x=np.array(buffer,dtype=np.float64)
y=np.array(buffer2,dtype=np.float64)

print(x.shape)
print(y.shape)


print(j2(buffer, buffer2))
