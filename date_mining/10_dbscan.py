import csv
import numpy as np
import math

row_data=np.array(list(csv.reader(open(r"C:\Users\lenovo\Desktop\dbscan_test.csv","r"))))
row_data = row_data.astype(int)
row_data=row_data.tolist()

def dist (a , b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))

cent=[]
for data1 in row_data:
    count=0
    for data2 in row_data:
        if dist(data1,data2)<=2.01:
            count+=1
    if count>4:
        cent.append(data1)

norm=[]
for data1 in row_data:
    for data2 in cent:
        if dist(data1,data2)<2.01 and data1 not in cent:
            norm.append(data1)
            break

other=[]
for data in row_data:
    if data not in cent and data not in norm:
        other.append(data)

print(cent)
print(norm)
print(other)