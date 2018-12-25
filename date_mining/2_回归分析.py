
from sklearn.linear_model import LinearRegression
import csv
import numpy as np




filename = r"C:\Users\lenovo\Desktop\randn_data_regression_A.csv"
csvHand= open(filename,"r")
readcsv= csv.reader(csvHand)
buffer= []
for row in readcsv:
    row = list(map(eval, row))
    buffer.append(row)
filename = r"C:\Users\lenovo\Desktop\randn_data_regression_b.csv"
csvHand= open(filename, "r")
readcsv= csv.reader(csvHand)
buffer2= []
for row in readcsv:
    row = list(map(eval, row))
    buffer2.append(row)


def bgd(x, y):
    row, col = x.shape
    w = np.ones(col)
    loss = 10
    count = 0
    step_size = 0.01
    max = 1000
    while loss > 0.005 and count < max:
        loss = 0
        error = np.zeros(col)
        for i in range(row):
            predict = np.dot(w.T, x[i])
            for j in range(col):
                error[j] += (y[i] - predict) * x[i][j]
        for j in range(col):
            #w[j] += step_size * error[j]/ row
            w[j] = w[j] *(1- 20* step_size/row) + step_size * error[j]/ row
        for i in range(row):
            predict = np.dot(w.T, x[i])
            error = abs(predict - y[i])
            error = (1 / (row * col)) * np.power((predict - y[i]), 2)
            loss += error
        print(count,loss)
        count += 1
    return w,count


x=np.array(buffer)
y=np.array(buffer2)
w = bgd(x, y)
print(w)

# l=LinearRegression()
# regr = l.fit(buffer, buffer2)
# print(l.coef_)
# print(l.intercept_)
# print(l.score(buffer,buffer2))
# print(np.mean((l.predict(buffer)-buffer2)**2))
