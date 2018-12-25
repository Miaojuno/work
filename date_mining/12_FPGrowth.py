from orangecontrib.associate.fpgrowth import *
import time

def loaddate():
    file=open(r'C:\Users\lenovo\Desktop\Transactions.csv')
    date=[]
    title=[]
    flag=0
    for line in file.readlines():
        if flag==0:
            flag=1
            title.append(line.strip().split(','))
            continue
        line=line.strip().split(',')
        flag2=0
        lined=[]
        for i,x in enumerate(line):
            if flag2==0:
                flag2=1
                continue
            else:
                if x=='1':
                    lined.append(title[0][i])
        date.append(lined)
    return date

date=loaddate()
time_start=time.time()
itemsets = dict(frequent_itemsets(date, 5000))
rules = association_rules(itemsets, 0.01)
time_end=time.time()
print('totally cost',time_end-time_start,'seconds')
rules = list(rules)
for line in rules:
    print(line)

