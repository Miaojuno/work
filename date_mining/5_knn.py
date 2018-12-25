import heapq
from collections import Counter

def knn(path_train_content,path_train_tabel,path_test_content,path_test_tabel,k):
    listlabel = []
    listdistance = []
    distance=0
    loss=0
    test_tabel=open(path_test_tabel, "r", encoding='utf-8')
    train_tabel = open(path_train_tabel, "r", encoding='utf-8')
    test_content=open(path_test_content, "r",encoding='utf-8')
    train_contenr=open(path_train_content, "r",encoding='utf-8')
    test_tabel_list = []
    train_tabel_list = []
    test_content_list = []
    train_contenr_list = []
    for line in test_tabel:
        test_tabel_list.append(eval(line))
    for line in train_tabel:
        train_tabel_list.append(eval(line))
    for line in test_content:
        test_content_list.append(list(eval(line)))
    for line in train_contenr:
        train_contenr_list.append(list(eval(line)))

    for i,line_test in enumerate(test_content_list):
        for j,line_train in enumerate(train_contenr_list):
            for item in range(len(line_train)):
                distance+=abs(line_test[item]-line_train[item])
            listdistance.append(distance)
            distance=0
        lst=list(map(listdistance.index, heapq.nsmallest(k, listdistance)))
        listdistance=[]

        for l in lst:
            listlabel.append(train_tabel_list[l])
        predict=Counter(listlabel).most_common(1)[0][0]
        listlabel = []
        if predict!=test_tabel_list[i]:
            loss+=1
    print("准确率:",end="")
    print(1-loss/(i+1))
path_train_content=r"C:\Users\lenovo\Desktop\knn\cancer_X.csv"
path_train_tabel=r"C:\Users\lenovo\Desktop\knn\cancer_y.csv"
path_test_content=r"C:\Users\lenovo\Desktop\knn\cancer_X.csv"
path_test_tabel=r"C:\Users\lenovo\Desktop\knn\cancer_y.csv"
k=3
knn(path_train_content,path_train_tabel,path_test_content,path_test_tabel,k)

path_train_content=r"C:\Users\lenovo\Desktop\knn\cancer_X.csv"
path_train_tabel=r"C:\Users\lenovo\Desktop\knn\cancer_y.csv"
path_test_content=r"C:\Users\lenovo\Desktop\knn\cancer_X.csv"
path_test_tabel=r"C:\Users\lenovo\Desktop\knn\cancer_y.csv"
k=1
knn(path_train_content,path_train_tabel,path_test_content,path_test_tabel,k)

path_train_content=r"C:\Users\lenovo\Desktop\knn\amazon_X.csv"
path_train_tabel=r"C:\Users\lenovo\Desktop\knn\amazon_y.csv"
path_test_content=r"C:\Users\lenovo\Desktop\knn\amazon_X.csv"
path_test_tabel=r"C:\Users\lenovo\Desktop\knn\amazon_y.csv"
k=1
knn(path_train_content,path_train_tabel,path_test_content,path_test_tabel,k)