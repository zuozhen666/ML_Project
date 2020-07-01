# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 08:23:32 2020

@author: brave
"""
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#数据准备
def data_load(file_name):
    data_set = pd.read_csv(file_name)
    mat = data_set.iloc[:,:-1]
    label = data_set.iloc[:,-1]
    label = LabelEncoder().fit_transform(label)
    imp = SimpleImputer(missing_values = 0, strategy = 'mean')
    mat = imp.fit_transform(mat)
    train_mat, test_mat, train_label, test_label = train_test_split(mat, label, test_size = 0.3)
    return train_mat, train_label, test_mat, test_label

#KNN
def f1_KNN(train_mat, train_label, test_mat, test_label):
    best_score = 0
    for i in range(1,11):
        knn_clf = KNeighborsClassifier(n_neighbors = i)
        knn_clf.fit(train_mat, train_label)
        scores = knn_clf.score(test_mat, test_label)
        if scores > best_score:
            best_score = scores
    return best_score

#SVM    
def f2_SVM(train_mat, train_label, test_mat, test_label):
    cls = svm.LinearSVC()
    cls.fit(train_mat,train_label)
    return cls.score(test_mat, test_label)

#LogisticRegression
def f3_LogisticRegression(train_mat, train_label, test_mat, test_label):
    lr = LogisticRegression()
    lr.fit(train_mat, train_label)
    return lr.score(test_mat, test_label)

#GuassianNB
def f4_GuassianNB(train_mat, train_label, test_mat, test_label):
    gnb = GaussianNB()
    gnb.fit(train_mat, train_label)
    return gnb.score(test_mat, test_label)   
    
if __name__ == '__main__':
    times = []  #记录各算法时间消耗
    scores = [] #记录各算法准确率
    train_mat, train_label, test_mat, test_label = data_load('voice.csv')
    
    time_start = time.time()
    scores.append(f1_KNN(train_mat, train_label, test_mat, test_label))
    time_end = time.time()
    times.append(time_end - time_start)
    
    time_start = time.time()
    scores.append(f2_SVM(train_mat, train_label, test_mat, test_label))
    time_end = time.time()
    times.append(time_end - time_start)
    
    time_start = time.time()
    scores.append(f3_LogisticRegression(train_mat, train_label, test_mat, test_label))
    time_end = time.time()
    times.append(time_end - time_start)
    
    time_start = time.time()
    scores.append(f4_GuassianNB(train_mat, train_label, test_mat, test_label))
    time_end = time.time()
    times.append(time_end - time_start)
    #分别绘制时间消耗和准确率的柱状图
    name_list = ('KNN', 'SVM', 'LogisticRegression', 'GuassianNB')
    plt.bar(name_list, times, color = 'rbg')
    plt.title('time used')
    plt.xlabel('Arithmetic Name')
    plt.ylabel('time/s')
    for a, b in zip(name_list, times):
        plt.text(a, b, b, ha = 'center', va = 'bottom')
    plt.show()
    
    plt.bar(name_list, scores, color = 'rbg')
    plt.title('accuracy_rate')
    plt.xlabel('Arithmetic Name')
    plt.ylabel('num')
    for a, b in zip(name_list, scores):
        plt.text(a, b, '%.6f' % b, ha = 'center', va = 'bottom')
    plt.show()

    
    