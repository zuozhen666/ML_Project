# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:54:02 2020

@author: brave
"""
import numpy as np
import pandas as pd
import random

dataset = pd.read_csv('voice.csv')
x = dataset.values[:, :-1]
y = dataset.values[:, -1]
'''
#随机划分训练集与测试集，比例7:3
train_num = random.sample(range(0, 3167), 2218) #设置随机数生成从0-3167中随机挑选2218个随机数
test_num = list(set(range(0, 3167)).difference(set(train_num)))

train_mat = np.array(x)[train_num]
train_label = np.array(y)[train_num]

test_mat = np.array(x)[test_num]
test_label = np.array(y)[test_num]
'''
#分层采样
male_train_num = random.sample(range(0, 1583), 1109)
male_test_num = list(set(range(0, 1583)).difference(set(male_train_num)))

female_train_num = random.sample(range(1584, 3167), 1109)
female_test_num = list(set(range(1584, 3167)).difference(set(female_train_num)))

train_mat = np.append(np.array(x)[male_train_num], np.array(x)[female_train_num], axis = 0)
train_label = np.append(np.array(y)[male_train_num], np.array(y)[female_train_num], axis = 0)
test_mat = np.append(np.array(x)[male_test_num], np.array(x)[female_test_num], axis = 0)
test_label = np.append(np.array(y)[male_test_num], np.array(y)[female_test_num], axis = 0)

#划分训练集里的男女
male_list = []
female_list = []

L_train = len(train_label)
for i in range(L_train):
    if train_label[i] == 'male':
        male_list.append(i)
    else:
        female_list.append(i)

#正态分布密度函数
import math

def gaussian(x, mean, std):
    return 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(x - mean) ** 2) / (2 * std * std))

#先验概率
import collections
num = collections.Counter(train_label)
num = dict(num)
male_para = num['male'] / L_train
female_para = num['female'] / L_train

#高斯分布需要的参数并构建字典
continuousPara = {}
for i in range(20):
    fea_data = train_mat[male_list, i]
    mean = fea_data.mean()
    std = fea_data.std()
    continuousPara[(i, 'male')] = (mean, std)
    fea_data = train_mat[female_list, i]
    mean = fea_data.mean()
    std = fea_data.std()
    continuousPara[(i, 'female')] = (mean, std)

#计算P(feature = x|C)
def P(feature_Index, x, C):
    fea_para = continuousPara[(feature_Index, C)]
    mean = fea_para[0]
    std = fea_para[1]
    ans = gaussian(x, mean, std)
    return ans

#测试函数
def Bayes(X):
    Result = []
    L_X = len(X)
    for i in range(L_X):
        ans_male = math.log(male_para)
        ans_female = math.log(female_para)
        L_Xi = len(X[i])
        for j in range(L_Xi):
            ans_male += math.log(P(j, X[i][j], 'male'))
            ans_female += math.log(P(j, X[i][j], 'female'))
        if ans_male > ans_female:
            Result.append('male')
        else:
            Result.append('female')
    return Result

#测试并生成混淆矩阵
from sklearn.metrics import confusion_matrix

predict_label = Bayes(test_mat)     
confusionMatrix = confusion_matrix(test_label, predict_label, labels = ['male', 'female'])

#对参数做统计
feature_mean = []   #记录各个特征在各标签下的均值
feature_std = []    #记录各个特征在各标签下的方差

for key in continuousPara.keys():
    feature_mean.append(continuousPara[(key)][0])
    feature_std.append(continuousPara[(key)][1])
   
male_mean = feature_mean[::2]
female_mean = feature_mean[1::2]

male_std = feature_std[::2]
female_std = feature_std[1::2]

#绘制通过高斯分布得到的参数
import matplotlib.pyplot as plt

names = ['0','1','2','3','4','5','6','7','8','9'
        ,'10','11','12','13','14','15','16','17','18','19']
x = range(len(names))
plt.plot(x, male_mean, marker = 'o', mec = 'r', mfc = 'w', label = 'mean under male')
plt.plot(x, female_mean, marker = 'o', mec = 'r', mfc = 'w', label = 'std under male')
plt.plot(x, male_std, marker = '*', ms = 10, label = 'mean under female')
plt.plot(x, female_std, marker = '*', ms = 10, label = 'std under female')
plt.legend()  #让图例生效
plt.xticks(x, names, rotation = 45)
plt.margins(0)
plt.subplots_adjust(bottom = 0.15)
plt.xlabel("feature_number") #X轴标签
plt.ylabel("Value") #Y轴标签
plt.title("Parameter") #标题
plt.show()

plt.show()

#绘制混淆矩阵
import itertools

def paintConfusion_float(lmr_matrix,classes):
    plt.figure(figsize = (15, 10))
    plt.imshow(lmr_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 90, size = 18)
    plt.yticks(tick_marks, classes, size = 18)
    plt.xlabel('Predict label', size = 20)
    plt.ylabel('True label', size = 20)
    lmr_matrix = lmr_matrix.astype('float') / lmr_matrix.sum(axis = 1)[:,np.newaxis]
    fmt='.6f' 
    thresh = lmr_matrix.max() / 2.
    for i, j in itertools.product(range(lmr_matrix.shape[0]), range(lmr_matrix.shape[1])):
        plt.text(j, i, format(lmr_matrix[i, j], fmt),
                     horizontalalignment = "center",
                     color = "red" if lmr_matrix[i, j] > thresh else "black", size = 22)
    plt.tight_layout()
    plt.show()

classes=['male', 'female']
paintConfusion_float(confusionMatrix, classes)
