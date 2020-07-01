# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:58:30 2020

@author: brave
"""
import numpy as np
import pandas as pd
import random
import collections
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import time

#加载数据集
def data_load(file_name):
    data_set = pd.read_csv(file_name)
    x = data_set.values[:, :-1]
    y = data_set.values[:, -1]
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

    return train_mat, train_label, test_mat, test_label

#求高斯分布需要的参数并构建字典
def get_para(train_mat, train_label):
    male_list = []  #男声的序号
    female_list = []    #女声的序号

    L_train = len(train_label)
    for i in range(L_train):
        if train_label[i] == 'male':
            male_list.append(i)
        else:
            female_list.append(i)
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
    return continuousPara

#绘制高斯分布需要的参数图
def show_para(continuousPara):
    feature_mean = []   #记录各个特征在各标签下的均值
    feature_std = []    #记录各个特征在各标签下的方差

    for key in continuousPara.keys():
        feature_mean.append(continuousPara[(key)][0])
        feature_std.append(continuousPara[(key)][1])
   
    male_mean = feature_mean[::2]
    female_mean = feature_mean[1::2]

    male_std = feature_std[::2]
    female_std = feature_std[1::2]
    
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

#正态分布函数
def gaussian(x, mean, std):
    return 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(x - mean) ** 2) / (2 * std * std))

#计算P(feature = x|C)
def P(feature_Index, x, C, continuousPara):
    fea_para = continuousPara[(feature_Index, C)]
    mean = fea_para[0]
    std = fea_para[1]
    ans = gaussian(x, mean, std)
    return ans

#高斯贝叶斯过程
def Bayes(X, train_label, continuousPara):
    #求先验概率
    L_train = len(train_label)
    num = collections.Counter(train_label)
    num = dict(num)
    male_para = num['male'] / L_train
    female_para = num['female'] / L_train
    
    Result = []
    L_X = len(X)
    for i in range(L_X):
        ans_male = math.log(male_para)
        ans_female = math.log(female_para)
        L_Xi = len(X[i])
        for j in range(L_Xi):
            '''
            if j == 7|6|19:
                continue
            '''
            ans_male += math.log(P(j, X[i][j], 'male', continuousPara))
            ans_female += math.log(P(j, X[i][j], 'female', continuousPara))
            '''
            if j == 12:
                ans_male += math.log(P(j, X[i][j], 'male', continuousPara))
                ans_female += math.log(P(j, X[i][j], 'female', continuousPara)) 
                ans_male += math.log(P(j, X[i][j], 'male', continuousPara))
                ans_female += math.log(P(j, X[i][j], 'female', continuousPara)) 
            '''   
        if ans_male > ans_female:
            Result.append('male')
        else:
            Result.append('female')
    return Result

#绘制混淆矩阵
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

#主模块
def main_():
    train_mat, train_label, test_mat, test_label = data_load('voice.csv')  #加载数据集
    continuousPara = get_para(train_mat, train_label)   #求高斯分布需要的参数并构建字典
    #show_para(continuousPara)   #绘制高斯分布需要的参数图
    predict_label = Bayes(test_mat, train_label, continuousPara)   #高斯贝叶斯过程
    confusionMatrix = confusion_matrix(test_label, predict_label, labels = ['male', 'female']) #得出混淆矩阵
    male_accuracy_rate = confusionMatrix[0][0]/(confusionMatrix[0][0] + confusionMatrix[0][1])
    female_accuracy_rate = confusionMatrix[1][1]/(confusionMatrix[1][0] + confusionMatrix[1][1])
    accuracy_rate = (confusionMatrix[0][0] + confusionMatrix[1][1]) / len(test_label)
    classes=['male', 'female']
    #paintConfusion_float(confusionMatrix, classes) #绘制混淆矩阵，显示男女声的正确率与错误率
    return male_accuracy_rate, female_accuracy_rate, accuracy_rate

#交互模块
if __name__ == '__main__':
    male_accuracy_rate = []
    female_accuracy_rate = []
    accuracy_rate = []
    temp_accuracy_rate = ()
    times = int(input("输入训练次数:\n"))
    time_start = time.time()
    for i in range(times):
        temp_accuracy_rate = main_()
        male_accuracy_rate.append(temp_accuracy_rate[0])
        female_accuracy_rate.append(temp_accuracy_rate[1])
        accuracy_rate.append(temp_accuracy_rate[2])
    time_end = time.time()
    male_accuracy_rate = np.array(male_accuracy_rate)
    female_accuracy_rate = np.array(female_accuracy_rate)
    accuracy_rate = np.array(accuracy_rate)
    print('male_accuracy_rate: %f' %(male_accuracy_rate.mean()))
    print('female_accuracy_rate: %f' %(female_accuracy_rate.mean()))
    print('accuracy_rate: %f' %(accuracy_rate.mean()))
    print('Time used: %fs' %(time_end - time_start)) 