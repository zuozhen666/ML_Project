# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 22:20:22 2020

@author: brave
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

dataset = pd.read_csv('voice.csv')
x = dataset.values[:, :-1]
y = dataset.values[:, -1]  
feature_num = ['0','1','2','3','4','5','6','7','8','9'
             ,'10','11','12','13','14','15','16','17','18','19']
x_male = np.arange(0, 1584)
x_female = np.arange(1584, 3168)

for i in range(20):
    
    y_ = x[: , i]
    y_male = np.array(y_[0:1584])
    y_female = np.array(y_[1584:3168])
    plt.bar(x_male, y_male, color = 'steelblue', label = 'male')
    plt.bar(x_female, y_female, color = 'red', label = 'female')
    plt.legend()

    plt.xlabel('number')
    plt.ylabel('value')

    plt.title('属性{}'.format(feature_num[i]), FontProperties = font)

    plt.show()