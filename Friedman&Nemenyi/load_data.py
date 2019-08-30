# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:45:00 2019

@author: liuxin
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取10个数据
X = []
Y = []
X_train = [0 for i in range(10)]
X_test = [0 for i in range(10)]
Y_train = [0 for i in range(10)]
Y_test = [0 for i in range(10)]
sc = StandardScaler()
for i in range(10):
    data = np.loadtxt("Data" + str(i + 1) + ".txt", dtype = np.float32)
    #对训练集数据进行归一化
    sc.fit(data[:,:-1])
    X.append(sc.transform(data[:,:-1]))
    Y.append(data[:,-1])
    #将数据分为训练集和测试集
    X_train[i], X_test[i], Y_train[i], Y_test[i]= train_test_split(X[i], Y[i], test_size = 0.3, random_state = 1)