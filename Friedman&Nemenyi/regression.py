# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:28:13 2019

@author: liuxin
"""

from sklearn import linear_model
from sklearn.metrics import mean_squared_error


#导入处理过的数据
from load_data import X_train, Y_train, X_test, Y_test

#创建四个数组记录每个模型的得分
linear_scores = [0 for i in range(10)]
ridge_scores = [0 for i in range(10)]
lasso_scores = [0 for i in range(10)]
elasticNet_scores = [0 for i in range(10)]

#评估函数MSE
MSE = mean_squared_error

# 线性回归
model_LR = linear_model.LinearRegression()
for i in range(10):
    model_LR.fit(X_train[i], Y_train[i])
    Y_pred = model_LR.predict(X_test[i])
    linear_scores[i] = MSE(Y_test[i], Y_pred)

# 岭回归
model_RR = linear_model.Ridge()
for i in range(10):
    model_RR.fit(X_train[i], Y_train[i])
    Y_pred = model_RR.predict(X_test[i])
    ridge_scores[i] = MSE(Y_test[i], Y_pred)

# Lasso回归
model_Lasso = linear_model.Lasso()
for i in range(10):
    model_Lasso.fit(X_train[i], Y_train[i])
    Y_pred = model_Lasso.predict(X_test[i])
    lasso_scores[i] = MSE(Y_test[i], Y_pred)

# ElasticNet回归
model_elastic = linear_model.ElasticNet()
for i in range(10):
    model_elastic.fit(X_train[i], Y_train[i])
    Y_pred = model_elastic.predict(X_test[i])
    elasticNet_scores[i] = MSE(Y_test[i], Y_pred)