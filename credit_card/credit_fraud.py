# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     credit_fraud
   Description :
   Author :       zwb
   date：          2019/9/8
-------------------------------------------------
   Change Activity:
                   2019/9/8:
-------------------------------------------------
"""
__author__ = 'zwb'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
if __name__ == '__main__':
    data = pd.read_csv("creditcard.csv")
    print(data.describe())
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    ax = sns.countplot(x='Class', data=data)
    plt.show()

    # 显示交易笔数，欺诈交易笔数
    num_trade = len(data)
    num_fraud = len(data[data['Class'] == 1])
    print("总交易笔数{}，其中，欺诈交易笔数{}".format(num_trade, num_fraud))

    # 欺诈和正常交易可视化
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    bins = 50
    ax1.hist(data.Time[data.Class == 1], bins=bins, color='deeppink')
    ax1.set_title('诈骗交易')
    ax2.hist(data.Time[data.Class == 0], bins=bins, color='deepskyblue')
    ax2.set_title('正常交易')
    plt.xlabel('时间')
    plt.ylabel('交易次数')
    plt.show()

    # 对Amount进行数据规范化
    data["Amount_Norm"] = StandardScaler().fit_transform(data["Amount"].values.reshape(-1, 1))
    # 特征选择
    y = np.array(data["Class"].tolist())
    data = data.drop(['Time', "Amount", "Class"], axis=1)
    x = np.array(data.as_matrix())
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.1,random_state=33)

    # 逻辑回归分类
    clf=LogisticRegression()
    clf.fit(train_x,train_y)
    predict_y=clf.predict(test_x)
    score_y=clf.decision_function(test_x)
    # 计算显混淆矩阵，并示
    cm=confusion_matrix(test_y,predict_y)
    print(cm)

