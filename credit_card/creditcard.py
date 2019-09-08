# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     creditcard
   Description :
   Author :       zwb
   date：          2019/9/5
-------------------------------------------------
   Change Activity:
                   2019/9/5:
-------------------------------------------------
"""
__author__ = 'zwb'
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def GridSearchCV_work(pipeline: Pipeline, train_x, train_y, test_x, test_y, param_grid, score='accuracy'):
    grid_serch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score,cv=5)
    search = grid_serch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" % search.best_score_)
    predict_y = search.predict(test_x)
    print("预测分数为：%04lf" % accuracy_score(test_y, predict_y))

if __name__ == '__main__':
    # 1加载数据
    data = pd.read_csv("./UCI_Credit_Card.csv")
    print(data.shape)
    print(data.describe())

    is_show = 0
    if is_show:
        next_month = data["default.payment.next.month"].value_counts()
        print(next_month)
        print(next_month.index)
        print(next_month.values)
        df = pd.DataFrame({"default.payment.next.month": next_month.index, "values": next_month.values})
        # draw a figure
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.figure(figsize=(10, 10))
        plt.title('信用卡违约率客户\n (违约：1，守约：0)')
        sns.set_color_codes("pastel")
        sns.barplot(x='default.payment.next.month', y="values", data=df)
        locs, labels = plt.xticks()
        plt.show()

    # x,y define
    target = data["default.payment.next.month"].values
    columns = data.columns.tolist()  # type: list
    columns.remove("ID")
    columns.remove("default.payment.next.month")
    features = data[columns].values
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify=target,
                                                        random_state=1)

    # 构造各种分类器，固定参数可以写到里面
    classfiers = [
        SVC(random_state=1, kernel='rbf'),
        DecisionTreeClassifier(random_state=1, criterion='gini'),
        RandomForestClassifier(random_state=1, criterion='gini'),
        KNeighborsClassifier(metric='minkowski')
    ]

    classfier_name = [
        'svc',
        'decisiontreeclassfier',
        'randomforestclassifier',
        'kneighborsclassifier'
    ]

    classfier_params = [
        {'svc__C': [1], 'svc__gamma': [0.01]},
        {'decisiontreeclassfier__max_depth': [6, 9, 11]},
        {'randomforestclassifier___n_estimators': [3, 5, 6]},
        {'kneighborsclassifier__n_neighbors': [4, 6, 8]}
    ]


    for model,model_name,model_param_grid in  zip(classfiers,classfier_name,classfier_params):
        print("处理模型:{}".format(model_name))
        pipeline=Pipeline([
            ('scale',StandardScaler()),
            (model_name,model)
        ])

        GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,model_param_grid,score='accuracy')
