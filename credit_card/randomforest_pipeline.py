# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     randomforest_pipeline
   Description :  随机森林+pipeline应用例子
   Author :       zwb
   date：          2019/9/8
-------------------------------------------------
   Change Activity:
                   2019/9/8:
-------------------------------------------------
"""
__author__ = 'zwb'
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = load_iris()
    rf = RandomForestClassifier()
    parameters = {"randomforestclassifier__n_estimators": range(1, 11)}
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("randomforestclassifier", rf)
    ])

    clf=GridSearchCV(pipeline,parameters)
    clf.fit(data.data,data.target)
    print(clf.best_score_)
    print(clf.best_params_)

