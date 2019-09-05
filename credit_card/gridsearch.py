# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gridsearch
   Description :
   Author :       zwb
   date：          2019/9/5
-------------------------------------------------
   Change Activity:
                   2019/9/5:
-------------------------------------------------
"""
__author__ = 'zwb'
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

if __name__ == '__main__':
    rf = RandomForestClassifier()
    parameters = {"n_estimators": range(1, 11)}
    iris = load_iris()
    clf = GridSearchCV(estimator=rf, param_grid=parameters)
    clf.fit(iris.data, iris.target)

    print("最优参数：%.4f" % clf.best_score_)
    print("最优参数：", clf.best_params_)
