# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     fit_transform_test
   Description :
   Author :       zwb
   date：          2019/9/8
-------------------------------------------------
   Change Activity:
                   2019/9/8:
-------------------------------------------------
"""
__author__ = 'zwb'
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if __name__ == '__main__':
    data = np.array(np.random.randint(-100, 100, 24)).reshape(6, 4)
    train = data[:4]
    test = data[4:]
    minmaxTransformer = MinMaxScaler(feature_range=(0, 1))
    tf=MinMaxScaler.fit_transform(train)
    print(tf)
