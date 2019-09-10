# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     time_seq_ex1
   Description :
   Author :       zwb
   date：          2019/9/10
-------------------------------------------------
   Change Activity:
                   2019/9/10:
-------------------------------------------------
"""
__author__ = 'zwb'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

plt.rcParams['font.sans-serif'] = ['SimHei']


def test_time_series(time_series: pd.DataFrame):
    # 可视化处理
    # 时间窗口大小为12
    rol_mean = time_series.rolling(window=12).mean()
    rol_std = time_series.rolling(window=12).std()

    orig = plt.plot(time_series, color='blue', label='Original')
    mean = plt.plot(rol_mean, color='green', label='mean')
    stdv = plt.plot(rol_std, color='red', label='std')

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=["第一个是adt检验的结果，也就是t统计量的值 Test Statistic", "第二个是t统计量的P值p-value",
                                             "第三个是计算过程中用到的延迟阶数#Lags Used",
                                             "第四个是用于ADF回归和计算的观测值的个数Number of Observations Used"])

    for k, v in dftest[4].items():
        dfoutput["Critical Value ({})".format(k)] = v
    print(dfoutput)


if __name__ == '__main__':
    data = pd.read_csv("data/AirPassengers.csv", parse_dates=['Month'], index_col='Month')
    print(data.head())
    ts = data["#Passengers"]
    print(ts.head())

    # plt.plot(ts)
    # plt.show()

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=["第一个是adt检验的结果，也就是t统计量的值 Test Statistic", "第二个是t统计量的P值p-value",
                                             "第三个是计算过程中用到的延迟阶数#Lags Used",
                                             "第四个是用于ADF回归和计算的观测值的个数Number of Observations Used"])

    for k, v in dftest[4].items():
        dfoutput["Critical Value ({})".format(k)] = v
    print(dfoutput)

    # test_time_series(ts)
    # 趋势变换
    ts_log = np.log(ts)  # type:pd.DataFrame
    plt.title("使用滑动窗口")
    plt.plot(ts_log)
    # 趋势变换第二步：移动平均数
    moving_mean = ts_log.rolling(12).mean()
    plt.plot(moving_mean, color='red')
    plt.show()

    ts_log_moving_avg_diff = ts_log - moving_mean
    ts_log_moving_avg_diff.dropna(inplace=True)
    print(ts_log_moving_avg_diff.head(10))
    test_stationarity(ts_log_moving_avg_diff)
    test_time_series(ts_log_moving_avg_diff)

    # 使用加权移动平均法
    expwighted_avg = ts_log.ewm(halflife=12).mean() #type:pd.DataFrame
    plt.title("加权移动平均法")
    plt.plot(ts_log)
    plt.plot(expwighted_avg)
    plt.show()
    ts_log_ewma_diff = ts_log - expwighted_avg
    test_stationarity(ts_log_ewma_diff)
    test_time_series(ts_log_ewma_diff)

