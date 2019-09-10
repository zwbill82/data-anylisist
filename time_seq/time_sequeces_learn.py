# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     time_sequeces_learn
   Description :  时间序列联系
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


class TimeSeuece(object):

    def __init__(self, timeseries: pd.DataFrame) -> None:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        self.ts = timeseries
        self.ts_log = np.log(self.ts)

    def set_plot_title(self,title):
        plt.title(title)

    def test_stationarity(self, timeseries):
        # Determing rolling statistics
        rolmean = timeseries.rolling(window=12).mean()
        rolstd = timeseries.rolling(window=12).std()

        # Plot rolling statistics:
        plt.plot(timeseries, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.show(block=False)

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

    def expwight_smooth(self):
        expwighted_avg = self.ts_log.ewm(halflife=12).mean()  # type:pd.DataFrame
        print("移动加权序列")
        self.set_plot_title("移动加权序列")
        self.test_stationarity(expwighted_avg)
        print("对数-移动加权平均结果")
        self.set_plot_title("对数-移动加权平均结果")
        ts_log_ewma_diff = self.ts_log - expwighted_avg
        self.test_stationarity(ts_log_ewma_diff)

    def diff_smooth(self):
        """
        差分平滑处理
        :return:
        """
        ts_log_diff=self.ts_log.diff().diff()
        ts_log_diff.dropna(inplace=True)
        print("差分平滑处理")
        self.set_plot_title("差分平滑处理")
        self.test_stationarity(ts_log_diff)



