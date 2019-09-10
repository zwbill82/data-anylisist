# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     bitcoin
   Description :
   Author :       zwb
   date：          2019/9/9
-------------------------------------------------
   Change Activity:
                   2019/9/9:
-------------------------------------------------
"""
__author__ = 'zwb'
import pandas as pd
import  matplotlib.pyplot as plt

if __name__ == '__main__':

    # 数据加载
    df = pd.read_csv("bitcoin_2012-01-01_to_2018-10-31.csv")
    # 设置时间序列2
    df.Timestamp = pd.to_datetime(df.Timestamp)
    df.index = df.Timestamp

    df_month=df.resample('M').mean() #各列的按月份平均值 ，前提要求有时间序列
    df_Q=df.resample('Q-DEC').mean()
    df_year=df.resample('A-DEC').mean()
    #print(df_month["Weighted_Price"])

    fig=plt.figure(figsize=(15,7))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.suptitle("比特币各维度平均值",fontsize=20)
    #按天
    plt.subplot(221)
    plt.plot(df["Weighted_Price"],'-',label='按天')
    plt.legend()
    #plt.show()
    #按月
    plt.subplot(222)
    plt.plot(df_month["Weighted_Price"],'-',label='按月')
    plt.legend()
    #按季度
    plt.subplot(223)
    plt.plot(df_Q["Weighted_Price"],'-',label='按季度')
    plt.legend()
    #按年份
    plt.subplot(224)
    plt.plot(df_year["Weighted_Price"],'-',label='按年份')
    plt.legend()
    plt.show()


