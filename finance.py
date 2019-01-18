#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:13:19 2018

@author: Mac
"""

import pandas as pd 
import numpy as np 
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings 
import math 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.dates import date2num
from datetime import datetime

class holder():
    1


#detrender 
def detrender(prices, method):
    if method == "difference" :
        detrended = prices.Close[1:] - prices.Close[:-1].values
        
    elif  method == "linear":
        x = np.arange(0, len(prices))
        y = prices.Close.values
        modle = LinearRegression()
        modle.fit(x.reshape(-1,1),y.reshape(-1,1))
        trend = modle.predict(x.reshape(-1,1))
        trend = trend.reshape((len(prices),))
        detrended = prices.Close - trend
    else:
        print("you did not insert valid argument in the function")
    return detrended 



# Fourier Series Expansion Fitting Function F = a0 + a1 cos(wx) + b1 sin(wx)
def fseries(x, a0,a1,b1,w):
    """
    :param x: the hours (independent variable)
    :param a0: first fourier series coeffiecient 
    :param a1: second fourier series coeffiecient
    :param b1: third fourier series coeffiecient
    :param w: fourier series frequency
    :return: the value of the fourier function
    """
    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)
    
    return f
# F = a0 + b1 sin(wx)
def sseries(x, a0,b1,w):
    """
    :param x: the hours (independent variable)
    :param a0: first sine series coeffiecient 
    :param b1: second sine series coeffiecient
    :param w: sine series frequency
    :return: the value of the sine function
    """
    f = a0 + b1 * np.sin(w * x)
    
    return f  
# Fourier series coefficient calculator Function
def fourier(prices, periods, method ="difference",to_plot=False):
    """
    :param prices: OHLC dataframe
    :param periods: list of period for which to compute coefficient[3,5,10 ...]
    :param method: method by which to detrend the data
    :return: dic of dataframes contianing coefficient for said period
    """
    results = holder()
    dict = {} 
    #compute the coeffecient of the series 
    detrended = detrender(prices, method)
    for i in range(0, len(periods)):
        coeffs = [np.nan for k in range(0,4*(periods[i]))] # set the first n windows into nan values (note: detrend remove the first row)
        for j in range(periods[i], len(detrended)+1):
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]
            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
                
                try:
                    res = scipy.optimize.curve_fit(fseries,x,y)
                except(RuntimeError, OptimizeWarning):
                        res = np.empty((1,4))
                        res[0,:] = np.NAN
            
            if to_plot == True:
                xt = np.linspace(0,periods[i],100)
                yt = fseries(xt,res[0][0],res[0][1],res[0][2],res[0][3])
                plt.plot(x,y,'b')
                plt.plot(xt,yt,'r')
                plt.show()
                
            coeffs = np.append(coeffs, res[0],axis = 0)     
        warnings.filterwarnings('ignore', category= np.VisibleDeprecationWarning)
        coeffs = np.array(coeffs).reshape((len(coeffs)//4,4))
        
        df = pd.DataFrame(coeffs, index=prices.iloc[:].index)
        df.columns =['a0_fourier_'+str(periods[i]),'a1_fourier_'+str(periods[i]),'b1_fourier_'+str(periods[i]),'w_fourier_'+str(periods[i])]
        df = df.fillna(method='bfill')
        dict[periods[i]] = df
     
        results.coeffs = dict
        return results

# sine series coefficient calculator Function
def sine(prices, periods, method ="difference",to_plot=False):
    """
    :param prices: OHLC dataframe
    :param periods: list of period for which to compute coefficient[3,5,10 ...]
    :param method: method by which to detrend the data
    :return: dic of dataframes contianing coefficient for said period
    """
    results = holder()
    dict = {}

    #compute the coeffecient of the series 
    detrended = detrender(prices, method)
    for i in range(0, len(periods)):
        coeffs = [np.nan for k in range(0,3*(periods[i]))] # set the first n windows into nan values (note: detrend remove the first row)
        for j in range(periods[i], len(detrended)+1):
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]
            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)
                
                try:
                    res = scipy.optimize.curve_fit(sseries,x,y)
                except(RuntimeError, OptimizeWarning):
                        res = np.empty((1,3))
                        res[0:] = np.NAN
            if to_plot == True:
                xt = np.linspace(0,periods[i],100)
                yt = sseries(xt,res[0][0],res[0][1],res[0][2])
                plt.plot(x,y,'b')
                plt.plot(xt,yt,'r')
                
                plt.show()
            coeffs = np.append(coeffs, res[0],axis = 0)
        warnings.filterwarnings('ignore', category= np.VisibleDeprecationWarning)
        coeffs = np.array(coeffs).reshape(((len(coeffs)//3,3)))
        df = pd.DataFrame(coeffs, index=prices.iloc[:].index)
        df.columns = ['a0_sine_'+str(periods[i]),'b1_sine_'+str(periods[i]),'w_sine_'+str(periods[i])]
        df = df.fillna(method='bfill')
        dict[periods[i]] = df
     
        results.coeffs = dict
        return results      
     

# Williams Accumlation Distrbution Function
def wadl(prices,periods):
    """
    :param prices: dataframe of OHLC prices
    :param periods: (list) periods for which to calculate the function.
    :return: william accumlation distribution line for each period.
    """
    results = holder()
    dict = {}
    for i in range(0,len(periods)):
        WAD = []
        for j in range(periods[i], len(prices)-periods[i]):
            TRH = np.array([prices.High.iloc[j],prices.Close.iloc[j-1]]).max()
            TRL = np.array([prices.Low.iloc[j],prices.Close.iloc[j-1]]).min()
            if prices.Close.iloc[j] > prices.Close.iloc[j-1]:
                PM = prices.Close.iloc[j] - TRL
            elif prices.Close.iloc[j] < price.Close.iloc[j-1]:
                PM = Prices.Close.iloc[j] - TRH
            elif prices.Close.iloc[j] == prices.iloc[j-1]:
                PM = 0
            else:
                print("Unknow error occured, see administrator?")
            AD = PM * prices.Volume.iloc[j]
            WAD = np.append(WAD,AD)
        WAD = WAD.cumsum()
        WAD = pd.DataFrame(WAD, index = prices.iloc[periods[i]:-periods[i]])
        WAD.columns = [['Close']]
        dict[periods[i]] = WAD
    results.wadl = dict
    return results
        
        
        
        
        
        
        
        
        
        
        
        
        
        