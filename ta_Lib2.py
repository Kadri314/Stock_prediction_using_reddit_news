import numpy  
import pandas as pd  
import math as m

# from pyti import catch_errors
# from pyti.function_helper import fill_for_noncomputable_vals
from six.moves import range


#calculate GARCH model 
from random import gauss
from random import seed
from matplotlib import pyplot
from arch import arch_model
def GARCH(df,n=15):
    #For now it works only for n=15
#     data=df["askclose"].values
    data=(100 *df["Close"].pct_change()).values # this will set the first value into nan
#     data=numpy.log(1+data) #try log return 
    #initialize the first 15 var and mean 
#     GARCH_vars= [data[0:i].var() for i in range(1,16)]
#     GARCH_means=[data[0:i].mean() for i in range(1,16)]

    GARCH_vars= [numpy.nan for i in range(1,n+1)] # set first 15 values into nan 
    GARCH_means=[numpy.nan for i in range(1,n+1)] # set first 15 values into nan
#     i=0
#     j=15
    i=1 # start from 1 since the percentage change didn't account for the first input 
    j=n+1
    # we need to forcast len(data)-15 observations
    for k in range (n,len(data)):
        history=data[i:j]
        # define model
#          ‘ARX’ and ‘HARX’
        model = arch_model(history, mean='Constant', vol='GARCH', p=1, q=3) #1, and 3 according to the paper 
        # fit model
        model_fit = model.fit(disp="off")
        # forecast the next day variance 
        yhat = model_fit.forecast(horizon=1)
        # append the observations 
        GARCH_vars.append(yhat.variance.iloc[yhat.variance.shape[0]-1].values[0])
        GARCH_means.append(yhat.mean.iloc[yhat.mean.shape[0]-1].values[0])
        #update indices
        i+=1
        j+=1
    
    #calculate the actual variance
#     var= [data[i-16:i].var() for i in range(17,len(data)+1)]
#     print(len(GARCH_vars[15:-1]),len(var))
#     pyplot.plot(range(0,84),GARCH_vars[15:-1], c='b')
#     pyplot.plot(range(0,84),var, c= 'g')
#     pyplot.show()
#     print(np.mean(var))
#     print(mean_absolute_error(var,GARCH_vars[15:-1]))

#     appending the results into the data frame 
    df["GARCH_tmw_vars_"+str(n)]=GARCH_vars
    df["GARCH_tmw_means_"+str(n)]=GARCH_means
    return df

#slope calculation
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
def slope(df,n=3):
    close_prices=df["High"].values
    slopes=[numpy.nan for i in range(0,n-1)]
    i=0
    j=n
    for k in range (n,len(close_prices)+1):
        y=close_prices[i:j]
#         X=np.arange(0,n).reshape((-1, 1))
        X=numpy.arange(0,n)
#         model=LinearRegression().fit(X, y)
        model=linregress(X,y=y)
#         slopes.append(model.intercept_)
        slopes.append(model.slope)
        i+=1
        j+=1
    df["slope_"+str(n)]=slopes
    return df

#Percentage change
def percntage_change(df):
    pc=100*df["Close"].pct_change()
    df["percentage_change"]=pc
    return df

#High price average 
def HPA(df,n=2):
    hpa= df["High"].rolling(n).mean()
    hpa=pd.Series(hpa,name="HPA_"+str(n))
    df=df.join(hpa)
    return df

#Low price average 
def LPA(df,n=2):
    lpa= df["Low"].rolling(n).mean()
    lpa=pd.Series(lpa,name="LPA_"+str(n))
    df=df.join(lpa)
    return df

#from https://github.com/kylejusticemagnuson/pyti/blob/master/pyti/weighted_moving_average.py
def WMA(data, period):
    """
    Weighted Moving Average.
    Formula:
    (P1 + 2 P2 + 3 P3 + ... + n Pn) / K
    where K = (1+2+...+n) = n(n+1)/2 and Pn is the most recent price
    """
#     catch_errors.check_for_period_error(data, period)
    k = (period * (period + 1)) / 2.0

    wmas = []
    for idx in range(0, len(data)-period+1):
        product = [data[idx + period_idx] * (period_idx + 1) for period_idx in range(0, period)]
        wma = sum(product) / k
        wmas.append(wma)
    wmas = fill_for_noncomputable_vals(data, wmas)

    return wmas

#Weighted Closing Price (WPC)
def WPC(df,n):
    # got the formula from : https://www.metastock.com/customer/resources/taaz/?p=124
    wpc=((df["Close"]*2)+df["High"]+df["Low"])/4
    wpc= pd.Series(wpc, name='wpc')
    df=df.join(wpc)
    return df

# Accumulation Distribution
def accumulation_distribution(df):
    # got the formula from: https://www.investopedia.com/terms/a/accumulationdistribution.asp
    wad=(((df["Close"]-df["Low"])-(df["High"]-df["Close"]))/(df["High"]-df["Low"]))*df["Volume"] #((close – low) – (high - close) / (high – low)) * volume
    wad=pd.Series(wad, name='WAD')
    df=df.join(wad)
    return df

#Heiken Ashi from : https://github.com/arkochhar/Technical-Indicators/blob/master/indicator/indicators.py
def HA(df, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Heiken Ashi Candles (HA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Heiken Ashi Close (HA_$ohlc[3])
            Heiken Ashi Open (HA_$ohlc[0])
            Heiken Ashi High (HA_$ohlc[1])
            Heiken Ashi Low (HA_$ohlc[2])
    """

    ha_open = 'HA_' + ohlc[0]
    ha_high = 'HA_' + ohlc[1]
    ha_low = 'HA_' + ohlc[2]
    ha_close = 'HA_' + ohlc[3]
    
    df[ha_close] = (df[ohlc[0]] + df[ohlc[1]] + df[ohlc[2]] + df[ohlc[3]]) / 4

    df[ha_open] = 0.00
    for i in range(0, len(df)):
        if i == 0:
            df[ha_open].iat[i] = (df[ohlc[0]].iat[i] + df[ohlc[3]].iat[i]) / 2
        else:
            df[ha_open].iat[i] = (df[ohlc[0]].iat[i - 1] + df[ohlc[3]].iat[i - 1]) / 2
            
    df[ha_high]=df[[ohlc[0], ohlc[1], ohlc[3]]].max(axis=1)
    df[ha_low]=df[[ohlc[0], ohlc[2], ohlc[3]]].min(axis=1)

    return df

"""
William's Accumulation/Distribution
Source: https://www.metastock.com/customer/resources/taaz/?p=125
Params: 
    data: pandas DataFrame
    high_col: the name of the HIGH values column
    low_col: the name of the LOW values column
    close_col: the name of the CLOSE values column
    
Returns:
    copy of 'data' DataFrame with 'williams_ad' column added
"""
def williams_ad(data, high_col='High', low_col='Low', close_col='Close'):
    # got the code from:  https://github.com/voice32/stock_market_indicators/blob/master/indicators.py
    data['williams_ad'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_value = data.at[index-1, 'williams_ad']
            prev_close = data.at[index-1, close_col]
            if row[close_col] > prev_close:
                ad = row[close_col] - min(prev_close, row[low_col])
            elif row[close_col] < prev_close:
                ad = row[close_col] - max(prev_close, row[high_col])
            else:
                ad = 0.
                                                                                                        
            data.at[index, 'williams_ad']=(ad+prev_value)
        
    return data

def A_D(df):
    ad = ((df["High"]-df["Close"].shift()) / (df["High"]-df["Low"])) 
    ad=   pd.Series(ad, name='A\D')
    df=df.join(ad)
    return df


#Moving Average  
def MA(df, n):  
    MA = pd.Series(pd.rolling_mean(df['Close'], n), name = 'MA_' + str(n))  
    df = df.join(MA)  
    return df

#Exponential Moving Average  
def EMA(df, n):  
    EMA=pd.Series(df['Close'].ewm(span = n, min_periods = n-1 ).mean(), name = 'EMA_' + str(n)) 
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['Close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['Close'].diff(n - 1)  
    N = df['Close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)  
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR= pd.Series(TR_s.ewm(span = n, min_periods = n).mean(), name = 'ATR_' + str(n))  
    
    df = df.join(ATR)  
    return df

# #Bollinger Bands  
# def BBANDS(df, n):  
#     MA = pd.Series(pd.rolling_mean(df['Close'], n))  
#     MSD = pd.Series(pd.rolling_std(df['Close'], n))  
#     b1 = 4 * MSD / MA  
#     B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
#     df = df.join(B1)  
#     b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)  
#     B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))  
#     df = df.join(B2)  
#     return df

#Bollinger Bands  
def BBANDS(df, n):
    # source: https://traderhq.com/ultimate-guide-to-bollinger-bands/
    MA = pd.Series(df['Close'].rolling(window= n).mean())
    MSD = pd.Series(df['Close'].rolling(window=n).std())  
    
    middle = MA
    upper= MA+(2*MSD)
    lower=MA-(2*MSD)
    
#     middle=pd.Series(middle, name = 'BollingerB_middle_' + str(n))
    upper=pd.Series(upper, name = 'BollingerB_upper_' + str(n))
    lower=pd.Series(lower, name = 'BollingerB_lower_' + str(n))

#     df = df.join(middle)  
    df = df.join(upper)  
    df = df.join(lower)  

    return df

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)  
    R1 = pd.Series(2 * PP - df['Low'])  
    S1 = pd.Series(2 * PP - df['High'])  
    R2 = pd.Series(PP + df['High'] - df['Low'])  
    S2 = pd.Series(PP - df['High'] + df['Low'])  
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))  
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR)  
    return df




# taken from : http://www.andrewshamlet.net/2017/07/13/python-tutorial-stochastic-oscillator/
def STOK(df, n): 
    STOK = ((df["Close"] - df["Low"].rolling(n).min()) / (df["High"].rolling(n).max() - df["Low"].rolling(n).min())) * 100
    STOK=pd.Series(STOK, name='STO%K_'+str(n))
    df=df.join(STOK)
    return df
# taken from : http://www.andrewshamlet.net/2017/07/13/python-tutorial-stochastic-oscillator/
def STOD(df, n):
    STOK = ((df["Close"] - df["Low"].rolling(n).min()) / (df["High"].rolling(n).max() - df["Low"].rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    STOD =  pd.Series(STOD, name='STO%D_'+str(n))
    df=df.join(STOD)
    return df
    
                    

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)  
def STO(df,  nK, nD, nS=1):  
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))  
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))  
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    df = df.join(SOk)  
    df = df.join(SOd)  
    return df  

# Stochastic Oscillator, SMA smoothing, nS = slowing (1 if no slowing)  
def STO(df, nK, nD,  nS=1):  
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))  
    SOd = pd.Series(SOk.rolling(window=nD, center=False).mean(), name = 'SO%d'+str(nD))  
    SOk = SOk.rolling(window=nS, center=False).mean()  
    SOd = SOd.rolling(window=nS, center=False).mean()  
    df = df.join(SOk)  
    df = df.join(SOd)  
    return df  
#Trix  
def TRIX(df, n):  
    EX1 = df['Close'].ewm(span = n, min_periods = n - 1).mean()
    EX2 = EX1.ewm(span = n, min_periods = n - 1).mean()
    EX3 = EX2.ewm(span = n, min_periods = n - 1).mean()  
    i = 0  
    ROC_l = [0]  
    while i + 1 <= df.index[-1]:  
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]  
        ROC_l.append(ROC)  
        i = i + 1  
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))  
    df = df.join(Trix)  
    return df

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.at[i + 1, 'High'] - df.at[i, 'High']  
        DoMove = df.at[i, 'Low'] - df.at[i + 1, 'Low']  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(TR_s.ewm(span = n, min_periods = n).mean())  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(UpI.ewm(span = n, min_periods = n - 1).mean() / ATR)  
    NegDI = pd.Series(DoI.ewm(span = n, min_periods = n - 1).mean() / ATR)
    val=abs(PosDI - NegDI) / (PosDI + NegDI)
    ADX = pd.Series(val.ewm(span = n_ADX, min_periods = n_ADX - 1).mean(), name = 'ADX_' + str(n) + '_' + str(n_ADX))  
    df = df.join(ADX)  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(df['Close'].ewm(span = n_fast, min_periods = n_slow - 1).mean())  
    EMAslow = pd.Series(df['Close'].ewm( span = n_slow, min_periods = n_slow - 1).mean())  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(MACD.ewm(span = 9, min_periods = 8).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

#Mass Index  
def MassI(df):  
    Range = df['High'] - df['Low']  
    EX1 = Range.ewm( span = 9, min_periods = 8).mean()
    EX2 = EX1.ewm(span = 9, min_periods = 8).mean()
    Mass = EX1 / EX2  
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.at[i + 1, 'High'], df.at[i, 'Close']) - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.at[i + 1, 'High'] - df.at[i, 'Low']) - abs(df.at[i + 1, 'Low'] - df.at[i, 'High'])  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))  
    df = df.join(VI)  
    return df





#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['Close'].diff(r1 - 1)  
    N = df['Close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(r2 - 1)  
    N = df['Close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['Close'].diff(r3 - 1)  
    N = df['Close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['Close'].diff(r4 - 1)  
    N = df['Close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df

# def RSI(df, n):  
#     i = 0  
#     UpI = [0]  
#     DoI = [0]  
#     while i + 1 <= df.index[-1]:  
#         UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')  
#         DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
#         if UpMove > DoMove and UpMove > 0:  
#             UpD = UpMove  
#         else: UpD = 0  
#         UpI.append(UpD)  
#         if DoMove > UpMove and DoMove > 0:  
#             DoD = DoMove  
#         else: DoD = 0  
#         DoI.append(DoD)  
#         i = i + 1  
#     UpI = pd.Series(UpI)  
#     DoI = pd.Series(DoI)  
#     PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))  
#     NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))  
#     RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
#     df = df.join(RSI)  
#     return df    


# def rsi(close, n=14, fillna=False):
#     """Relative Strength Index (RSI)
#     Compares the magnitude of recent gains and losses over a specified time
#     period to measure speed and change of price movements of a security. It is
#     primarily used to attempt to identify overbought or oversold conditions in
#     the trading of an asset.
#     https://www.investopedia.com/terms/r/rsi.asp
#     Args:
#         close(pandas.Series): dataset 'Close' column.
#         n(int): n period.
#         fillna(bool): if True, fill nan values.
#     Returns:
#         pandas.Series: New feature generated.
#     """
#     diff = close.diff()
#     which_dn = diff < 0

#     up, dn = diff, diff*0
#     up[which_dn], dn[which_dn] = 0, -up[which_dn]

#     emaup = EMA(up, n)
#     emadn = EMA(dn, n)

#     rsi = 100 * emaup / (emaup + emadn)
#     if fillna:
#         rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
#     return pd.Series(rsi, name='rsi')

def rsi(df, period=14):
    prices=df["Close"].values
    """
    The Relative Strength Index (RSI) is a momentum oscillator.
    It oscillates between 0 and 100.
    It is considered overbought/oversold when it's over 70/below 30.
    Some traders use 80/20 to be on the safe side.
    RSI becomes more accurate as the calculation period (min_periods)
    increases.
    This can be lowered to increase sensitivity or raised to decrease
    sensitivity.
    10-day RSI is more likely to reach overbought or oversold levels than
    20-day RSI. The look-back parameters also depend on a security's
    volatility.
    Like many momentum oscillators, overbought and oversold readings for RSI
    work best when prices move sideways within a range.
    You can also look for divergence with price.
    If the price has new highs/lows, and the RSI hasn't, expect a reversal.
    Signals can also be generated by looking for failure swings and centerline
    crossovers.
    RSI can also be used to identify the general trend.
    The RSI was developed by J. Welles Wilder and was first introduced in his
    article in the June, 1978 issue of Commodities magazine, now known as
    Futures magazine. It is detailed in his book New Concepts In Technical
    Trading Systems.
    http://www.csidata.com/?page_id=797
    http://stockcharts.com/help/doku.php?id=chart_school:technical_indicators:relative_strength_in
    Input:
      prices ndarray
      period int > 1 and < len(prices) (optional and defaults to 14)
    Output:
      rsis ndarray
    Test:
    >>> import numpy as np
    >>> import technical_indicators as tai
    >>> prices = np.array([44.55, 44.3, 44.36, 43.82, 44.46, 44.96, 45.23,
    ... 45.56, 45.98, 46.22, 46.03, 46.17, 45.75, 46.42, 46.42, 46.14, 46.17,
    ... 46.55, 46.36, 45.78, 46.35, 46.39, 45.85, 46.59, 45.92, 45.49, 44.16,
    ... 44.31, 44.35, 44.7, 43.55, 42.79, 43.26])
    >>> print(tai.rsi(prices))
    [ 70.02141328  65.77440817  66.01226849  68.95536568  65.88342192
      57.46707948  62.532685    62.86690858  55.64975092  62.07502976
      54.39159393  50.10513101  39.68712141  41.17273382  41.5859395
      45.21224077  37.06939108  32.85768734  37.58081218]
    """

    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    # this could be named gains/losses to save time/memory in the future
    changes = prices[1:] - prices[:-1]
    #num_changes = len(changes)

    rsi_range = num_prices - period

    rsis = numpy.zeros(rsi_range)

    gains = numpy.array(changes)
    # assign 0 to all negative values
    masked_gains = gains < 0
    gains[masked_gains] = 0

    losses = numpy.array(changes)
    # assign 0 to all positive values
    masked_losses = losses > 0
    losses[masked_losses] = 0
    # convert all negatives into positives
    losses *= -1

    avg_gain = numpy.mean(gains[:period])
    avg_loss = numpy.mean(losses[:period])

    if avg_loss == 0:
        rsis[0] = 100
    else:
        rs = avg_gain / avg_loss
        rsis[0] = 100 - (100 / (1 + rs))

    for idx in range(1, rsi_range):
        avg_gain = ((avg_gain * (period - 1) + gains[idx + (period - 1)]) /
                    period)
        avg_loss = ((avg_loss * (period - 1) + losses[idx + (period - 1)]) /
                    period)

        if avg_loss == 0:
            rsis[idx] = 100
        else:
            rs = avg_gain / avg_loss
            rsis[idx] = 100 - (100 / (1 + rs))
            
    rsis = pd.Series(rsis,name = 'RSI_' + str(period))
    nan= pd.Series([ float('nan') for i in range(0,period)],name = 'RSI_' + str(period))
    rsis=nan.append(rsis,ignore_index=True)
    df = df.join(rsis) 
    return df

#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['Close'].diff(1))  
    aM = abs(M)  
    EMA1 = pd.Series(M.ewm(span = r, min_periods = r - 1).mean())  
    aEMA1 = pd.Series(aM.ewm(span = r, min_periods = r - 1).mean())  
    EMA2 = pd.Series(EMA1.ewm(span = s, min_periods = s - 1).mean())  
    aEMA2 = pd.Series(aEMA1.ewm(span = s, min_periods = s - 1).mean())  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI)  
    return df

#Accumulation/Distribution  
def ACCDIST(df, n):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))  
    df = df.join(AD)  
    return df

# #Chaikin Oscillator  
# def Chaikin(df):  
#     ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
#     Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')  
#     df = df.join(Chaikin)  
#     return df


"""
Code from : https://github.com/voice32/stock_market_indicators/blob/master/indicators.py
Chaikin Oscillator
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_oscillator
Params: 
    data: pandas DataFrame
    periods_short: period for the shorter EMA (3 days recommended)
    periods_long: period for the longer EMA (10 days recommended)
    high_col: the name of the HIGH values column
    low_col: the name of the LOW values column
    close_col: the name of the CLOSE values column
    vol_col: the name of the VOL values column
    
Returns:
    copy of 'data' DataFrame with 'ch_osc' column added
"""
def chaikin_oscillator(data, periods_short=3, periods_long=10, high_col='High',
                       low_col='Low', close_col='Close', vol_col='Volume'):
    ac = pd.Series([])
    val_last = 0

    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            val = val_last + ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            val = val_last
        ac.at[index]= val
    val_last = val

    ema_long = ac.ewm(ignore_na=False, min_periods=0, com=periods_long, adjust=True).mean()
    ema_short = ac.ewm(ignore_na=False, min_periods=0, com=periods_short, adjust=True).mean()
    data['ch_osc_s'+str(periods_short)] = ema_short - ema_long

    return data

#Money Flow Index and Ratio  
def MFI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    i = 0  
    PosMF = [0]  
    while i < df.index[-1]:  
        if PP[i + 1] > PP[i]:  
            PosMF.append(PP[i + 1] * df.at[i + 1, 'Volume'])  
        else:  
            PosMF.append(0)  
        i = i + 1  
    PosMF = pd.Series(PosMF)  
    TotMF = PP * df['Volume']  
    MFR = pd.Series(PosMF / TotMF)  
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))  
    df = df.join(MFI)  
    return df

#On-balance Volume  
def OBV(df, n):  
    i = 0  
    OBV = [0]  
    while i < df.index[-1]:  
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] > 0:  
            OBV.append(df.at[i + 1, 'Volume'])  
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] == 0:  
            OBV.append(0)  
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] < 0:  
            OBV.append(-df.at[i + 1, 'Volume'])  
        i = i + 1  
    OBV = pd.Series(OBV)  
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))  
    df = df.join(OBV_ma)  
    return df

#Force Index  
def FORCE(df, n):  
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))  
    df = df.join(F)  
    return df

#Ease of Movement  
def EOM(df, n):  
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])  
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))  
    df = df.join(Eom_ma)  
    return df

def cci(df, n=20, c=0.015, fillna=False):
    """Commodity Channel Index (CCI)
    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        c(int): constant.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    pp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cci = (pp - pp.rolling(n).mean()) / (c * pp.rolling(n).std())
    if fillna:
        cci = cci.replace([np.inf, -np.inf], np.nan).fillna(0)
    cci= pd.Series(cci, name='cci_'+str(n))
    df=df.join(cci)
    return df

#Coppock Curve  
def COPP(df, n):  
    M = df['Close'].diff(int(n * 11 / 10) - 1)  
    N = df['Close'].shift(int(n * 11 / 10) - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(int(n * 14 / 10) - 1)  
    N = df['Close'].shift(int(n * 14 / 10) - 1)  
    ROC2 = M / N
    val=ROC1 + ROC2
    Copp = pd.Series(val.ewm(span = n, min_periods = n).mean(), name = 'Copp_' + str(n))  
    df = df.join(Copp)  
    return df

#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

#Ultimate Oscillator  
def ULTOSC(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        TR_l.append(TR)  
        BP = df.at[i + 1, 'Close'] - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Donchian Channel  
def DONCH(df, n):  
    i = 0  
    DC_l = []  
    while i < n - 1:  
        DC_l.append(0)  
        i = i + 1  
    i = 0  
    while i + n - 1 < df.index[-1]:  
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])  
        DC_l.append(DC)  
        i = i + 1  
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))  
    DonCh = DonCh.shift(n - 1)  
    df = df.join(DonCh)  
    return df

#Standard Deviation  
def STDDEV(df, n):  
    df = df.join(pd.Series(pd.rolling_std(df['Close'], n), name = 'STD_' + str(n)))  
    return df  

def wr(df, lbp=14, fillna=False):
    """Williams %R
    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
    Developed by Larry Williams, Williams %R is a momentum indicator that is the inverse of the
    Fast Stochastic Oscillator. Also referred to as %R, Williams %R reflects the level of the close
    relative to the highest high for the look-back period. In contrast, the Stochastic Oscillator
    reflects the level of the close relative to the lowest low. %R corrects for the inversion by
    multiplying the raw value by -100. As a result, the Fast Stochastic Oscillator and Williams %R
    produce the exact same lines, only the scaling is different. Williams %R oscillates from 0 to -100.
    Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.
    Unsurprisingly, signals derived from the Stochastic Oscillator are also applicable to Williams %R.
    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.
    From: https://www.investopedia.com/terms/w/williamsr.asp
    The Williams %R oscillates from 0 to -100. When the indicator produces readings from 0 to -20, this indicates
    overbought market conditions. When readings are -80 to -100, it indicates oversold market conditions.
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period
        fillna(bool): if True, fill nan values with -50.
    Returns:
        pandas.Series: New feature generated.
    """

    hh = df["High"].rolling(lbp).max() #highest high over lookback period lbp
    ll = df["Low"].rolling(lbp).min()  #lowest low over lookback period lbp

    wr = -100 * (hh - df["Close"]) / (hh - ll)

    if fillna:
        wr = wr.replace([np.inf, -np.inf], np.nan).fillna(-50)
    wr= pd.Series(wr, name='wr_'+str(lbp))
    df=df.join(wr)
    return df                                                                                      
