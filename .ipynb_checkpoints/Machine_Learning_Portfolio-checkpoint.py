
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import xgboost as xgb
from sklearn.model_selection import train_test_split
import empyrical
from pypfopt.efficient_frontier import EfficientFrontier
import time
import multiprocessing as mp

from functools import partial


tickers = ["GE", "PFE","SBUX", "GME", "DAL", "AAPL", "GOOGL" ]
stocks = pd.concat([yf.Ticker(i).history(period = "max")["Close"].pct_change().dropna().rename(i) for i in tickers], axis=1, join = 'inner')




def Machine_Portfolio(days):
 
    X_sample = stocks.resample(str(days)+"d").mean()
    dates = X_sample.index[1:-1]
    portfolio = pd.DataFrame()

    saved_weights = []
    r0s = []
    sharpes = []

    for day in dates[2:]: 
        sample_1 = stocks[day-pd.offsets.DateOffset(days = 2*days):day-pd.offsets.DateOffset(days=days-1)]
        sample_2 = stocks[day-pd.offsets.DateOffset(days = days):day-pd.offsets.DateOffset(days=1)]
        X1 = sample_1.mean().values.reshape(-1,1)
        X2 = sample_2.mean().values.reshape(-1,1)
        X_data = np.hstack((X1, X2))
        #sample = stocks[day+pd.offsets.DateOffset(days = 7):day+pd.offsets.DateOffset(days=13)]
        #Using Previous Data:
        #mean = sample.mean(axis=0)
        mean = pd.Series(model.predict(X_data), index = sample_1.columns)
        covar = sample_2.cov()
    
    
        weights_vector = [EfficientFrontier(mean,covar).max_quadratic_utility()[ticker] for ticker in stocks.columns]
        portfolio = pd.concat([portfolio,stocks[day:day+pd.offsets.DateOffset(days=days-1)].dot(weights_vector)])
        saved_weights.append(weights_vector)
    portfolio = portfolio.sort_index()
    saved_weights = pd.DataFrame(saved_weights, index = dates[2:], columns = tickers)
    return sharpes.append(empyrical.sharpe_ratio(portfolio))




def run_test(tickers,stocks,windows):
    with mp.Pool(mp.cpu_count()) as p:
        y = p.map(partial(Machine_Portfolio,days=days), windows)
    return y