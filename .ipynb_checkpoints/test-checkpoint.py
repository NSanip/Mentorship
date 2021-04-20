
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




def Mean_Variance_Rolling(days, tickers, stocks):
    X = stocks.resample(str(days)+"d").mean()
    dates = X.index[1:-1]
    portfolio = pd.DataFrame()

    saved_weights = []
    r0s = []

    for day in dates[:-1]:
        sample = stocks[day:day+pd.offsets.DateOffset(days = (days-1))]
        #sample = stocks[day+pd.offsets.DateOffset(days = 7):day+pd.offsets.DateOffset(days=13)]

        mean = sample.mean(axis=0)
        covar = sample.cov()
        weights_vector = [EfficientFrontier(mean,covar).min_volatility()[ticker] for ticker in stocks.columns]
        portfolio = pd.concat([portfolio,stocks[day+pd.offsets.DateOffset(days = days):day+pd.offsets.DateOffset(days=((2*days)-1))].dot(weights_vector)])
        saved_weights.append(weights_vector)
    portfolio = portfolio.sort_index()
    saved_weights = pd.DataFrame(saved_weights, index = dates[:-1], columns = tickers)
    return portfolio




def run_test(tickers,stocks,windows):
    with mp.Pool(mp.cpu_count()) as p:
        y = p.map(partial(Mean_Variance_Rolling,tickers=tickers,stocks=stocks), windows)
    return y