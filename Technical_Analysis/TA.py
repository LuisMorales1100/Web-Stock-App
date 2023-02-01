import pandas as pd
import ta 
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import Web_App_Stock_Prediction.ForecastModel as FM
import seaborn as sns
import imp
import Technical_Analysis.Strategies as TAS


def GetData(Tickets:list,StartDate:str,EndDate:str):
    data = yf.download(Tickets,StartDate,EndDate)
    return data

# ("AAPL","MSFT","WALMEX.MX","ABNB","WMT","AMD","BIMBOA.MX","^GSPC")

Tickets = ["WALMEX.MX"]
data = GetData(Tickets=Tickets,StartDate="2022-01-01",EndDate="2023-01-31")
data

imp.reload(FM)
imp.reload(TAS)

Model = FM.Forecast_Model(data)
Model.Model(0.90)
Model.Prediction(25)
Model.Plot_2()
Model.Assumptions()

Signals = TAS.Strategy(data)
Signals.MovingAverage(10,35)
Signals.Fibonacci(5,26,10)
Signals.MACD(4,30,15)


(data["Buy_MA"].sum() - data["Sell_MA"].sum()) / data["Buy_MA"].sum()



bb=ta.volatility.BollingerBands(close=data["Adj Close"],window=15,window_dev=2)
rsi = ta.momentum.RSIIndicator(data["Close"])
sma = ta.trend.SMAIndicator(data["Close"],window=5)
data_bb = data.copy()
data_bb["Bollinger_MA"] = bb.bollinger_mavg()
data_bb["Bollinger_High"] = bb.bollinger_hband()
data_bb["Bollinger_Low"] = bb.bollinger_lband()
data_bb["RSI"] = rsi.rsi()
data_bb["SMA"] = sma.sma_indicator()

data_bb[["Close","Bollinger_MA","Bollinger_High","Bollinger_Low","RSI","SMA"]].plot(title="Bollinger Bands")
plt.show()
