import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from plotly.subplots import make_subplots

class Strategy():
    def __init__(self,df):
        self.df = df
    
    def MovingAverage(self,Period_1:int,Period_2:int):
        Buy = []
        Sell = []
        Flag = 0
        MA_1 = self.df["Close"].rolling(window=Period_1).mean()
        MA_2 = self.df["Close"].rolling(window=Period_2).mean()        

        for i in range(len(self.df)):
            if MA_1.iloc[i] < MA_2.iloc[i] and Flag==0:
                Buy.append(self.df["Close"][i])
                Sell.append(np.nan)
                Flag = 1
            elif MA_1.iloc[i] > MA_2.iloc[i] and Flag==1:
                Sell.append(self.df["Close"][i])
                Buy.append(np.nan)
                Flag = 0
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        
        self.df["Buy_MA"] = Buy
        self.df["Sell_MA"] = Sell
        
        self.Buy = [buy for buy in Buy if str(buy) != "nan"]
        self.Sell = [sell for sell in Sell if str(sell) != "nan"]
        
        self.PnL_MA = pd.DataFrame([self.Buy[0:len(self.Sell)],self.Sell],index=["Buys","Sells"])
        self.PnL_MA = self.PnL_MA.T
        self.PnL_MA["PnL"] = (self.PnL_MA["Sells"] - self.PnL_MA["Buys"]) / self.PnL_MA["Buys"]
        
        fig = go.Figure(data=[go.Candlestick(x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'])])

        fig.add_trace(go.Scatter(x=self.df.index,y=MA_1,line=dict(color="blue"),name=f"Moving Average {Period_1}"))
        fig.add_trace(go.Scatter(x=self.df.index,y=MA_2,line=dict(color="orange"),name=f"Moving Average {Period_2}"))
        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["Buy_MA"],mode="markers",marker_symbol="5",marker_color="purple",name="Compra",marker=dict(size=15)))
        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["Sell_MA"],mode="markers",marker_symbol="6",marker_color="green",name="Venta",marker=dict(size=15)))

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(yaxis_title="Close Price $", xaxis_title="Date", title="Time Serie <br> 2 Moving Averages Strategy")
        fig.update_layout(width=1000,height=800)
        return fig
    
    def Fibonacci(self,Period_ShortEMA:int,Period_LongEMA:int,Period_SignalEMA:int):
        def getLevels(Price:float):
            
            self.Max = self.df["Close"].max()
            self.Min = self.df["Close"].min()
            diff = self.Max - self.Min
            self.level1 = self.Max - 0.236 * diff
            self.level2 = self.Max - 0.382 * diff
            self.level3 = self.Max - 0.5 * diff
            self.level4 = self.Max - 0.618 * diff
            
            if Price >= self.level1:
                return(self.Max,self.level1)
            elif Price >= self.level2:
                return (self.level1,self.level2)
            elif Price >= self.level3:
                return (self.level2,self.level3)
            elif Price >= self.level4:
                return (self.level3,self.level4)
            else:
                return (self.level4,self.Min)
        
        buy_list = []
        sell_list = []
        flag = 0
        last_buy_price = 0
        
        #MACD
        ShortEMA = self.df["Close"].ewm(span=Period_ShortEMA,adjust=False).mean()
        LongEMA = self.df["Close"].ewm(span=Period_LongEMA,adjust=False).mean()
        MACD = ShortEMA - LongEMA
        Signal = MACD.ewm(span=Period_SignalEMA,adjust=False).mean()
        self.df["MACD"] = MACD
        self.df["Signal Line"] = Signal

        for i in range(0,self.df.shape[0]):
            price = self.df["Close"][i]
            if i == 0:
                upper_lvl, lower_lvl = getLevels(price)
                buy_list.append(np.nan)
                sell_list.append(np.nan)
            elif price >= upper_lvl or price <= lower_lvl:
                if self.df["Signal Line"][i] > self.df["MACD"][i] and flag == 0:
                    last_buy_price = price
                    buy_list.append(price)
                    sell_list.append(np.nan)
                    flag = 1
                elif self.df["Signal Line"][i] < self.df["MACD"][i] and flag == 1 and price >= last_buy_price:
                    buy_list.append(np.nan)
                    sell_list.append(price)
                    flag = 0
                else:
                    buy_list.append(np.nan)
                    sell_list.append(np.nan)
            else:
                buy_list.append(np.nan)
                sell_list.append(np.nan)
        
        self.df["Buy_Fibonacci"] = buy_list
        self.df["Sell_Fibonacci"] = sell_list
        
        self.Buy = [buy for buy in buy_list if str(buy) != "nan"]
        self.Sell = [sell for sell in sell_list if str(sell) != "nan"]
        
        self.PnL_Fib = pd.DataFrame([self.Buy[0:len(self.Sell)],self.Sell],index=["Buys","Sells"])
        self.PnL_Fib = self.PnL_Fib.T
        self.PnL_Fib["PnL"] = (self.PnL_Fib["Sells"] - self.PnL_Fib["Buys"]) / self.PnL_Fib["Buys"]

        fig = go.Figure(data=[go.Candlestick(x=self.df.index,
                        open=self.df['Open'],
                        high=self.df['High'],
                        low=self.df['Low'],
                        close=self.df['Close'])])

        fig.add_hrect(y0=self.level1, y1=self.Max, line_width=0, fillcolor="lightsalmon", opacity=0.2,annotation_position="left",annotation_text=f"Level 1: Max = ${round(self.Max,2)} <br> <br> <br> <br> <br> <br> Min = ${round(self.level1,2)}")
        fig.add_hrect(y0=self.level2, y1=self.level1, line_width=0, fillcolor="palegoldenrod", annotation_position="left",opacity=0.2,annotation_text=f"Level 2: Max = ${round(self.level1,2)} <br> <br> <br> Min = ${round(self.level2,2)}")
        fig.add_hrect(y0=self.level3, y1=self.level2, line_width=0, fillcolor="lightblue",annotation_position="left", opacity=0.2,annotation_text=f"Level 3: Max = ${round(self.level2,2)} <br> <br> Min = ${round(self.level3,2)}")
        fig.add_hrect(y0=self.level4, y1=self.level3, line_width=0, fillcolor="grey",annotation_position="left" ,opacity=0.2,annotation_text=f"Level 4: Max = ${round(self.level3,2)} <br> <br> Min = ${round(self.level4,2)}")
        fig.add_hrect(y0=self.Min, y1=self.level4, line_width=0, fillcolor="palegreen",annotation_position="left", opacity=0.2,annotation_text=f"Level 5 <br> Max = ${round(self.level4,2)}  <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> Min = ${round(self.Min,2)}")
        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["Buy_Fibonacci"],mode="markers",marker_symbol="5",marker_color="purple",name="Compra",marker=dict(size=15)))
        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["Sell_Fibonacci"],mode="markers",marker_symbol="6",marker_color="green",name="Venta",marker=dict(size=15)))

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(yaxis_title="Close Price $", xaxis_title="Date", title="Time Serie <br> Fibonacci Strategy")
        fig.update_layout(width=1000,height=800)
        return fig
    
    def MACD(self,Period_ShortEMA:int,Period_LongEMA:int,Period_SignalEMA:int):
        ShortEMA = self.df["Close"].ewm(span=Period_ShortEMA,adjust=False).mean()
        LongEMA = self.df["Close"].ewm(span=Period_LongEMA,adjust=False).mean()
        MACD = ShortEMA - LongEMA
        Signal = MACD.ewm(span=Period_SignalEMA,adjust=False).mean()
        self.df["MACD"] = MACD
        self.df["Signal Line"] = Signal
        
        Buy = []
        Sell = []
        Flag = 0

        for i in range(0,len(self.df)):
            if self.df["MACD"][i] > self.df["Signal Line"][i]:
                Sell.append(np.nan)
                if Flag == 0:
                    Buy.append(self.df["Close"][i])
                    Flag = 1
                else:
                    Buy.append(np.nan)
            elif self.df["MACD"][i] < self.df["Signal Line"][i]:
                Buy.append(np.nan)
                if Flag == 1:
                    Sell.append(self.df["Close"][i])
                    Flag = 0
                else:
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        
        self.df["Buy_MACD"] = Buy
        self.df["Sell_MACD"] = Sell
        
        self.Buy = [buy for buy in Buy if str(buy) != "nan"]
        self.Sell = [sell for sell in Sell if str(sell) != "nan"]
        
        self.PnL_MACD = pd.DataFrame([self.Buy[0:len(self.Sell)],self.Sell],index=["Buys","Sells"])
        self.PnL_MACD = self.PnL_MACD.T
        self.PnL_MACD["PnL"] = (self.PnL_MACD["Sells"] - self.PnL_MACD["Buys"]) / self.PnL_MACD["Buys"]
                
        fig = make_subplots(rows=2,cols=1,subplot_titles=("","MACD"))

        fig.add_trace(go.Candlestick(x=self.df.index,
                        open=self.df['Open'],
                        high=self.df['High'],
                        low=self.df['Low'],
                        close=self.df['Close']))

        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["Buy_MACD"],mode="markers",marker_symbol="5",marker_color="purple",name="Compra",marker=dict(size=15)))
        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["Sell_MACD"],mode="markers",marker_symbol="6",marker_color="green",name="Venta",marker=dict(size=15)))

        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["MACD"],name="Diff Short/Long EMA",mode="lines",line=dict(color="orange")),row=2,col=1)
        fig.add_trace(go.Scatter(x=self.df.index,y=self.df["Signal Line"],name="MACD EMA",mode="lines",line=dict(color="blue")),row=2,col=1)
        fig.add_hline(0,line_dash="dash",line_color="green",row=2,col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(yaxis_title="Close Price $", xaxis_title="Date", title="Time Serie <br> MACD Strategy (Moving Average Convergence Divergence)")
        fig.update_layout(width=1000,height=800)
        return fig