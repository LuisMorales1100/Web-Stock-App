import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        
        #ask = input("Would you like to see the plot: \n 1: Yes \n 2: No \n")
        #if int(ask) == 1:
        plt.plot(self.df.index,self.df["Close"])
        plt.plot(self.df.index,MA_1,label=f"Moving Average {Period_1}")
        plt.plot(self.df.index,MA_2,label=f"Moving Average {Period_2}")
        plt.scatter(x=self.df.index,y=self.df["Buy_MA"],label="Compra",marker="^",color="red",s=100)
        plt.scatter(x=self.df.index,y=self.df["Sell_MA"],marker="v",label="Venta",color="green",s=100)
        plt.legend()
        plt.show()
    
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
        
        #ask = input("Would you like to see the plot: \n 1: Yes \n 2: No \n")
        #if int(ask) == 1:
        plt.subplot(2,1,1)
        plt.plot(self.df.index,self.df["Close"])
        plt.scatter(x=self.df.index,y=self.df["Buy_Fibonacci"],label="Compra",marker="^",color="red",s=100)
        plt.scatter(x=self.df.index,y=self.df["Sell_Fibonacci"],marker="v",label="Venta",color="green",s=100)
        plt.axhspan(self.level1,self.Max,color="lightsalmon",alpha=0.5)
        plt.axhspan(self.level2,self.level1,alpha=0.5,color="palegoldenrod")
        plt.axhspan(self.level3,self.level2,alpha=0.5,color="lightblue")
        plt.axhspan(self.level4,self.level3,alpha=0.5,color="grey")
        plt.axhspan(self.Min,self.level4,alpha=0.5,color="palegreen")
        plt.legend()
        plt.subplot(2,1,2)
        plt.subplot(2,1,2)
        plt.plot(self.df.index,self.df["MACD"],label="Diff Short/Long EMA")
        plt.plot(self.df.index,self.df["Signal Line"],label="MACD EMA")
        plt.axhline(y=0,color="red",linestyle='dashed')
        plt.show()
    
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
        
        #ask = input("Would you like to see the plot: \n 1: Yes \n 2: No \n")
        #if int(ask) == 1:
        plt.subplot(2,1,1)
        plt.plot(self.df.index,self.df["Close"])
        plt.scatter(x=self.df.index,y=self.df["Buy_MACD"],label="Compra",marker="^",color="red",s=100)
        plt.scatter(x=self.df.index,y=self.df["Sell_MACD"],marker="v",label="Venta",color="green",s=100)
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(self.df.index,self.df["MACD"],label="Diff Short/Long EMA")
        plt.plot(self.df.index,self.df["Signal Line"],label="MACD EMA")
        plt.axhline(y=0,color="red",linestyle='dashed')
        plt.legend()
        plt.show()