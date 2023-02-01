import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from datetime import timedelta
from plotly import graph_objs as go
import plotly.express as px
import streamlit as st
import scipy.stats as stats
import pylab

class Forecast_Model():
    def __init__(self,df):
        self.df = df[["Close"]]
    
    def __Preprocessing(self):
        self.scaler = MinMaxScaler()
        self.df["scale"] = self.scaler.fit_transform(self.df["Close"].values.reshape(-1,1))
        self.x = np.array(self.df["scale"].values.reshape(-1,1))
        self.y = np.array(self.df["scale"].values.reshape(-1,1))
    
    def Train_Test(self,sizeTrain_Proportion):
        size = round(len(self.df)*sizeTrain_Proportion)
        self.x_train,self.x_test = self.x[0:size], self.x[size:]
        self.y_train, self.y_test = self.y[0:size], self.y[size:]

    def Model(self,sizeTrain_Proportion):
        self.__Preprocessing()
        self.Train_Test(sizeTrain_Proportion)
        self.mlp = MLPRegressor(hidden_layer_sizes=(100,50,25),activation="tanh",solver="adam")
        self.mlp.fit(self.x_train,self.y_train)
        self.train_predict = self.mlp.predict(self.x_train)
        self.train_predict_inv = self.scaler.inverse_transform(self.train_predict.reshape(-1,1))
        #print("Coefficient of Determination (Training):",round(self.mlp.score(self.x_train,self.y_train),2))
        st.write("-"*80)
        st.write("Coefficient of Determination (Training):", round(self.mlp.score(self.x_train,self.y_train),2))
        test_pred = self.mlp.predict(self.x_test)
        #print("Coefficient of Determination (Testing):",round(r2_score(test_pred,self.y_test),2))
        st.write("\n")
        st.write("Coefficient of Determination (Testing):", round(r2_score(test_pred,self.y_test),2))

    def Prediction(self,Days_Prediction):
        self.ts = np.array(self.df["Close"])
        self.DaysPredic = Days_Prediction
        for i in range(self.DaysPredic):
            t = (np.array(self.ts[-2:]).sum() / 2) + np.random.normal(loc=0,scale=1) # (np.amax(self.ts[-4:])/2 - np.amin(self.ts[-4:])/2)**(1/3)
            #np.random.randint(-np.round(np.amax(self.ts[-4:]) - np.amin(self.ts[-4:])),np.round(np.amax(self.ts[-4:]) - np.amin(self.ts[-4:])))
            self.ts = np.append(self.ts,t)
        
        self.x_predict = self.scaler.fit_transform(np.array(self.ts[-Days_Prediction:].reshape(-1,1)))
        self.x_predicted = self.mlp.predict(self.x_predict)
        self.x_predicted = self.scaler.inverse_transform(self.x_predicted.reshape(-1,1))
        return self.x_predicted
    
    def Plot(self):
        self.index = self.df.index.tolist()
        for i in range(self.DaysPredic):
            self.index.append(self.index[-1] + timedelta(1))
        self.df["Category"] = ["Close" for name in range(len(self.df))]
        df_x_predicted = pd.Series(self.x_predicted.ravel())
        df_x_predicted = df_x_predicted.to_frame(name="Close")
        self.new_df = pd.concat([self.df,df_x_predicted],axis=0)
        self.new_df.index = self.index
        self.new_df["Category"] = self.new_df["Category"].fillna("Predicted")
        
        fig = px.line(self.new_df,x=self.new_df.index,y="Close",color="Category",color_discrete_sequence=["#0DF9FF", "red"],title="Forecasting Modeling",labels={"index":"Date"})
        
        '''
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.new_df.index, y=self.new_df['Close']))
        fig.layout.update(title_text='Time Series', xaxis_rangeslider_visible=False)'''
        st.plotly_chart(fig)
        
    def Plot_2(self):
        self.index = self.df.index.tolist()
        for i in range(self.DaysPredic):
            self.index.append(self.index[-1] + timedelta(1))
        self.df["Category"] = ["Close" for name in range(len(self.df))]
        df_x_predicted = pd.Series(self.x_predicted.ravel())
        df_x_predicted = df_x_predicted.to_frame(name="Close")
        self.new_df = pd.concat([self.df,df_x_predicted],axis=0)
        self.new_df.index = self.index
        self.new_df["Category"] = self.new_df["Category"].fillna("Predicted")
        sns.lineplot(data=self.new_df,x=self.new_df.index,y="Close",hue="Category")
        plt.show()
        
    def Assumptions(self):
        self.resid = [(self.x_train[i][0] - self.train_predict[i]) for i in range(self.x_train.shape[0])]
        fig = plt.figure(figsize=(12,12))
        plt.subplot(2,2,1)
        # Normality
        sns.histplot(x=self.resid)
        plt.title("Histogram for Residuals")
        plt.xlabel("Residuals")
        plt.subplot(2,2,3)
        stats.probplot(self.resid, dist="norm",plot=plt)
        
        # Homocedasticity
        plt.subplot(2,2,2)
        sns.scatterplot(x=self.train_predict_inv.flatten().tolist(),y=self.resid)
        plt.title("Homocedasticity for Residuals")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")        

        # Independence
        plt.subplot(2,2,4)
        sns.lineplot(x=np.arange(1,self.train_predict.shape[0]+1),y=self.resid)
        plt.axhline(np.mean(self.resid),color="red")
        plt.title("Independence for Residuals")
        plt.xlabel("Order")
        plt.ylabel("Residuals")                
        plt.suptitle("Assumptions for Model")
        plt.show()
        return fig

        
    



selected_stock = ["AAPL"]
def load_data(ticker):
    data = yf.download(ticker, "2015-01-01", "2023-01-01")
    return data
data = load_data(selected_stock)
data

model = Forecast_Model(data)
model.Model(sizeTrain_Proportion=0.85)
model.Prediction(Days_Prediction=5)
model.Plot()
model.new_df
model.x_predict
model.ts