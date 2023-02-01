import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import yfinance as yf
import datetime
import seaborn as sns
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.offline import plot
import streamlit as st

selected_stock = ["AAPL","MSFT"]
def load_data(ticker):
    data = yf.download(ticker, "2022-01-01", "2023-01-01")
    return data
data = load_data(selected_stock)


class Portfolio:
    def __init__(self,df):
        self.df = df["Adj Close"]
    
    def Mean_Var_Matrix(self):
        self.mean_returns = self.df.pct_change().mean().dropna()
        self.cov_matrix = self.df.pct_change().cov()
    
    def portfolio_annualised_performance(self,weights):
        self.returns = np.sum(self.mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        return self.returns, std
    
    def random_portfolios(self,num_portfolios,risk_free_rate=0):
        self.results = np.zeros((3,num_portfolios))
        self.weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(len(self.mean_returns))
            weights /= np.sum(weights)
            self.weights_record.append(weights)
            returns,std = self.portfolio_annualised_performance(weights)
            print (returns)
            print (weights)
            self.results[0,i] = std
            self.results[1,i] = returns
            self.results[2,i] = (returns - risk_free_rate) / std
        return self.results, self.weights_record
    
    def display_simulated_ef_with_random(self, num_portfolios, risk_free_rate=0):
        results, weights = self.random_portfolios(num_portfolios,risk_free_rate)
        
        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
        max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=self.mean_returns.index,columns=['Proporción'])
        max_sharpe_allocation["Proporción"] = [round(i*100,2) for i in max_sharpe_allocation["Proporción"]]
        max_sharpe_allocation = max_sharpe_allocation.T
        
        min_vol_idx = np.argmin(results[0])
        sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
        min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=self.mean_returns.index,columns=['Proporción'])
        min_vol_allocation["Proporción"] = [round(i*100,2)for i in min_vol_allocation["Proporción"]]
        min_vol_allocation = min_vol_allocation.T
        
        #print ("-"*80)
        st.write("-"*80)
        #print ("Sharpe Ratio Máximo\n")
        st.write("Sharpe Ratio Máximo\n")
        #print ("Rendimientos Anuales:", round(rp,2))
        st.write("Rendimientos Anuales:", round(rp*100,2))
        #print ("Riesgo Anual:", round(sdp,2))
        st.write("Riesgo Anual:", round(sdp*100,2))
        #print ("\n")
        st.write("\n")
        #print (max_sharpe_allocation)
        st.write(max_sharpe_allocation)
        #print ("-"*80)
        st.write("-"*80)
        #print ("Riesgo Mínimo \n")
        st.write("Riesgo Mínimo \n")
        #print ("Rendimientos Anuales:", round(rp_min,2))
        st.write("Rendimientos Anuales:", round(rp_min*100,2))
        #print ("Riesgo Anual:", round(sdp_min,2))
        st.write("Riesgo Anual:", round(sdp_min*100,2))
        #print ("\n")
        st.write("\n")
        #print (min_vol_allocation)
        st.write(min_vol_allocation)
        st.write("-"*80)
        
        MaxSharpeRatio = go.Scatter(
            name='Maximium Sharpe Ratio',
            mode='markers',
            x=[sdp],
            y=[rp],
            marker=dict(color='red',size=14,line=dict(width=3, color='black'))
            )
        
        #Min Vol
        MinVol = go.Scatter(
            name='Mininium Volatility',
            mode='markers',
            x=[sdp_min],
            y=[rp_min],
            marker=dict(color='green',size=14,line=dict(width=3, color='black')))
        
        EF_curve = go.Scatter(
            name='Efficient Frontier',
            mode="markers",
            x=results[0,:],
            y=results[1,:],
            marker=dict(color=results[1,:],colorscale="Viridis",size=6,colorbar=dict(title="Sharp Ratio",thickness=20),line=dict(width=1, color='black')) #color="#0DF9FF",
            )
        
        data = [ EF_curve,MaxSharpeRatio, MinVol]
        layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='white',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    
        fig = go.Figure(data=data, layout=layout)
        
        '''
        fig=plt.figure(figsize=(10, 7))
        plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
        plt.colorbar()
        plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Sharpe Ratio Máximo')
        plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Riesgo Mínimo')
        plt.title('Optimización de Portfolio')
        plt.xlabel('Riesgo Anual')
        plt.ylabel('Rendimiento Anual')
        plt.legend(labelspacing=0.8)'''
        return fig


a = Portfolio(data)
a.Mean_Var_Matrix()
weights = np.random.random(len(a.mean_returns))
weights /= np.sum(weights)
np.sum(a.mean_returns*weights)*252
a.portfolio_annualised_performance(weights)
a.random_portfolios(num_portfolios=10,risk_free_rate=0)
a.display_simulated_ef_with_random(100)