import streamlit as st
from datetime import date
import datetime
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import PortfolioOptimisation as PO
import ForecastModel as FM


st.title('Stock Portfolio App')

# Sidebar
st.sidebar.subheader('Dates')
start_date = st.sidebar.date_input("Start date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

stocks = ("AAPL","MSFT","WALMEX.MX","ABNB","WMT","AMD","BIMBOA.MX","^GSPC")
selected_stock = st.multiselect('Select Stocks',stocks)

def AnnualReturns_Risk():
	df = pd.DataFrame(columns=["Annual Returns %","Annual Risk %","Sharpe Ratio"])
	for stock in selected_stock:
		Returns = data["Adj Close"][stock].pct_change().dropna()
		Annual_Returns = (Returns.mean() * 252) * 100
		Annual_Risk = ((Returns.var() * 252) ** (1/2)) * 100
		Sharpe = Annual_Returns / Annual_Risk
		df = df.append({"Annual Returns %":Annual_Returns,"Annual Risk %":Annual_Risk,"Sharpe Ratio":Sharpe},ignore_index = True)
	df.index = [selected_stock]
	return df

def AnnualReturns_Risk_Unique():
	df = pd.DataFrame(columns=["Annual Returns %","Annual Risk %","Sharpe Ratio"])
	Returns = data["Adj Close"].pct_change().dropna()
	Annual_Returns = (Returns.mean() * 252) * 100
	Annual_Risk = ((Returns.var() * 252) ** (1/2)) * 100
	Sharpe = Annual_Returns / Annual_Risk
	df = df.append({"Annual Returns %":Annual_Returns,"Annual Risk %":Annual_Risk,"Sharpe Ratio":Sharpe},ignore_index = True)
	df.index = [selected_stock]
	return df


if selected_stock:
	if len(selected_stock) > 1:
		@st.cache
		def load_data(ticker):
			data = yf.download(ticker, start_date, end_date)
			return data
		data = load_data(selected_stock)
		
		st.subheader("Statistics")
		cols = st.columns(len(selected_stock))
		for col,stock in zip(cols,selected_stock):
			col.write(data["Adj Close"][stock].describe())

		col1,col2,col3 = st.columns([1,3,1])
		with col1:
			st.write(" ")
		with col2:
			st.write(AnnualReturns_Risk())
		with col3:
			st.write(" ")

		st.subheader("Statics Visualization")

		# Histogram
		fig = px.histogram(data["Adj Close"],title="Histogram of Close Price",labels={"value":"Adj Close $/Pesos"})
		st.plotly_chart(fig)
		# Pairplot
		fig = px.scatter_matrix(data["Adj Close"],title="Scatter Matrix")
		st.plotly_chart(fig)
		# Time Serie
		st.line_chart(data["Adj Close"])
		# Heatmap
		df = AnnualReturns_Risk()["Sharpe Ratio"]
		df1 = pd.DataFrame(columns=selected_stock)
		for num,stock in enumerate(selected_stock):
			df1 = df1.append({stock:df.values[num]},ignore_index=True)
		df1.index = selected_stock
		df1.fillna(0,inplace=True)
		fig = px.imshow(df1,text_auto=True,color_continuous_scale="blues",title="Heat Map for Sharpe Ratio")
		st.plotly_chart(fig)
		Portfolio = PO.Portfolio(data)
		Portfolio.Mean_Var_Matrix()
		st.plotly_chart(Portfolio.display_simulated_ef_with_random(num_portfolios=10000))


	else:
		@st.cache
		def load_data(ticker):
			data = yf.download(ticker, start_date, end_date)
			return data

		data = load_data(selected_stock)

		st.subheader("Statistics")
		st.write(data.describe())
		col1,col2,col3 = st.columns([1,3,1])
		with col1:
			st.write(" ")
		with col2:
			st.write(AnnualReturns_Risk_Unique())
		with col3:
			st.write(" ")

		st.subheader("Statics Visualization")
		# Histogram
		fig = px.histogram(data["Adj Close"],title="Histogram of Close Price",labels={"value":"Adj Close $/Pesos"})
		st.plotly_chart(fig)

		st.subheader("AI Forecast Modeling")
		st.write("\n")
		st.write("Stock:",selected_stock[0])
		Model = FM.Forecast_Model(data)
		Model.Model(sizeTrain_Proportion=0.85)
		Days_Prediction = st.slider('Days of Prediction:',1,20)
		Model.Prediction(Days_Prediction)
		Model.Plot()
		st.subheader("Model - Assumptions")
		st.pyplot(Model.Assumptions())

