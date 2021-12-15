#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import yfinance as yf
import streamlit as st
from PIL import Image


######################
# Page Title and logo
######################

image = Image.open('stock.jpg')

st.image(image, use_column_width=True)


st.write("""
# Simple Stock Price App 
Shown are the stock closing price and volume 
""")

######################
# Input Text Box
######################


st.subheader('Enter the Company Name')

tickerSymbol_input = "GOOGL"


tickerSymbol = st.text_area("tickerSymbol_ input", tickerSymbol_input, height=25)

st.write("""
***
""")



tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker

st.subheader('Enter starting date')

start_date= "2010-5-31"

tickerSymbol = st.text_area("start_date", start_date, height=25)

st.write("""
***
""")

st.subheader('Enter Ending date')

end_date= "2021-12-15"


tickerSymbol = st.text_area("end_date", end_date, height=25)

st.write("""
***
""")
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
# Open	High	Low	Close	Volume	Dividends	Stock Splits

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)

