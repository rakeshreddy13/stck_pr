import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

# Title of the web app
st.title('📈 Stock Market Prediction with ARIMA/SARIMA')

# Input field for stock ticker
ticker = st.text_input('Enter Stock Ticker Symbol:', 'AAPL').strip().upper()

if not ticker:
    st.error("Please enter a valid stock ticker symbol!")
    st.stop()  # Stop execution if no valid ticker is provided

# Fetch historical stock data using yfinance
st.write(f"Fetching data for {ticker}...")
try:
    data = yf.download(ticker, start="2010-01-01", end="2022-12-31")
    if data.empty:
        st.error("No data found for this ticker! Please enter a valid stock symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Show first few rows of the data
st.write(data.head())

# Plot the stock's closing price
st.subheader(f"📊 Stock Closing Price of {ticker}")
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Actual Closing Price', color='blue')
ax.set_title(f'{ticker} Closing Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)

# ARIMA Model Configuration
st.subheader('⚙️ ARIMA/SARIMA Configuration')

# Input fields for ARIMA parameters (p, d, q)
p = st.number_input('p (AR term):', min_value=0, max_value=10, value=5)
d = st.number_input('d (Differencing):', min_value=0, max_value=3, value=1)
q = st.number_input('q (MA term):', min_value=0, max_value=10, value=5)

# SARIMA Parameters (seasonal p, d, q, s)
seasonal_p = st.number_input('Seasonal p:', min_value=0, max_value=10, value=1)
seasonal_d = st.number_input('Seasonal d:', min_value=0, max_value=3, value=1)
seasonal_q = st.number_input('Seasonal q:', min_value=0, max_value=10, value=1)
seasonal_s = st.number_input('Seasonal period (s):', min_value=1, max_value=12, value=5)

# Fix: Ensure seasonal_p ≠ p to avoid SARIMAX conflicts
if seasonal_p == p:
    seasonal_p = max(0, p - 1)  # Adjust seasonal p if it's equal to p

# Fit the ARIMA model
st.write(f"Fitting ARIMA model with parameters: (p={p}, d={d}, q={q}) and seasonal (P={seasonal_p}, D={seasonal_d}, Q={seasonal_q}, S={seasonal_s})")

try:
    model = sm.tsa.statespace.SARIMAX(
        data['Close'],
        order=(p, d, q),
        seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_s),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model
    results = model.fit()

    # Show model summary
    st.subheader("📜 Model Summary")
    st.text(results.summary())

    # Make Predictions
    forecast_steps = st.slider('📅 Select number of days to forecast:', 1, 365, 30)

    # Forecast using get_forecast() for confidence intervals
    forecast_result = results.get_forecast(steps=forecast_steps)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Prepare forecasted data
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
    forecast_df = pd.DataFrame(forecast.values, index=forecast_index, columns=['Prediction'])

    # Show prediction results
    st.subheader(f"🔮 Forecasted Stock Prices for the next {forecast_steps} days")
    st.write(forecast_df)

    # Plot forecasted stock prices
    st.subheader('📉 Stock Price Forecast')
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Actual Price', color='blue')
    ax.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='dashed')
    ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    ax.set_title(f'Stock Price Prediction for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ Error in model fitting: {e}")