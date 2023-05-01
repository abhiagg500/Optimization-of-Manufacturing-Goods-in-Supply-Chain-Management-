import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
#from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Forecasting Pallets")

st.write("""
# Forecasting Pallets
""")

# Create file uploader for new data
file = st.file_uploader("Upload Excel", type=["xlsx"])

if file is not None:
    # Read uploaded CSV file
    df1 = pd.read_excel(file)
    
    # Rename columns if necessary
    df  = df1[['CREATE DATE', 'QUANTITY']]
    
    # Convert date column to datetime
    df['CREATE DATE'] = pd.to_datetime(df['CREATE DATE'])
    
    # Set date column as index
    df.set_index('CREATE DATE', inplace=True)
    
    # Resample the data to weekly frequency(W)
    df = df.resample('W').sum()
    
    # Fill in missing values
    df = df.replace(0, method='ffill')
    
    # Perform Augmented Dickey-Fuller test to check for stationarity
    from statsmodels.tsa.stattools import adfuller
    
    # Extract the column of interest
    data = df['QUANTITY']

    # Perform the Augmented Dickey-Fuller test
    result = adfuller(data)

    # Print the test statistic and p-value
    st.write('ADF Statistic: %f' % result[0])
    st.write('p-value: %f' % result[1])

    # Interpret the results
    if result[1] > 0.05:
        st.write('The data is non-stationary.')
    else:
        st.write('The data is stationary.')
    
    # Split data into training and testing sets
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Fit ARIMA model
    model = ARIMA(train['QUANTITY'], order=(3,0,1))
    model_fit = model.fit()

    # Make predictions
    pred_train = model_fit.predict(start=train.index[0], end=test.index[-1], dynamic=False)
    pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)

    # Make forecast for next 12 weeks
    forecast = model_fit.forecast(steps=12)
    forecast_values = pd.Series(forecast)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test['QUANTITY'], pred_test)
    st.write('Mean Absolute Percentage Error: {:.2f}%'.format(mape * 100))

    # Plot actual vs. predicted values
    plt.plot(df['QUANTITY'], label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    plt.title('Forecasting Pallets')
    st.pyplot()
    
    # Show forecast values
    st.write('Forecasted Values:')
    st.write(forecast_values)
