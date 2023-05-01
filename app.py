# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:51:29 2023

@author: USER
"""
#pip install streamlit --upgrade





import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pickle

#load the model
model_fit = pickle.load(open('AR_model_final.pkl','rb'))


# Define the forecasting function
def predict_forecast(date):
    # Preprocess the input date
    date = dt.datetime.strptime(date, '%Y-%m-%d')
    
    # Generate the forecast for the next month
    forecast = model.predict(start=date, end=date+pd.offsets.MonthEnd(1))
    
    # Format the forecast result
    result = f"Forecast for {date.strftime('%B %Y')}: {forecast[0]:.2f} units"
    
    return result

# Create the Streamlit app
st.title("AR Model Forecasting")

# Add an image
st.image('C:\\Users\\USER\\Desktop\\project 99\\model deployment by streamlit\\pallets.jpg', use_column_width=True)

# Create a date input field
date_input = st.date_input("Select a date for forecasting", dt.date.today())

# Add a button to trigger the forecasting function
if st.button("Forecast"):
    result = predict_forecast(str(date_input))
    st.write(result)

from typing import Protocol

class MyProtocol(Protocol):
    def my_method(self) -> str:
        pass

class MyClass:
    def my_method(self) -> str:
        return "Hello, World!"

def my_function(obj: MyProtocol) -> None:
    print(obj.my_method())

my_obj = MyClass()
my_function(my_obj)


    
    
    