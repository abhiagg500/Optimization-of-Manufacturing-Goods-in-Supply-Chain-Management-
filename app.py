# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:39:50 2023

@author: USER
"""

from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask("Optimization in supply chain management")

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    input_data = request.form['input_data']
    
    # Convert input data to pandas dataframe
    df = pd.read_csv(input_data)
    
    # Train ARIMA model
    model = ARIMA(df['QUANTITY'], order=(1,0,0))
    model_fit = model.fit()
    
    # Generate predictions
    predictions = model_fit.predict(start=len(df), end=len(df)+12, dynamic=True)
    
    # Render template with predictions
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
