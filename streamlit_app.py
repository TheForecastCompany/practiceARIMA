import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

st.title('Chocolate Shop Sales Forecasting')
st.write('This app uses a SARIMA(1,1,1)(1,0,1,52) model to forecast weekly sales for a chocolate shop.')

uploaded_file = st.file_uploader('Upload your sales data CSV file', type='csv')

if uploaded_file is not None:
    # Load CSV and explicitly convert 'date' column to datetime
    data = pd.read_csv(uploaded_file)
    try:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    except (ValueError, KeyError):
        st.error('The CSV must have a "date" column in a valid datetime format (e.g., YYYY-MM-DD).')
        st.stop()
    
    if len(data) != 156:
        st.error('The data must have exactly 156 weekly observations for 3 years.')
        st.stop()
    
    if 'sales' not in data.columns:
        st.error('The CSV must have a "sales" column with numeric values.')
        st.stop()
    
    # Split the data into train and test
    train = data.iloc[:104]
    test = data.iloc[104:]
    
    # Fit SARIMA model on train
    try:
        model = SARIMAX(train['sales'], order=(1,1,1), seasonal_order=(1,0,1,52))
        fitted_model = model.fit()
        
        # Predict test set
        predictions = fitted_model.forecast(steps=len(test))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((test['sales'] - predictions)**2))
        st.write(f'Root Mean Squared Error on test set: {rmse:.2f}')
        
        # Fit model on full data
        full_model = SARIMAX(data['sales'], order=(1,1,1), seasonal_order=(1,0,1,52))
        full_fitted_model = full_model.fit()
        
        # Forecast next 52 weeks
        forecast = full_fitted_model.forecast(steps=52)
        
        # Allow user to select week
        week = st.slider('Select week number (1-52) for the upcoming year', 1, 52)
        st.write(f'Predicted sales for week {week}: {forecast[week-1]:.2f}')
        
        # Plot historical data and forecast
        st.write('Historical Sales:')
        st.line_chart(data['sales'])
        st.write('Forecast for the next 52 weeks:')
        st.line_chart(forecast)
    except Exception as e:
        st.error(f'Error fitting SARIMA model: {str(e)}')
else:
    st.write('Please upload a CSV file with columns "date" (in YYYY-MM-DD format) and "sales" (numeric).')
