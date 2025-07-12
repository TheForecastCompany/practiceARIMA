import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import streamlit as st

st.title('Chocolate Shop Sales Forecasting')
st.write('This app uses an auto-selected ARIMA model to forecast weekly sales for a chocolate shop.')

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
    
    # Fit auto_arima model on train
    try:
        model = auto_arima(train['sales'], seasonal=True, m=52, trace=True, error_action='ignore', suppress_warnings=True)
        fitted_model = model.fit(train['sales'])
        
        # Predict test set
        predictions = fitted_model.forecast(steps=len(test))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((test['sales'] - predictions)**2))
        st.write(f'Root Mean Squared Error on test set: {rmse:.2f}')
        
        # Fit auto_arima on full data
        full_model = auto_arima(data['sales'], seasonal=True, m=52, trace=True, error_action='ignore', suppress_warnings=True)
        full_fitted_model = full_model.fit(data['sales'])
        
        # Forecast next 10 weeks with confidence intervals (95%)
        forecast_dict = full_fitted_model.predict(n_periods=10, return_conf_int=True, alpha=0.05)
        forecast = forecast_dict[0]
        conf_int = forecast_dict[1]
        
        # Allow user to select week
        week = st.slider('Select week number (1-10) for the upcoming period', 1, 10)
        pred_value = forecast.iloc[week-1]
        lower_ci = conf_int[week-1, 0]
        upper_ci = conf_int[week-1, 1]
        st.write(f'Predicted sales for week {week}: {pred_value:.2f} (95% CI: [{lower_ci:.2f}, {upper_ci:.2f}])')
        
        # Prepare data for plotting with confidence intervals
        forecast_dates = pd.date_range(start=data.index[-1], periods=11, freq='W')[1:]
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast,
            'lower_ci': conf_int[:, 0],
            'upper_ci': conf_int[:, 1]
        })
        
        # Plot historical data and forecast with confidence intervals
        st.write('Historical Sales:')
        st.line_chart(data['sales'])
        st.write('Forecast for the next 10 weeks with 95% Confidence Interval:')
        fig_data = pd.concat([data['sales'], forecast_df.set_index('date')['forecast']])
        st.line_chart(fig_data)
        st.line_chart(forecast_df.set_index('date')[['lower_ci', 'upper_ci']], color=['#ff9999', '#ff9999'], alpha=0.3)  # Shaded CI
    except Exception as e:
        st.error(f'Error fitting auto_arima model: {str(e)}')
else:
    st.write('Please upload a CSV file with columns "date" (in YYYY-MM-DD format) and "sales" (numeric).')
