import os

# Fix permission error by setting Streamlit config dir to home
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(os.path.expanduser("~"), ".streamlit")
os.makedirs(os.environ["STREAMLIT_CONFIG_DIR"], exist_ok=True)

import streamlit as st

# Disable usage stats (good for HF Spaces stability)
st.set_option("browser.gatherUsageStats", False)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------ Load Data ------------------------------
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# ------------------------------ Logo and Title ------------------------------
logo_url = "https://i.imgur.com/oDM4ECC.jpeg"
col1, col2 = st.columns([1, 8])
with col1:
    st.image(logo_url, use_container_width=True)
with col2:
    st.title("Chocolate Sales Forecast (Optimized ARIMA)")

# ------------------------------ Train/Test Split and Fit ------------------------------
train = df.iloc[:-52]
test = df.iloc[-52:]
order = (2, 0, 2)

with st.spinner(f"Training ARIMA{order}..."):
    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

# ------------------------------ Forecasts ------------------------------
forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()
forecast_dates = pd.date_range(start="2025-01-05", periods=52, freq='W-SUN')
forecast.index = forecast_dates
conf_int.index = forecast_dates
forecast_rounded = forecast.round(2)
conf_int_rounded = conf_int.round(2)

# ------------------------------ Tabs ------------------------------
tabs = st.tabs([
    "2025 Forecast & Summary",
    "2024 Model Evaluation",
    "Residual Diagnostics",
    "Historical Sales Lookup"
])

# ------------------------------ Tab 1: Forecast & Summary ------------------------------
with tabs[0]:
    st.subheader("Forecasted Chocolate Sales for 2025")

    # Plotly Forecast Chart
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_rounded.index,
        y=forecast_rounded,
        mode="lines",
        name="Forecasted Sales",
        line=dict(color="blue")
    ))
    fig_forecast.add_trace(go.Scatter(
        x=list(forecast_rounded.index) + list(forecast_rounded.index[::-1]),
        y=list(conf_int_rounded.iloc[:, 0]) + list(conf_int_rounded.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(0,0,255,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="90% Confidence Interval"
    ))
    fig_forecast.update_layout(
        title="Projected Chocolate Sales (2025)",
        xaxis_title="Week",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Week Selector
    st.subheader("Select a Week in 2025")
    selected_date = st.date_input(
        "Choose a forecast week:",
        min_value=forecast_rounded.index.min().date(),
        max_value=forecast_rounded.index.max().date(),
        value=forecast_rounded.index.min().date(),
        key="forecast_date"
    )
    selected_date = pd.to_datetime(selected_date)

    if selected_date not in forecast_rounded.index:
        st.warning("Please select a valid forecast week in 2025.")
    else:
        selected_forecast = forecast_rounded[selected_date]
        selected_ci = conf_int_rounded.loc[selected_date]
        st.metric("Forecasted Sales", f"{selected_forecast:.2f}")
        st.write(f"90% Confidence Interval: **[{selected_ci[0]:.2f}, {selected_ci[1]:.2f}]**")

    # Summary Metrics
    st.subheader("2025 Forecast Summary")
    total_sales = forecast_rounded.sum()
    avg_sales = forecast_rounded.mean()
    min_sales = forecast_rounded.min()
    max_sales = forecast_rounded.max()
    min_week = forecast_rounded.idxmin().date()
    max_week = forecast_rounded.idxmax().date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Forecast Sales", f"{total_sales:,.2f}")
    col2.metric("Average Weekly Sales", f"{avg_sales:.2f}")
    col3.metric("Min Weekly Sales", f"{min_sales:.2f}", f"Week of {min_week}")
    col4.metric("Max Weekly Sales", f"{max_sales:.2f}", f"Week of {max_week}")

    # Download CSV
    download_df = pd.DataFrame({
        "date": forecast_rounded.index,
        "forecasted_sales": forecast_rounded.values,
        "ci_lower_90": conf_int_rounded.iloc[:, 0].values,
        "ci_upper_90": conf_int_rounded.iloc[:, 1].values
    }).set_index("date")

    csv = download_df.to_csv().encode('utf-8')
    st.download_button("Download 2025 Forecast as CSV", csv, "chocolate_sales_forecast_2025.csv", "text/csv")

# ------------------------------ Tab 2: 2024 Evaluation ------------------------------
with tabs[1]:
    st.subheader("Model Performance on 2024 Actual Data")

    test_forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)
    test_forecast = test_forecast_result.predicted_mean
    test_conf_int = test_forecast_result.conf_int()
    test_forecast.index = test.index
    test_forecast_rounded = test_forecast.round(2)
    test_conf_int_rounded = test_conf_int.round(2)

    r2 = r2_score(test["sales"], test_forecast_rounded)
    mse = mean_squared_error(test["sales"], test_forecast_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["sales"], test_forecast_rounded)
    mape = np.mean(np.abs((test["sales"] - test_forecast_rounded) / test["sales"])) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("R²", f"{r2:.4f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("MAE", f"{mae:.2f}")
    col5.metric("MAPE (%)", f"{mape:.2f}")

    # Show actual vs predicted plot
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=test.index, y=test["sales"], mode="lines", name="Actual Sales"))
    fig_eval.add_trace(go.Scatter(x=test.index, y=test_forecast_rounded, mode="lines", name="Predicted Sales"))
    fig_eval.update_layout(title="2024 Actual vs Predicted Sales", xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(fig_eval, use_container_width=True)

# ------------------------------ Tab 3: Residual Diagnostics ------------------------------
with tabs[2]:
    st.subheader("Residual Diagnostics")

    residuals = test["sales"] - test_forecast_rounded

    # Residual plot
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=test.index, y=residuals, mode="markers+lines", name="Residuals"))
    fig_residual.update_layout(title="Residuals Over Time", xaxis_title="Date", yaxis_title="Residual")
    st.plotly_chart(fig_residual, use_container_width=True)

    # Histogram and Q-Q plot with matplotlib (show inline)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(residuals, bins=20, color="skyblue", edgecolor="black")
    axes[0].set_title("Residuals Histogram")

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")

    st.pyplot(fig)

# ------------------------------ Tab 4: Historical Sales Lookup ------------------------------
with tabs[3]:
    st.subheader("Lookup Historical Weekly Sales")

    lookup_date = st.date_input(
        "Select a date to view sales:",
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        value=df.index.max().date(),
        key="lookup_date"
    )
    lookup_date = pd.to_datetime(lookup_date)

    if lookup_date not in df.index:
        st.warning("Date not found in historical sales data.")
    else:
        sales_value = df.loc[lookup_date, "sales"]
        st.metric(f"Sales on {lookup_date.date()}", f"${sales_value:.2f}")
