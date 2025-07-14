import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

# --- Page config ---
st.set_page_config(page_title="Chocolate Sales Forecast", layout="wide")

# --- Load data ---
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# --- Logo & title ---
logo_url = "https://i.imgur.com/oDM4ECC.jpeg"
col1, col2 = st.columns([1, 8])
with col1:
    st.image(logo_url, use_container_width=True)
with col2:
    st.title("Chocolate Sales Forecast (ARIMA)")

# --- Train/test split ---
train = df.iloc[:-52]
test  = df.iloc[-52:]
order = (2, 0, 2)

# --- Fit ARIMA ---
with st.spinner(f"Training ARIMA{order}..."):
    model     = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

# --- Forecast 2025 & 2024 ---
def make_forecast(model_fit, n_steps, start_date):
    fcst_res = model_fit.get_forecast(steps=n_steps, alpha=0.10)
    fcst     = fcst_res.predicted_mean.round(2)
    ci       = fcst_res.conf_int().round(2)
    dates    = pd.date_range(start=start_date, periods=n_steps, freq="W-SUN")
    fcst.index = dates
    ci.index   = dates
    return fcst, ci

forecast_2025, ci_2025 = make_forecast(model_fit, 52, start_date="2025-01-05")
forecast_2024, _      = make_forecast(model_fit, 52, start_date=test.index[0])

# --- Tabs ---
tabs = st.tabs([
    "2025 Forecast",
    "2024 Evaluation",
    "Residuals",
    "Historical Lookup"
])

# Tab 1: 2025 Forecast
with tabs[0]:
    st.subheader("2025 Weekly Sales Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_2025.index, y=forecast_2025, mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(
        x=list(forecast_2025.index)+list(forecast_2025.index[::-1]),
        y=list(ci_2025.iloc[:,0])+list(ci_2025.iloc[:,1][::-1]),
        fill="toself", fillcolor="rgba(0,0,255,0.2)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        name="90% CI"
    ))
    fig.update_layout(xaxis_title="Week", yaxis_title="Sales", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Select a Week")
    sel = st.date_input("Pick a 2025 Sunday:", min_value=forecast_2025.index.min().date(),
                        max_value=forecast_2025.index.max().date(), value=forecast_2025.index.min().date())
    sel = pd.to_datetime(sel)
    if sel in forecast_2025.index:
        st.metric("Forecast", f"${forecast_2025[sel]:.2f}")
        lo, hi = ci_2025.loc[sel]
        st.write(f"90% CI: [${lo:.2f}, ${hi:.2f}]")
    else:
        st.warning("Choose a Sunday in 2025")

    st.subheader("2025 Summary")
    total = forecast_2025.sum(); avg = forecast_2025.mean()
    mn, mx = forecast_2025.min(), forecast_2025.max()
    wmin, wmax = forecast_2025.idxmin().date(), forecast_2025.idxmax().date()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"${total:,.2f}")
    c2.metric("Average", f"${avg:.2f}")
    c3.metric("Min", f"${mn:.2f}", f"Week of {wmin}")
    c4.metric("Max", f"${mx:.2f}", f"Week of {wmax}")

    # download CSV
    out = pd.DataFrame({
        "forecast": forecast_2025,
        "ci_lower": ci_2025.iloc[:, 0],
        "ci_upper": ci_2025.iloc[:, 1]
    })
    st.download_button("Download CSV", out.to_csv().encode(), "forecast_2025.csv")

# Tab 2: 2024 Evaluation
with tabs[1]:
    st.subheader("2024 Model Evaluation")
    y_actual = test["sales"]
    y_pred   = forecast_2024.reindex(test.index)
    r2    = r2_score(y_actual, y_pred)
    rmse  = mean_squared_error(y_actual, y_pred, squared=False)
    mae   = mean_absolute_error(y_actual, y_pred)
    mape  = np.mean(np.abs((y_actual - y_pred) / y_actual))*100
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("R²", f"{r2:.3f}")
    d2.metric("RMSE", f"${rmse:.2f}")
    d3.metric("MAE", f"${mae:.2f}")
    d4.metric("MAPE", f"{mape:.2f}%")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test.index, y=y_actual, mode="lines", name="Actual"))
    fig2.add_trace(go.Scatter(x=test.index, y=y_pred, mode="lines", name="Forecast"))
    fig2.update_layout(xaxis_title="Week", yaxis_title="Sales")
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3: Residuals
with tabs[2]:
    st.subheader("Residual Diagnostics")
    res = model_fit.resid
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=res.index, y=res, mode="lines", name="Residuals"))
    st.plotly_chart(fig3, use_container_width=True)

    fig4, ax4 = plt.subplots()
    ax4.hist(res, bins=20, edgecolor="k")
    st.pyplot(fig4)
    fig5, ax5 = plt.subplots()
    stats.probplot(res, dist="norm", plot=ax5)
    st.pyplot(fig5)
    fig6, ax6 = plt.subplots()
    plot_acf(res, ax=ax6, lags=40)
    st.pyplot(fig6)

# Tab 4: Historical Lookup
with tabs[3]:
    st.subheader("Historical Weekly Sales")
    dd = st.date_input("Select week:", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.max().date())
    dd = pd.to_datetime(dd)
    if dd in df.index:
        st.metric("Sales", f"${df.at[dd,'sales']:.2f}")
    else:
        st.warning("Pick a Sunday in the data range")

# Footer
st.markdown("---")
st.markdown(
    "© 2024 The Forecast Company. All Rights Reserved.  "
    "[Email](mailto:theforecastcompany@gmail.com) | [Call](tel:8563040922)",
    unsafe_allow_html=True
)
