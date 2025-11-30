---
title: Neural Prophet Stock Predictor
emoji: 🔮
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
python_version: 3.10
pinned: false
---

# NeuralProphet Stock Predictor 📈

## Project Goal
This project uses **NeuralProphet**, a PyTorch-based hybrid forecasting framework (inspired by Facebook Prophet and Neural Networks), to predict the **90-day forward price** for any valid stock ticker found on Yahoo Finance. 

Unlike simple linear regressions, this model accounts for time-series specific components such as:
- **Trend Changes:** Detecting shifts in the stock's growth trajectory.
- **Seasonality:** analyzing Yearly and Weekly patterns (e.g., does the stock typically rise in January?).

## How to Use

### Using the UI
1. Enter a valid Ticker Symbol in the textbox (e.g., `AAPL` for Apple, `AZN.L` for AstraZeneca UK).
2. Click **"Analyze Stock"**.
3. Wait approximately 10-30 seconds for the model to train on-the-fly.
4. View the **ROI Verdict**, the **Forecast Chart**, and the **Seasonality breakdown**.

### Interpretation
- **Blue Line (Forecast):** The predicted price path.
- **Yearly Seasonality:** Shows which months are historically bullish or bearish for this specific asset.
- **Verdict:**
    - 🟢 **STRONG BUY:** ROI > 10%
    - 🟢 **BUY:** ROI > 2%
    - 🟡 **HOLD:** ROI > -5%
    - 🔴 **SELL:** ROI <= -5%

## Model Methodology

This application trains a *fresh* model every time you request a ticker. It does not use pre-trained weights because stock data changes daily.

1.  **Data Fetching:** Downloads 3 years of daily historical data via `yfinance`.
2.  **Preprocessing:** Cleans data and handles timezone formatting.
3.  **Training:** Fits a NeuralProphet model with:
    *   Yearly Seasonality: Enabled
    *   Weekly Seasonality: Enabled
    *   Daily Seasonality: Disabled (Noise reduction)
    *   Learning Rate: 0.01
4.  **Forecasting:** Projects 90 days into the future.

## Tech Stack
- **NeuralProphet:** For time-series forecasting.
- **YFinance:** For live market data.
- **Gradio:** For the web interface.
- **Plotly:** For interactive charting.

*Disclaimer: Stock market predictions are inherently uncertain. Do not trade based solely on these results.*
