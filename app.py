import gradio as gr
import torch
import logging
import warnings
import os
import yfinance as yf
import pandas as pd
from neuralprophet import NeuralProphet
import plotly.graph_objs as go

# --- STEP 1: CONFIGURATION & PATCHES ---

# Suppress messy logs
logging.getLogger("neuralprophet").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fix for PyTorch 2.6+ security check
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# --- STEP 2: PREDICTION LOGIC ---

def predict_stock(ticker):
    """
    Takes a ticker symbol, trains a NeuralProphet model, 
    and returns a textual report and two Plotly figures.
    """
    ticker = ticker.strip().upper()
    
    if not ticker:
        return "⚠️ Please enter a ticker symbol.", None, None

    # Status update for the logs
    print(f"Processing {ticker}...")

    try:
        # 1. Get Data
        data = yf.download(ticker, period="3y", interval="1d", progress=False)

        # Handle cases where yfinance returns empty dataframe or multi-index columns
        if data.empty:
            return f"❌ Could not find data for ticker '{ticker}'. Please check the symbol.", None, None
        
        # Flatten MultiIndex if present (yfinance update quirk)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                # Attempt to extract just the Close column for the specific ticker
                df = data.xs(ticker, axis=1, level=1)
                if 'Close' in df.columns:
                    df = df[['Close']].reset_index()
                else:
                    df = data['Close'].reset_index()
            except:
                # Brute force flatten
                df = data.copy()
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                # Look for a column containing "Close"
                close_col = [c for c in df.columns if "Close" in c][0]
                df = df[[close_col]].reset_index()
        else:
            df = data[['Close']].reset_index()

        # Rename for NeuralProphet
        df.columns = ['ds', 'y']
        
        # Ensure dates are timezone-naive
        df['ds'] = df['ds'].dt.tz_localize(None)

        if len(df) < 100:
            return f"❌ Not enough historical data found for {ticker} (Need > 100 days).", None, None

        # 2. Train Model
        # FIX: Removed 'trainer_config' to prevent PyTorch Lightning crash
        m = NeuralProphet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            learning_rate=0.01
        )

        m.fit(df, freq="D")

        # 3. Predict 90 Days out
        future = m.make_future_dataframe(df, periods=90)
        forecast = m.predict(future)

        # 4. Extract Metrics
        current_price = df['y'].iloc[-1]
        predicted_price = forecast['yhat1'].iloc[-1]

        # Calculate ROI
        roi = ((predicted_price - current_price) / current_price) * 100

        # Generate Verdict
        if roi > 10: verdict = "STRONG BUY 🟢"
        elif roi > 2: verdict = "BUY 🟢"
        elif roi > -5: verdict = "HOLD 🟡"
        else: verdict = "SELL 🔴"

        # 5. formatting Output Text
        report = f"""
### 📊 Analysis Report: {ticker}
**Current Price:** {current_price:.2f}
**90-Day Target:** {predicted_price:.2f}
**Projected ROI:** {roi:.2f}%
**Verdict:** {verdict}

*Disclaimer: This is an AI-generated forecast based on historical trends. Not financial advice.*
        """

        # 6. Generate Plots
        # Note: We rely on standard m.plot() which returns a plotly figure
        fig_forecast = m.plot(forecast)
        fig_components = m.plot_components(forecast)

        return report, fig_forecast, fig_components

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ An error occurred while processing {ticker}: {str(e)}", None, None

# --- STEP 3: GRADIO INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 NeuralProphet Stock Predictor")
    gr.Markdown("Enter a stock ticker (e.g., `AAPL`, `TSLA`, `AZN.L`) to generate a 90-day forecast.")
    
    with gr.Row():
        ticker_input = gr.Textbox(label="Ticker Symbol", placeholder="e.g. AZN.L", value="AZN.L")
        submit_btn = gr.Button("Analyze Stock", variant="primary")
    
    result_text = gr.Markdown(label="Verdict")
    
    with gr.Row():
        plot1 = gr.Plot(label="Price Forecast")
        plot2 = gr.Plot(label="Seasonality Components")

    submit_btn.click(
        fn=predict_stock, 
        inputs=ticker_input, 
        outputs=[result_text, plot1, plot2]
    )

if __name__ == "__main__":
    demo.launch()