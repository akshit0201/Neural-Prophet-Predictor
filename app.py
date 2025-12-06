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

    print(f"Processing {ticker}...")

    try:
        # 1. Get Data
        data = yf.download(ticker, period="3y", interval="1d", progress=False)

        if data.empty:
            return f"❌ Could not find data for ticker '{ticker}'. Please check the symbol.", None, None
        
        # Flatten MultiIndex if present
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
                close_col = [c for c in df.columns if "Close" in c][0]
                df = df[[close_col]].reset_index()
        else:
            df = data[['Close']].reset_index()

        # Rename for NeuralProphet
        df.columns = ['ds', 'y']
        df['ds'] = df['ds'].dt.tz_localize(None)

        if len(df) < 100:
            return f"❌ Not enough historical data found for {ticker} (Need > 100 days).", None, None

        # 2. Train Model
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

        # Generate Verdict & Colors
        if roi > 10: 
            verdict = "STRONG BUY 🚀"
            color = "#10B981" # Green
            bg_color = "#D1FAE5"
        elif roi > 2: 
            verdict = "BUY 🟢"
            color = "#10B981" # Green
            bg_color = "#D1FAE5"
        elif roi > -5: 
            verdict = "HOLD 🟡"
            color = "#F59E0B" # Yellow
            bg_color = "#FEF3C7"
        else: 
            verdict = "SELL 🔴"
            color = "#EF4444" # Red
            bg_color = "#FEE2E2"

        # 5. Format Output HTML (Pretty Dashboard)
        # Using inline CSS to ensure it looks good in Gradio
        html_report = f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 20px; background-color: {bg_color}; color: #1F2937; text-align: center; margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 1.5rem; text-transform: uppercase; color: {color};">{verdict}</h2>
            <p style="margin-top: 5px; font-size: 0.9rem; opacity: 0.8;">Forecast Horizon: 90 Days</p>
            
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Current</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">{current_price:.2f}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Target</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">{predicted_price:.2f}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">ROI</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{roi:+.2f}%</div>
                </div>
            </div>
        </div>
        """

        # 6. Generate Plots
        fig_forecast = m.plot(forecast)
        fig_forecast.update_layout(title_text="Price Forecast (Blue = Prediction)", title_x=0.5)
        
        fig_components = m.plot_components(forecast)
        fig_components.update_layout(title_text="Seasonality & Trend Analysis", title_x=0.5)

        return html_report, fig_forecast, fig_components

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h3 style='color: red'>❌ Error: {str(e)}</h3>", None, None

# --- STEP 3: GRADIO INTERFACE ---

# Custom CSS for a cleaner look
custom_css = """
.container { max-width: 900px; margin: auto; }
.footer { text-align: center; font-size: 0.8em; margin-top: 20px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            # 🔮 NeuralProphet Stock Predictor
            **AI-Powered 90-Day Price Forecasts**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                ticker_input = gr.Textbox(
                    label="Stock Ticker", 
                    placeholder="e.g. AZN.L, AAPL, TSLA", 
                    value="AZN.L",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("🚀 Analyze", variant="primary")

        # HTML Result Dashboard
        result_html = gr.HTML(label="Analysis Results")
        
        with gr.Row():
            plot1 = gr.Plot(label="Forecast")
            plot2 = gr.Plot(label="Seasonality")

        with gr.Accordion("ℹ️ Disclaimer & Info", open=False):
            gr.Markdown("""
            **How it works:** This app downloads 3 years of daily data and trains a NeuralProphet model on-the-fly. 
            It detects yearly and weekly seasonality to project price action 90 days out.
            
            **Disclaimer:** This tool is for educational purposes only. It is not financial advice. 
            AI models can hallucinate trends. Always do your own research.
            """)
            
        gr.Examples(
            examples=["AZN.L", "AAPL", "NVDA", "TSCO.L", "BTC-USD"],
            inputs=ticker_input
        )

    submit_btn.click(
        fn=predict_stock, 
        inputs=ticker_input, 
        outputs=[result_html, plot1, plot2]
    )

if __name__ == "__main__":
    demo.launch()