import streamlit as st
from agents.data_collector import DataCollectorAgent
from agents.model_runner import ModelRunnerAgent
from agents.predictor_agent import PredictorAgent
from agents.insight_agent import InsightAgent
from agents.output_agent import OutputAgent

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.set_page_config(page_title="Investment Forecast", layout="wide")
st.title("📈 Smart Stock Forecasting Assistant")

ticker = st.text_input("Enter a stock ticker (e.g. AAPL, TSLA, GOOGL)", "AAPL")

if st.button("🔍 Analyze Stock"):
    with st.spinner("Collecting historical data..."):
        collector = DataCollectorAgent()
        df = collector.get_stock_data(ticker, period="2y")

    if df.empty:
        st.error("Failed to fetch data. Check the ticker symbol.")
    else:
        st.success(f"Collected {len(df)} rows of historical data.")
        st.line_chart(df.set_index("Date")["Close"].tail(90))

        with st.spinner("Training models..."):
            runner = ModelRunnerAgent()
            result = runner.train_models(df)
            best_model = result["model"]
            model_name = result["model_name"]
            features = result.get("features", None)
            st.success(f"✅ Best model: {model_name.upper()} (RMSE: {result['error']:.2f})")

        with st.spinner("Forecasting next 3 months..."):
            predictor = PredictorAgent()
            forecast_df = predictor.predict_next_90_days(
                model_name=model_name,
                model=best_model,
                hist_df=df,
                features=features
            )
            st.line_chart(forecast_df.set_index("Date")["Predicted"])

        with st.spinner("Generating insights..."):
            insight_agent = InsightAgent(runner)
            insights = insight_agent.generate_insights(df)

        with st.spinner("Building final report..."):
            output_agent = OutputAgent(
                forecast_df=forecast_df,
                insights=insights
            )
            report = output_agent.generate_report()

        st.markdown("---")
        st.subheader("🧠 Model Insights")
        st.markdown(insights)

        st.subheader("📘 Investment Report")
        st.markdown(report)

        with st.expander("📉 Forecast Table"):
            st.dataframe(forecast_df.set_index("Date").style.format({"Predicted": "${:.2f}"}))