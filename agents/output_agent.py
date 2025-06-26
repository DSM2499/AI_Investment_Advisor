import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"),
                base_url = "https://api.deepseek.com")

class OutputAgent:
    def __init__(self, forecast_df: pd.DataFrame, insights: str):
        self.forecast_df = forecast_df
        self.insights = insights

    def _summarize_forecast(self):
        start_price = self.forecast_df["Predicted"].iloc[0]
        end_price = self.forecast_df["Predicted"].iloc[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        min_price = self.forecast_df["Predicted"].min()
        max_price = self.forecast_df["Predicted"].max()

        summary = {
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "change_pct": round(change_pct, 2),
            "min_price": round(min_price, 2),
            "max_price": round(max_price, 2)
        }
        return summary
    
    def _generate_llm_report(self, forecast_summary):
        trend_direction = (
            "increase" if forecast_summary["change_pct"] > 2 else
            "decrease" if forecast_summary["change_pct"] < -2 else
            "stable"
        )

        prompt = f"""
You are a financial assistant who explains stock predictions to regular people with no finance background.
Use this information to write a short, friendly, and clear investment summary:

- Forecasted price starts at ${forecast_summary['start_price']} and ends at ${forecast_summary['end_price']}
- This is a {forecast_summary['change_pct']}% {trend_direction} over 3 months
- Price range: ${forecast_summary['min_price']} to ${forecast_summary['max_price']}

Write in a tone that's like a smart friend giving financial advice. End with a simple recommendation (Buy, Sell, or Hold) and why.
Do not use finance jargon like 'volatility', 'RMSE', or 'regression'.
No Markdown formatting.
"""
        
        try:
            response = client.chat.completions.create(
                model = "deepseek-chat",
                messages = [{"role": "user", "content": prompt}],
                temperature = 0.6,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating report: {e}")
            return None
        
    def generate_report(self):
        summary = self._summarize_forecast()
        return self._generate_llm_report(summary)