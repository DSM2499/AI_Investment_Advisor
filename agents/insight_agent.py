import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from agents.model_runner import ModelRunnerAgent

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"),
                base_url = "https://api.deepseek.com")

class InsightAgent:
    def __init__(self, model_runner: ModelRunnerAgent):
        self.model_runner = model_runner
    
    def generate_insights(self, df):
        results = self.model_runner.get_all_models()

        prompt = f"""
You are an expert financial advisor and analyst. You are given a dataframe and the models trained on it.
Your task is to analyze the models and provide insights on the stock price prediction.
You should provide a detailed analysis of the models, including the strengths and weaknesses of each model.
You should also defend the model witht the least error.
Tailor your response to be able to be used by a non-technical audience.
Think of it like talking to a person who knows nothing about the models and wants to make an investment decision.

Here is the dataframe:
{df.head(100).to_string()}

Here are the models:
{results}

Generate a text response. This will be used by another agent to generate a report.
"""
        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.5,
        )
        return response.choices[0].message.content