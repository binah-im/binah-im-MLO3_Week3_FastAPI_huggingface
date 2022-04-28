from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class PredictionRequest(BaseModel):
  query_string: str


app = FastAPI()
sentiment_model = pipeline("sentiment-analysis")

@app.post("/my-endpoint")
def my_endpoint(request: PredictionRequest):
    sentiment_query_sentence = request.query_string
    sentiment = sentiment_model(sentiment_query_sentence)
    return f"Sentiment test: {sentiment_query_sentence} == {sentiment}"