"""
FastAPI service that serves the trained sentiment analysis model.

Run locally:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts whether a product review is POSITIVE or NEGATIVE.",
    version="1.0.0",
)

# Load model + vectorizer once at startup
try:
    with open("/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    model, vectorizer = None, None


class ReviewRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running. Visit /docs to try it out."}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Run train_model.py first to generate model files.",
        )

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    vec = vectorizer.transform([request.text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))

    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    return PredictionResponse(text=request.text, sentiment=sentiment, confidence=round(confidence, 4))
