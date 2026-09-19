"""
Quick CLI to test the trained model without spinning up the API.
Run: python3 predict.py
"""
import pickle

with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Sentiment Analysis - type a review (or 'quit' to exit)\n")
while True:
    text = input("> ")
    if text.strip().lower() in ("quit", "exit"):
        break
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = max(model.predict_proba(vec)[0])
    label = "POSITIVE 🙂" if pred == 1 else "NEGATIVE 🙁"
    print(f"  {label}  (confidence: {proba:.2f})\n")
