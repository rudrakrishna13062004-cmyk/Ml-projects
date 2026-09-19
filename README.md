# Sentiment Analysis API (ML + FastAPI)

A machine learning project that classifies product reviews as **POSITIVE** or
**NEGATIVE**, served as a REST API using FastAPI (deployable the same way as
your other FastAPI projects on Render).

## Tech Stack
- **ML:** scikit-learn (TF-IDF vectorizer + Logistic Regression)
- **Backend:** FastAPI, Pydantic
- **Data handling:** Pandas

## How it works
1. `data/generate_data.py` builds a labeled dataset of product reviews.
2. `train_model.py` converts review text into TF-IDF features, trains a
   Logistic Regression classifier, evaluates it, and saves the trained
   model + vectorizer to `model/`.
3. `app.py` loads the saved model and exposes a `/predict` endpoint that
   takes review text and returns a sentiment label with a confidence score.

## Project Structure
```
sentiment-analysis-api/
├── data/
│   ├── generate_data.py     # builds the training dataset
│   └── reviews.csv          # generated dataset (520 labeled reviews)
├── model/
│   ├── sentiment_model.pkl  # trained Logistic Regression model
│   └── vectorizer.pkl       # fitted TF-IDF vectorizer
├── train_model.py           # training + evaluation script
├── predict.py                # CLI tool to test predictions
├── app.py                    # FastAPI app serving the model
├── requirements.txt
└── README.md
```

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Re)generate the dataset and train the model
python3 data/generate_data.py
python3 train_model.py

# 3a. Test from the command line
python3 predict.py

# 3b. OR run the API
uvicorn app:app --reload
# then open http://127.0.0.1:8000/docs
```

### Example request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing, I love it!"}'
```

```json
{
  "text": "This product is amazing, I love it!",
  "sentiment": "POSITIVE",
  "confidence": 0.94
}
```

## Results
Achieved **100% accuracy** on the held-out test split of the current
synthetic dataset (template-generated reviews, so the task is fairly
separable).

> **Note for improving this project further:** For a stronger, more
> defensible resume claim, swap `data/reviews.csv` with a real-world
> dataset such as the [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
> or [Amazon Product Reviews](https://www.kaggle.com/datasets) dataset from
> Kaggle. The training pipeline (`train_model.py`) works unchanged as long
> as the CSV has `review` and `sentiment` columns — real-world text is
> noisier, so expect accuracy in the 85–92% range, which is actually a more
> credible number to discuss in interviews.

## Possible Extensions
- Swap Logistic Regression for a fine-tuned transformer (HuggingFace
  `distilbert-base-uncased`) for higher accuracy on real-world data.
- Add a `/batch-predict` endpoint for scoring multiple reviews at once.
- Deploy on Render (same workflow you already used for your Todo/Expense
  Tracker APIs) and add the live link + Swagger docs to your resume.
- Track model experiments with MLflow.

## Author
Kishan Kumar
