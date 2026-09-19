"""
Trains a sentiment analysis model (TF-IDF + Logistic Regression) on
data/reviews.csv, evaluates it, and saves the trained model + vectorizer
to the model/ folder using pickle so app.py can load and serve it.

Run: python3 train_model.py
"""
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "reviews.csv"
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} reviews")
    print(df["sentiment"].value_counts(), "\n")

    X = df["review"]
    y = df["sentiment"]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Vectorize text -> TF-IDF features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Save model + vectorizer
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")

    # 7. Quick sanity check on new unseen sentences
    samples = [
        "This product is absolutely wonderful, I am so happy!",
        "Worst purchase ever, completely useless and broke fast.",
    ]
    samples_vec = vectorizer.transform(samples)
    preds = model.predict(samples_vec)
    print("\nSample predictions:")
    for s, p in zip(samples, preds):
        label = "POSITIVE" if p == 1 else "NEGATIVE"
        print(f"  [{label}] {s}")


if __name__ == "__main__":
    main()
