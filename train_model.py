"""
train_model.py
---------------
Trains a text classifier that maps a sentence to one of six emotions:
joy, sadness, anger, fear, surprise, love.

Pipeline: TF-IDF (word + bigram features) -> Logistic Regression.
This is a standard, explainable baseline for text classification —
easy to reason about in an interview and fast enough to retrain on the fly.

Run:
    python train_model.py
Produces:
    model/emotion_model.joblib   (the fitted sklearn Pipeline)
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/emotion_dataset.csv"
MODEL_PATH = "model/emotion_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], test_size=0.2, random_state=42, stratify=df["emotion"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
