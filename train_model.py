"""
train_model.py

Trains a comment-toxicity classifier on the Jigsaw Toxic Comment
Classification dataset (train.csv) and saves it to disk as model.pkl.

Usage:
    python train_model.py "C:\\Users\\STEPH\\Downloads\\train.csv"

If you don't pass a path, it defaults to train.csv in the current folder.
"""

import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from text_normalize import normalize_text

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    missing = [c for c in ["comment_text"] + LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["comment_text"]).reset_index(drop=True)
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].fillna(0).astype(int)
    print(f"Loaded {len(df)} rows.")

    print("Normalizing text (undoing leetspeak/symbol evasion)...")
    df["comment_text"] = df["comment_text"].astype(str).apply(normalize_text)

    X = df["comment_text"]
    y = df[LABEL_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print("Vectorizing text (word-level + character-level TF-IDF)...")
    # Word-level: catches whole-word meaning, e.g. "idiot", "hate".
    word_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    # Character-level: catches sub-word patterns, so an obfuscated word
    # like "p*nis" or "sl*t" still overlaps heavily with the clean
    # spelling at the character level even though it's a different token.
    char_vectorizer = TfidfVectorizer(
        max_features=30000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )
    vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer),
    ])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training classifier (this should take well under a minute)...")
    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(X_train_vec, y_train)

    print("\nEvaluation on held-out test set:")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=LABEL_COLUMNS, zero_division=0))

    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer, "labels": LABEL_COLUMNS}, f)

    print("\nSaved trained model to model.pkl")

if __name__ == "__main__":
    main()
