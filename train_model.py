import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


def train():
    print("Loading cleaned dataset...")

    df = pd.read_csv("data/cleaned_sentiment.csv")

    X = df["clean_text"]
    y = df["target"]

    print("Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    print("Predicting...")
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)

    print("\nAccuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, preds))

    # save
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\nSaved model successfully!")


if __name__ == "__main__":
    train()