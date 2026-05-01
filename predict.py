import joblib
from preprocess import preprocess_text


# load model
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_sentiment(text):
    cleaned = preprocess_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction


if __name__ == "__main__":
    while True:
        text = input("\nEnter text (type exit to stop): ")

        if text.lower() == "exit":
            break

        result = predict_sentiment(text)

        print("Sentiment:", result)