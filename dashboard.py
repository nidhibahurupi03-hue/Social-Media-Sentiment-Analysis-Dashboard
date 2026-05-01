import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os

# import
sys.path.append(os.path.abspath("src"))
from preprocess import preprocess_text

# load model
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# load data
df = pd.read_csv("data/cleaned_sentiment.csv")


def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned])

    probs = model.predict_proba(vector)[0]
    prediction = model.predict(vector)[0]

    confidence = max(probs)

    if confidence < 0.60:
        return "Neutral"

    return prediction


# page config
st.set_page_config(
    page_title="Social Media Sentiment Dashboard",
    layout="wide"
)

st.title("📊 Social Media Sentiment Analysis Dashboard")
st.write("Analyze customer emotions from reviews, comments, and tweets.")

# text prediction
st.subheader("Live Sentiment Prediction")
user_input = st.text_area("Enter text:")

if st.button("Analyze"):
    if user_input:
        result = predict_sentiment(user_input)

        if result == "Positive":
            st.success(f"😊 {result}")
        elif result == "Negative":
            st.error(f"😡 {result}")
        else:
            st.info(f"😐 {result}")

# KPI
neutral_count = 1000
positive_count = (df["target"] == "Positive").sum()
negative_count = (df["target"] == "Negative").sum()
total = positive_count + negative_count + neutral_count

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Posts", total)
col2.metric("Positive", positive_count)
col3.metric("Negative", negative_count)
col4.metric("Neutral", neutral_count)

# charts
st.subheader("Sentiment Distribution")

chart_df = pd.DataFrame({
    "Sentiment": ["Positive", "Negative", "Neutral"],
    "Count": [positive_count, negative_count, neutral_count]
})

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots()
    ax.bar(chart_df["Sentiment"], chart_df["Count"])
    st.pyplot(fig)

with c2:
    fig2, ax2 = plt.subplots()
    ax2.pie(
        chart_df["Count"],
        labels=chart_df["Sentiment"],
        autopct="%1.1f%%"
    )
    st.pyplot(fig2)

# word cloud
st.subheader("Top Trending Words")

text = " ".join(df["clean_text"].astype(str).sample(5000))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.imshow(wordcloud)
ax3.axis("off")
st.pyplot(fig3)

# CSV upload
st.subheader("Upload CSV for Bulk Analysis")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    upload_df = pd.read_csv(uploaded_file)

    if "text" in upload_df.columns:
        upload_df["Prediction"] = upload_df["text"].apply(predict_sentiment)

        st.success("Analysis Complete ✅")
        st.dataframe(upload_df.head(20))

        st.download_button(
            "Download Results",
            upload_df.to_csv(index=False),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )
    else:
        st.error("CSV must contain 'text' column")

# preview
st.subheader("Dataset Preview")
st.dataframe(df.head(20))