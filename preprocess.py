import re
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean social media text
    """

    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove @mentions
    text = re.sub(r"@\w+", "", text)

    # remove hashtags symbol
    text = re.sub(r"#", "", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text):
    words = text.split()

    filtered = [
        word for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(filtered)


def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


if __name__ == "__main__":
    sample = "@amazon Great product!! Loved it 😍 Visit http://abc.com #happy"

    print("Original:")
    print(sample)

    print("\nProcessed:")
    print(preprocess_text(sample))