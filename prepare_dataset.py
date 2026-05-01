from data_loader import load_data
from preprocess import preprocess_text


def prepare():
    print("Loading dataset...")
    df = load_data()

    print("Cleaning text... (थोडा वेळ लागेल)")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # remove empty rows
    df = df[df["clean_text"] != ""]

    # save cleaned data
    df.to_csv("data/cleaned_sentiment.csv", index=False)

    print("\nSaved:")
    print("data/cleaned_sentiment.csv")
    print("\nShape:", df.shape)

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    prepare()