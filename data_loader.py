import pandas as pd


def load_data(path="data/sentiment.csv"):
    """
    Load Sentiment140 dataset
    """

    columns = ["target", "id", "date", "flag", "user", "text"]

    df = pd.read_csv(
        path,
        encoding="latin-1",
        names=columns,
        header=None
    )

    # Keep only useful columns
    df = df[["target", "text"]]

    # Convert labels
    # 0 -> Negative
    # 4 -> Positive
    df["target"] = df["target"].replace({
        0: "Negative",
        4: "Positive"
    })

    return df


if __name__ == "__main__":
    df = load_data()

    print("\nDataset Loaded Successfully\n")
    print(df.head())
    print("\nShape:", df.shape)