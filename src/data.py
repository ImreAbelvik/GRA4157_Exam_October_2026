import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_and_split_data(filepath="data/IMDB Dataset.csv"):
    df = pd.read_csv(filepath)

    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    train, test = train_test_split(
        df, test_size=25000, stratify=df["sentiment"], random_state=42
    )

    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_nested_subsets(df_train, sizes=[100, 250, 500, 1000, 5000, 25000], seed=42):
    df = shuffle(df_train, random_state=seed)

    pos = df[df["sentiment"] == 1]
    neg = df[df["sentiment"] == 0]

    subsets = {}

    for size in sizes:
        n = size // 2

        # Keep the same samples when increasing the training size
        subset = pd.concat([pos.iloc[:n], neg.iloc[:n]])
        subset = shuffle(subset, random_state=seed)

        X = subset["review"].values
        y = subset["sentiment"].values

        subsets[size] = (X, y)

    return subsets
