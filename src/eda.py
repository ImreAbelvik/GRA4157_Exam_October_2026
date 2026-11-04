
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def run_eda(
    df,
    text_col="text",
    label_col="label",
    max_features=5000,
    output_dir=None,
):
    """
    Exploratory data analysis for IMDb sentiment classification.
    Displays and saves plots as pdf files.

    This code reproduces plot x in eda 
    """
    project_root = Path(__file__).resolve().parent.parent

    if output_dir is None:
        output_dir = project_root / "results" / "plots"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = df[[text_col, label_col]].dropna().copy()

    # 1. Class distribution
    print("\n=== Class Distribution ===")
    print(data[label_col].value_counts())
    print("\nPercentages:")
    print(
        data[label_col]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    data[label_col].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.pdf", dpi=300)
    plt.show()

    # 2. Review length distribution
    data["word_count"] = data[text_col].str.split().str.len()

    print("\n=== Review Length Statistics ===")
    print(data.groupby(label_col)["word_count"].describe())

    fig, ax = plt.subplots(figsize=(7, 4))
    data.boxplot(
        column="word_count",
        by=label_col,
        showfliers=False,
        ax=ax,
    )
    fig.suptitle("")
    ax.set_title("Review Length by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Words")

    fig.tight_layout()
    fig.savefig(output_dir / "review_length.pdf", dpi=300)
    plt.show()

    # 3. TF-IDF exploration
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(data[text_col])

    terms = vectorizer.get_feature_names_out()
    mean_scores = X_tfidf.mean(axis=0).A1

    top_terms = (
        pd.Series(mean_scores, index=terms)
        .nlargest(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    top_terms.plot.barh(ax=ax)
    ax.set_title("Top 15 Terms by Mean TF-IDF")
    ax.set_xlabel("Mean TF-IDF Score")

    fig.tight_layout()
    fig.savefig(output_dir / "tfidf_top_terms.pdf", dpi=300)
    plt.show()

    print(f"\nPlots saved to: {output_dir}")

    return data
