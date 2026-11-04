import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data import get_nested_subsets


TFIDF_GRID = {
    "tfidf__max_features": [10000, 20000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2, 5],
    "tfidf__stop_words": [None, "english"],
}

MODEL_CONFIGS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=2000),
        "params": {"model__C": [0.1, 1.0, 10.0]},
        "prefix": "lr",
    },
    "Naive Bayes": {
        "estimator": MultinomialNB(),
        "params": {"model__alpha": [0.1, 1.0, 10.0]},
        "prefix": "nb",
    },
}


def _text_array(df):
    """Return a NumPy array that is safe for sklearn CV positional indexing."""
    return df["review"].fillna("").astype(str).to_numpy(dtype=object, copy=True)


def _labels_array(df):
    return df["sentiment"].to_numpy(dtype=np.int64, copy=True)


def _pipeline(name):
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("model", MODEL_CONFIGS[name]["estimator"]),
    ])


def evaluate_model(search, X_train, y_train, X_test, y_test,
                   name, size, seed):
    """Run CV and evaluate the refitted model on the fixed test set."""
    start = time.perf_counter()
    search.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = search.predict(X_test)
    inference_time = time.perf_counter() - start

    return {
        "model": name,
        "sample_size": size,
        "seed": seed,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "best_params": search.best_params_,
        "cv_accuracy": search.best_score_,
        "train_time_sec": train_time,
        "infer_time_sec": inference_time,
        "infer_time_per_1k_sec": inference_time / len(y_test) * 1000,
    }


def fit_and_save_best_model(df_train, df_test, results_df,
                            selection_seed=42, output_dir="results",
                            fitted_estimators=None):
    """Choose the lowest-CV-error model at the largest sample size.

    Only results from ``selection_seed`` are considered, so we do not
    select a seed based on its observed performance. Reuses an already
    fitted estimator when called from ``run_traditional_experiment``;
    otherwise reconstructs and refits it from the saved parameters.

    Returns (best_model, predictions_df).
    """
    size = int(results_df["sample_size"].max())
    candidates = results_df.loc[
        (results_df["sample_size"] == size)
        & (results_df["seed"] == selection_seed)
    ]
    if candidates.empty:
        raise ValueError(
            f"No results for seed={selection_seed}, sample_size={size}. "
            "Choose a seed used in run_traditional_experiment."
        )

    # Minimizing CV classification error is equivalent to maximizing CV accuracy.
    best = candidates.loc[candidates["cv_accuracy"].idxmax()]
    name = best["model"]
    params = best["best_params"]
    if isinstance(params, str):  # CSV serialization stores dictionaries as strings
        params = ast.literal_eval(params)

    print(f"\nSelected model: {name} (training size={size}, seed={selection_seed})")
    print(f"CV accuracy: {best['cv_accuracy']:.4f}")
    print(f"CV error rate: {1 - best['cv_accuracy']:.4f}")
    print(f"Best parameters: {params}")

    if fitted_estimators is not None and name in fitted_estimators:
        best_model = fitted_estimators[name]
    else:
        best_model = _pipeline(name).set_params(**params)
        # Re-create the same subset used in the CV experiment; when size is
        # the full training set, this includes all 25,000 training reviews.
        subsets = get_nested_subsets(df_train, sizes=[size], seed=selection_seed)
        X_raw, y_raw = subsets[size]
        X_train = pd.Series(X_raw).fillna("").astype(str).to_numpy(dtype=object, copy=True)
        y_train = pd.Series(y_raw).to_numpy(dtype=np.int64, copy=True)
        best_model.fit(X_train, y_train)

    X_test = _text_array(df_test)
    y_test = _labels_array(df_test)
    predictions = best_model.predict(X_test)
    classes = list(best_model.classes_)
    probabilities = best_model.predict_proba(X_test)[:, classes.index(1)]
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stable column names for downstream comparisons, regardless of model.
    predictions_df = pd.DataFrame({
        "review": X_test,
        "sentiment": y_test,
        "traditional_prediction": predictions,
        "traditional_probability": probabilities,
        "model": name,
    })
    shared_file = output_dir / "traditional_predictions.csv"
    predictions_df.to_csv(shared_file, index=False)

    # Also retain the earlier LR-specific CSV schema (or its NB equivalent).
    prefix = MODEL_CONFIGS[name]["prefix"]
    specific_file = output_dir / f"{prefix}_predictions.csv"
    predictions_df.rename(columns={
        "traditional_prediction": f"{prefix}_prediction",
        "traditional_probability": f"{prefix}_probability",
    }).drop(columns="model").to_csv(specific_file, index=False)

    print(f"Saved {shared_file}")
    print(f"Saved {specific_file}")
    return best_model, predictions_df


def run_traditional_experiment(df_train, df_test, sizes, seeds,
                               selection_seed=42, output_dir="results"):
    """Run all learning-curve experiments and save the selected model predictions."""
    if selection_seed not in seeds:
        raise ValueError(f"selection_seed={selection_seed} must be in seeds={seeds}")

    X_test = _text_array(df_test)
    y_test = _labels_array(df_test)
    results = []
    fitted_final_candidates = {}
    max_size = max(sizes)

    for seed in seeds:
        print(f"\nSeed: {seed}")
        subsets = get_nested_subsets(df_train, sizes=sizes, seed=seed)

        for size in sizes:
            X_raw, y_raw = subsets[size]
            X_train = pd.Series(X_raw).fillna("").astype(str).to_numpy(dtype=object, copy=True)
            y_train = pd.Series(y_raw).to_numpy(dtype=np.int64, copy=True)
            print(f"\nTraining sample size: {size}")

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            for name, config in MODEL_CONFIGS.items():
                search = GridSearchCV(
                    estimator=_pipeline(name),
                    param_grid={**TFIDF_GRID, **config["params"]},
                    scoring="accuracy",
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                )
                result = evaluate_model(
                    search, X_train, y_train, X_test, y_test, name, size, seed
                )
                results.append(result)

                print(f"{name}: Accuracy={result['accuracy']:.4f}, CV={result['cv_accuracy']:.4f}")
                print(f"Best parameters: {result['best_params']}")

                # The GridSearchCV winner is already refitted on this subset.
                # Keep it so final prediction generation does not train twice.
                if seed == selection_seed and size == max_size:
                    fitted_final_candidates[name] = search.best_estimator_

    results_df = pd.DataFrame(results)
    summary_df = (
        results_df.groupby(["model", "sample_size"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "traditional_results.csv", index=False)
    summary_df.to_csv(output_dir / "traditional_summary.csv", index=False)

    fit_and_save_best_model(
        df_train, df_test, results_df,
        selection_seed=selection_seed,
        output_dir=output_dir,
        fitted_estimators=fitted_final_candidates,
    )
    return results_df, summary_df
