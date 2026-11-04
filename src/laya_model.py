
import time
import laya

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


LAYA_QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "Classify the sentiment of this movie review.",
        "criteria": {
            "positive": "The review expresses a positive opinion.",
            "negative": "The review expresses a negative opinion."
        }
    }
}


def run_laya_inference(df_test):
    router = laya.Router()
    predictions = []

    # Warm up the model before timing
    router.predict("This movie was great.", LAYA_QUESTIONS)

    start = time.perf_counter()

    for i, review in enumerate(df_test["review"]):
        response = router.predict(review, LAYA_QUESTIONS)
        answer = response["answers"]["sentiment"]["choice"]

        if answer not in ("positive", "negative"):
            raise ValueError(f"Unexpected sentiment: {answer}")

        predictions.append(1 if answer == "positive" else 0)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(df_test)} reviews")

    total_time = time.perf_counter() - start

    y_true = df_test["sentiment"].values

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "total_inference_time_sec": total_time,
        "infer_time_per_1k_sec": total_time / len(predictions) * 1000
    }

    return metrics, predictions


def run_laya_batch_inference(df_test, batch_size=32):
    router = laya.Router()

    # Warm up the model before timing
    router.predict("This movie was great.", LAYA_QUESTIONS)

    reviews = df_test["review"].tolist()

    requests = [
        {"state": review, "questions": LAYA_QUESTIONS}
        for review in reviews
    ]

    start = time.perf_counter()

    responses = router.predict_batch(requests, batch_size=batch_size)

    total_time = time.perf_counter() - start

    predictions = []

    for response in responses:
        answer = response["answers"]["sentiment"]["choice"]

        if answer not in ("positive", "negative"):
            raise ValueError(f"Unexpected sentiment: {answer}")

        predictions.append(1 if answer == "positive" else 0)

    y_true = df_test["sentiment"].values

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "total_inference_time_sec": total_time,
        "infer_time_per_1k_sec": total_time / len(predictions) * 1000
    }

    return metrics, predictions
