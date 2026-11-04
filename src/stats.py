
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def compare_models(y_true, y_pred_lr, y_pred_laya, n_bootstrap=1000, seed=42):
    y_true = np.asarray(y_true)
    y_pred_lr = np.asarray(y_pred_lr)
    y_pred_laya = np.asarray(y_pred_laya)

    if not (len(y_true) == len(y_pred_lr) == len(y_pred_laya)):
        raise ValueError("Prediction arrays must have the same length")

    if len(y_true) == 0:
        raise ValueError("Test set cannot be empty")

    lr_correct = y_pred_lr == y_true
    laya_correct = y_pred_laya == y_true

    acc_lr = lr_correct.mean()
    acc_laya = laya_correct.mean()
    acc_diff = acc_lr - acc_laya

    # Paired bootstrap confidence interval
    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        diff = lr_correct[indices].mean() - laya_correct[indices].mean()
        diffs.append(diff)

    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

    # McNemar's test
    both_correct = np.sum(lr_correct & laya_correct)
    only_lr = np.sum(lr_correct & ~laya_correct)
    only_laya = np.sum(~lr_correct & laya_correct)
    both_wrong = np.sum(~lr_correct & ~laya_correct)

    table = [
        [both_correct, only_lr],
        [only_laya, both_wrong]
    ]

    result = mcnemar(table, exact=(only_lr + only_laya < 25),
                     correction=True)

    return {
        "Model 1 Accuracy": acc_lr,
        "Model 2 Accuracy": acc_laya,
        "Model 1 - Model 2": acc_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "Model 1 Only correct:": int(only_lr),
        "Model 2 Only correct:": int(only_laya),
        "mcnemar_pvalue": result.pvalue
    }
