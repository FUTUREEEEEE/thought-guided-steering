"""Metrics for user-supplied, case-grouped out-of-fold probe predictions."""

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_probe(rows: list[dict], *, probability_column: str = "probability") -> dict:
    """Require case_id, step_id, label, fold and a probability column.

    Verify declared case separation and preserve undefined single-class fold AUCs.
    This checks supplied predictions, not the provenance of their training run.
    """
    if not rows:
        raise ValueError("Prediction rows must be nonempty")
    keys = [(str(row["case_id"]), str(row["step_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate decision rows")
    cases = np.array([key[0] for key in keys])
    folds = np.array([str(row["fold"]) for row in rows])
    labels = np.array([float(row["label"]) for row in rows])
    probabilities = np.array([float(row[probability_column]) for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Binary labels with both classes are required for pooled AUC")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Risk probabilities must be finite and in [0, 1]")
    if len(set(folds)) < 2:
        raise ValueError("At least two held-out folds are required")
    for case in np.unique(cases):
        if len(set(folds[cases == case])) != 1:
            raise ValueError("Case leakage across declared held-out folds")
    per_fold = []
    for fold in sorted(set(folds)):
        selected = folds == fold
        score = (float(roc_auc_score(labels[selected], probabilities[selected]))
                 if len(set(labels[selected])) == 2 else None)
        per_fold.append({"fold": fold, "contexts": int(selected.sum()), "roc_auc": score})
    undefined = sum(row["roc_auc"] is None for row in per_fold)
    weighted = (float(np.average([row["roc_auc"] for row in per_fold],
                                 weights=[row["contexts"] for row in per_fold]))
                if not undefined else None)
    return {"contexts": len(rows), "cases": len(set(cases)), "risky": int(labels.sum()),
            "safe": int((labels == 0).sum()), "probability_column": probability_column,
            "pooled_roc_auc": float(roc_auc_score(labels, probabilities)),
            "weighted_fold_roc_auc": weighted, "undefined_fold_count": undefined,
            "folds": per_fold}
