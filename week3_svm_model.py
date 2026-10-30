#!/usr/bin/env python3
"""Week 3: train, tune, evaluate, and visualize a breast-tumour classifier.

Dataset: the copy of UCI's Wisconsin Diagnostic Breast Cancer data bundled
with scikit-learn. The script needs no data download or local input file.

Install: python -m pip install numpy scikit-learn matplotlib
Run:     python week3_svm_model.py --output-dir results

Outputs: metrics.json, cv_results.csv, confusion_matrix.png, roc_curve.png,
         and cv_top_candidates.png. The holdout test set is used only after
         the model and hyperparameters have been selected on the training set.

Educational demonstration only: this model must not be used for diagnosis.
"""

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import matplotlib

# Use a non-interactive backend so plots can be produced on servers and in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
TEST_FRACTION = 0.20
FOLDS = 5
C_VALUES = [0.01, 0.1, 1, 10, 100]
GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1]
CLASS_WEIGHTS = [None, "balanced"]


def wilson_interval(successes, total, z=1.959963984540054):
    """Approximate two-sided 95% Wilson binomial interval for a proportion."""
    if total == 0:
        return [None, None]
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def class_counts(labels):
    """Return readable counts with malignant encoded as the positive class."""
    return {"benign_0": int(np.sum(labels == 0)), "malignant_1": int(np.sum(labels == 1))}


def candidate_label(params):
    """Compact, unambiguous label for the cross-validation comparison plot."""
    kernel = params["svc__kernel"]
    weight = params["svc__class_weight"] or "none"
    label = f"{kernel.upper()}  |  C={params['svc__C']}"
    if kernel == "rbf":
        label += f"  |  gamma={params['svc__gamma']}"
    return label + f"  |  {weight}"


def make_figures(out_dir, cm, y_test, decision_scores, search, baseline_f1):
    """Visualize the untouched-test evaluation and training-CV tuning results."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # Matrix rows are actual classes; columns are predicted classes.
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    display = ConfusionMatrixDisplay(cm, display_labels=["Benign (0)", "Malignant (1)"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Held-out test set: confusion matrix", fontweight="bold", pad=14)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ROC AUC uses the continuous SVM decision score, not hard labels or probabilities.
    fpr, tpr, _ = roc_curve(y_test, decision_scores, pos_label=1)
    auc = roc_auc_score(y_test, decision_scores)
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    ax.plot(fpr, tpr, color="#176d7c", linewidth=2.5, label=f"SVM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="Chance reference")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
           title="Held-out test set: ROC curve")
    ax.title.set_fontweight("bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out_dir / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Compare training-CV candidates. Dots are means, whiskers are fold SDs
    # (not confidence intervals). The horizontal axis is deliberately zoomed;
    # the independent test set is not used in this comparison.
    result = search.cv_results_
    order = np.argsort(result["rank_test_score"], kind="stable")[:8]
    labels = [candidate_label(result["params"][i]) for i in order]
    means = np.array([result["mean_test_score"][i] for i in order])
    stds = np.array([result["std_test_score"][i] for i in order])
    fig, ax = plt.subplots(figsize=(9.8, 5.0), constrained_layout=True)
    y_pos = np.arange(len(order))
    ax.errorbar(means, y_pos, xerr=stds, fmt="o", markersize=6.5,
                color="#176d7c", ecolor="#54777d", capsize=4, linewidth=1.4)
    ax.axvline(baseline_f1, color="#b45b43", linestyle="--", linewidth=1.5,
               label=f"Default SVC mean F1 = {baseline_f1:.3f}")
    ax.set_yticks(y_pos, labels=labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, min(float(np.min(means - stds)), baseline_f1) - 0.005),
                min(1.0, float(np.max(means + stds)) + 0.005))
    ax.set_xlabel("Mean validation F1, zoomed scale (whiskers: ±1 fold SD)")
    ax.set_title("Top hyperparameter candidates on training data", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.20), fontsize=10, frameon=False)
    fig.savefig(out_dir / "cv_top_candidates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results", help="Folder for metrics, CV table, and plots")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the public dataset. Scikit-learn's original encoding is 0 =
    # malignant and 1 = benign; reverse it so that malignant is positive (1).
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target
    if not np.isfinite(X).all():
        raise ValueError("Unexpected non-finite features; investigate missing data before training.")

    # 2. Keep a stratified test set aside before any scaling or model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y
    )

    # 3. The scaler is inside the pipeline: each CV fold learns scaling only
    # from its own training partition, preventing validation-data leakage.
    model = Pipeline([("scale", StandardScaler()), ("svc", SVC())])
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}

    # 4. Benchmark the default SVC with the same training folds used for tuning.
    # These CV estimates are NOT test-set scores.
    baseline = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    baseline_summary = {
        name: {"mean": float(np.mean(baseline[f"test_{name}"])),
               "std": float(np.std(baseline[f"test_{name}"]))}
        for name in scoring
    }

    # 5. Search 10 linear + 40 RBF settings = 50 candidates; 5 folds = 250
    # training fits. F1 balances precision and recall for malignant cases.
    # No gamma is supplied for a linear kernel because it has no effect there.
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": C_VALUES, "svc__class_weight": CLASS_WEIGHTS},
        {"svc__kernel": ["rbf"], "svc__C": C_VALUES, "svc__gamma": GAMMA_VALUES,
         "svc__class_weight": CLASS_WEIGHTS},
    ]
    search = GridSearchCV(
        model, param_grid, scoring="f1", cv=cv, refit=True, return_train_score=True,
        n_jobs=1, error_score="raise"
    )
    search.fit(X_train, y_train)  # Refit the selected pipeline on all training data.

    # 6. Evaluate the chosen model once on the previously untouched test set.
    # The SVC default decision threshold is 0; decision scores are not calibrated
    # probabilities. ROC AUC uses scores while the other metrics use predictions.
    y_pred = search.predict(X_test)
    decision_scores = search.decision_function(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    report = classification_report(y_test, y_pred, target_names=["benign", "malignant"],
                                   output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_malignant": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_malignant": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_malignant": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "specificity_benign": float(tn / (tn + fp)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, decision_scores)),
        "malignant_recall_wilson_95": wilson_interval(tp, tp + fn),
    }

    # 7. Save all candidate results for transparent, reproducible inspection.
    cv_results = search.cv_results_
    candidate_rows = []
    for i, params in enumerate(cv_results["params"]):
        candidate_rows.append({
            "rank": int(cv_results["rank_test_score"][i]),
            "kernel": params["svc__kernel"],
            "C": params["svc__C"],
            "gamma": params.get("svc__gamma", "not applicable"),
            "class_weight": params["svc__class_weight"] or "none",
            "mean_train_f1": float(cv_results["mean_train_score"][i]),
            "mean_validation_f1": float(cv_results["mean_test_score"][i]),
            "std_validation_f1": float(cv_results["std_test_score"][i]),
            "mean_fit_seconds": float(cv_results["mean_fit_time"][i]),
        })
    candidate_rows.sort(key=lambda row: (row["rank"], -row["mean_validation_f1"]))
    with (out_dir / "cv_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    results = {
        "dataset": {
            "name": "Wisconsin Diagnostic Breast Cancer (scikit-learn built-in copy)",
            "samples": int(len(y)), "features": int(X.shape[1]),
            "missing_values": int(np.isnan(X).sum()),
            "all_counts": class_counts(y), "train_counts": class_counts(y_train),
            "test_counts": class_counts(y_test),
            "train_size": int(len(y_train)), "test_size": int(len(y_test)),
            "encoding": {"0": "benign", "1": "malignant (positive)"},
        },
        "method": {
            "split": "stratified 80% training / 20% test", "random_seed": SEED,
            "cross_validation": "stratified 5-fold, shuffled", "selection_metric": "malignant F1",
            "candidates": len(candidate_rows), "grid_fits_excluding_refit": len(candidate_rows) * FOLDS,
            "c_values": C_VALUES, "gamma_values_rbf": GAMMA_VALUES, "class_weights": CLASS_WEIGHTS,
        },
        "baseline_training_cv": baseline_summary,
        "best_params": search.best_params_,
        "best_training_cv_f1_mean": float(search.best_score_),
        "best_training_cv_f1_std": float(cv_results["std_test_score"][search.best_index_]),
        "best_training_fold_f1_mean": float(cv_results["mean_train_score"][search.best_index_]),
        "test_metrics": metrics,
        "confusion_matrix_rows_actual_columns_predicted": cm.tolist(),
        "confusion_counts": {"true_negative": tn, "false_positive": fp, "false_negative": fn,
                             "true_positive": tp},
        "classification_report": report,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    make_figures(out_dir, cm, y_test, decision_scores, search, baseline_summary["f1"]["mean"])

    print(f"Dataset: {len(y)} samples, {X.shape[1]} features; train={len(y_train)}, test={len(y_test)}")
    print("Positive class: malignant (1). Confusion matrix rows=actual, columns=predicted:")
    print(cm)
    print(f"Default SVC training-CV F1: {baseline_summary['f1']['mean']:.4f}")
    print(f"Best training-CV F1: {search.best_score_:.4f}; parameters: {search.best_params_}")
    for key in ("accuracy", "precision_malignant", "recall_malignant", "f1_malignant", "roc_auc"):
        print(f"Test {key}: {metrics[key]:.4f}")
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
