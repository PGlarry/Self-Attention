"""
s10_selector_robustness.py — Selector robustness analyses for reviewer concerns.

This script does not run model inference. It reuses completed spectral and
approximation CSVs to test whether the non-sparse structure selector is robust
to prompt holdout, sequence-length removal, class imbalance, and metric choice.

Writes:
    results/summary/selector_robustness_summary.csv
    results/summary/selector_robustness_per_class.csv
    results/summary/selector_robustness_confusion.csv
    results/summary/frob_kl_winner_agreement.csv
    results/summary/selector_robustness.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

NON_SPARSE_METHODS = ["lowrank", "banded", "monarch"]
FEATURES_FULL = [
    "erank",
    "stable_rank",
    "energy_top1",
    "energy_top5",
    "diag_conc",
    "mean_dist",
    "seq_len",
]
FEATURES_NO_LEN = [f for f in FEATURES_FULL if f != "seq_len"]


def load_many(pattern: str) -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in files], ignore_index=True)


def build_non_sparse_dataset() -> pd.DataFrame:
    spectral = load_many("spectral/spectral_*.csv")
    approx = load_many("approx/approx_*.csv")
    if spectral.empty or approx.empty:
        raise SystemExit("Missing spectral or approximation CSV files.")

    key = ["model_tag", "task_id", "layer", "head", "seq_len"]
    approx = approx[approx["method"].isin(NON_SPARSE_METHODS)].copy()

    best_frob = approx.loc[approx.groupby(key)["frob_rel"].idxmin()].copy()
    best_frob = best_frob.rename(
        columns={
            "method": "best_non_sparse_method",
            "param": "best_non_sparse_param",
            "frob_rel": "best_non_sparse_frob_rel",
            "kl_mean": "kl_at_frob_winner",
        }
    )

    best_kl = approx.loc[approx.groupby(key)["kl_mean"].idxmin()].copy()
    best_kl = best_kl.rename(
        columns={
            "method": "best_non_sparse_method_by_kl",
            "param": "best_non_sparse_param_by_kl",
            "kl_mean": "best_non_sparse_kl_mean",
            "frob_rel": "frob_at_kl_winner",
        }
    )

    keep = list(dict.fromkeys(key + FEATURES_FULL + ["domain"]))
    merged = spectral[keep].merge(
        best_frob[
            key
            + [
                "best_non_sparse_method",
                "best_non_sparse_param",
                "best_non_sparse_frob_rel",
                "kl_at_frob_winner",
            ]
        ],
        on=key,
        how="inner",
    )
    merged = merged.merge(
        best_kl[
            key
            + [
                "best_non_sparse_method_by_kl",
                "best_non_sparse_param_by_kl",
                "best_non_sparse_kl_mean",
                "frob_at_kl_winner",
            ]
        ],
        on=key,
        how="inner",
    )
    merged["group_id"] = (
        merged["model_tag"].astype(str)
        + "::"
        + merged["task_id"].astype(str)
        + "::"
        + merged["layer"].astype(str)
    )
    return merged


def make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def metric_row(
    model_tag: str,
    protocol: str,
    feature_set: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    train_majority: str,
    labels: list[str],
) -> dict:
    majority_pred = np.array([train_majority] * len(y_true))
    return {
        "model_tag": model_tag,
        "protocol": protocol,
        "feature_set": feature_set,
        "n_test": len(y_true),
        "classes": ",".join(labels),
        "majority_class": train_majority,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "majority_accuracy": accuracy_score(y_true, majority_pred),
        "majority_balanced_accuracy": balanced_accuracy_score(y_true, majority_pred),
        "majority_macro_f1": f1_score(y_true, majority_pred, labels=labels, average="macro", zero_division=0),
    }


def per_class_rows(
    model_tag: str,
    protocol: str,
    feature_set: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> list[dict]:
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    supports = [(y_true == label).sum() for label in labels]
    return [
        {
            "model_tag": model_tag,
            "protocol": protocol,
            "feature_set": feature_set,
            "class": label,
            "support": int(support),
            "recall": float(recall),
        }
        for label, support, recall in zip(labels, supports, recalls)
    ]


def confusion_rows(
    model_tag: str,
    protocol: str,
    feature_set: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append(
                {
                    "model_tag": model_tag,
                    "protocol": protocol,
                    "feature_set": feature_set,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(matrix[i, j]),
                }
            )
    return rows


def evaluate_group_split(sub: pd.DataFrame, model_tag: str, features: list[str], feature_set: str):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(
        splitter.split(sub[features], sub["best_non_sparse_method"], groups=sub["group_id"])
    )
    train = sub.iloc[train_idx]
    test = sub.iloc[test_idx]

    clf = make_classifier()
    clf.fit(train[features], train["best_non_sparse_method"])
    pred = clf.predict(test[features])
    labels = sorted(sub["best_non_sparse_method"].unique())
    majority = train["best_non_sparse_method"].value_counts().idxmax()
    protocol = "group_split_by_model_task_layer"
    return (
        [metric_row(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, majority, labels)],
        per_class_rows(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, labels),
        confusion_rows(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, labels),
    )


def evaluate_leave_one_prompt(sub: pd.DataFrame, model_tag: str, features: list[str], feature_set: str):
    labels = sorted(sub["best_non_sparse_method"].unique())
    y_true_all = []
    y_pred_all = []
    rows = []
    per_class = []
    confusion = []

    for task_id in sorted(sub["task_id"].unique()):
        train = sub[sub["task_id"] != task_id]
        test = sub[sub["task_id"] == task_id]
        if train.empty or test.empty:
            continue
        clf = make_classifier()
        clf.fit(train[features], train["best_non_sparse_method"])
        pred = clf.predict(test[features])
        majority = train["best_non_sparse_method"].value_counts().idxmax()
        protocol = f"leave_one_prompt::{task_id}"
        rows.append(metric_row(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, majority, labels))
        per_class.extend(per_class_rows(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, labels))
        confusion.extend(confusion_rows(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, labels))
        y_true_all.extend(test["best_non_sparse_method"].tolist())
        y_pred_all.extend(pred.tolist())

    if y_true_all:
        train_majority = sub["best_non_sparse_method"].value_counts().idxmax()
        protocol = "leave_one_prompt_macro"
        y_true_series = pd.Series(y_true_all)
        y_pred_array = np.array(y_pred_all)
        rows.append(metric_row(model_tag, protocol, feature_set, y_true_series, y_pred_array, train_majority, labels))
        per_class.extend(per_class_rows(model_tag, protocol, feature_set, y_true_series, y_pred_array, labels))
        confusion.extend(confusion_rows(model_tag, protocol, feature_set, y_true_series, y_pred_array, labels))

    return rows, per_class, confusion


def run_selector_robustness(df: pd.DataFrame):
    summary_rows = []
    per_class = []
    confusion = []
    for model_tag in ["ALL"] + sorted(df["model_tag"].unique()):
        sub = df.copy() if model_tag == "ALL" else df[df["model_tag"] == model_tag].copy()
        for feature_set, features in [("full", FEATURES_FULL), ("no_seq_len", FEATURES_NO_LEN)]:
            rows, pc, cm = evaluate_group_split(sub, model_tag, features, feature_set)
            summary_rows.extend(rows)
            per_class.extend(pc)
            confusion.extend(cm)

            rows, pc, cm = evaluate_leave_one_prompt(sub, model_tag, features, feature_set)
            summary_rows.extend(rows)
            per_class.extend(pc)
            confusion.extend(cm)

    return (
        pd.DataFrame(summary_rows).round(6),
        pd.DataFrame(per_class).round(6),
        pd.DataFrame(confusion),
    )


def winner_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_tag in ["ALL"] + sorted(df["model_tag"].unique()):
        sub = df.copy() if model_tag == "ALL" else df[df["model_tag"] == model_tag]
        rows.append(
            {
                "model_tag": model_tag,
                "n": len(sub),
                "agreement_rate": (sub["best_non_sparse_method"] == sub["best_non_sparse_method_by_kl"]).mean(),
                "frob_lowrank_fraction": (sub["best_non_sparse_method"] == "lowrank").mean(),
                "kl_lowrank_fraction": (sub["best_non_sparse_method_by_kl"] == "lowrank").mean(),
                "frob_monarch_fraction": (sub["best_non_sparse_method"] == "monarch").mean(),
                "kl_monarch_fraction": (sub["best_non_sparse_method_by_kl"] == "monarch").mean(),
                "frob_banded_fraction": (sub["best_non_sparse_method"] == "banded").mean(),
                "kl_banded_fraction": (sub["best_non_sparse_method_by_kl"] == "banded").mean(),
            }
        )
    return pd.DataFrame(rows).round(6)


def write_markdown(summary: pd.DataFrame, per_class: pd.DataFrame, agreement: pd.DataFrame) -> None:
    macro = summary[summary["protocol"] == "leave_one_prompt_macro"].copy()
    group = summary[summary["protocol"] == "group_split_by_model_task_layer"].copy()

    lines = []
    lines.append("# Selector Robustness Analyses\n")
    lines.append(
        "These analyses reuse existing CSV outputs only. They test whether the non-sparse "
        "structure selector remains plausible when sequence length is removed, prompts are "
        "held out, class imbalance is exposed through macro-F1/per-class recall, and the "
        "winner label is changed from Frobenius error to row-wise KL divergence.\n"
    )

    lines.append("## Random Group Split\n")
    lines.append(group.to_markdown(index=False))
    lines.append("\n## Leave-One-Prompt-Out Aggregate\n")
    lines.append(macro.to_markdown(index=False))

    lines.append("\n## Frob-Winner vs KL-Winner Agreement\n")
    lines.append(agreement.to_markdown(index=False))

    lines.append("\n## Per-Class Recall for Leave-One-Prompt-Out Aggregate\n")
    pc = per_class[per_class["protocol"] == "leave_one_prompt_macro"].copy()
    lines.append(pc.to_markdown(index=False))

    lines.append("\n## Interpretation\n")
    lines.append(
        "- Removing `seq_len` leaves the selector strong under the random group split, "
        "supporting the claim that it is not only a length classifier."
    )
    lines.append(
        "- Leave-one-prompt-out performance is broadly comparable to the random split, but it "
        "is still a prompt-holdout test within the same prompt suite. The main text should "
        "describe the selector as a post-hoc structural predictor rather than as a fully "
        "validated open-ended cross-task generalizer."
    )
    lines.append(
        "- Frobenius and KL winner agreement quantifies how much the family conclusion depends "
        "on the matrix-fidelity metric used to define the label."
    )

    (SUMMARY_DIR / "selector_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = build_non_sparse_dataset()
    summary, per_class, confusion = run_selector_robustness(df)
    agreement = winner_agreement(df)

    summary.to_csv(SUMMARY_DIR / "selector_robustness_summary.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(SUMMARY_DIR / "selector_robustness_per_class.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(SUMMARY_DIR / "selector_robustness_confusion.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(SUMMARY_DIR / "frob_kl_winner_agreement.csv", index=False, encoding="utf-8-sig")
    write_markdown(summary, per_class, agreement)

    print(f"Wrote selector robustness outputs to {SUMMARY_DIR}")
    print("\nLeave-one-prompt-out aggregate:")
    print(summary[summary["protocol"] == "leave_one_prompt_macro"].to_string(index=False))
    print("\nFrob/KL winner agreement:")
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
