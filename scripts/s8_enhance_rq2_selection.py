"""
s8_enhance_rq2_selection.py — Strengthened RQ2 structure-selection analysis.

Usage:
    python s8_enhance_rq2_selection.py

Why this script exists:
    The first RQ2 pass shows sparse_topk is an extremely strong oracle baseline.
    For a paper about structured matrix approximation, we also need a fairer
    analysis among non-sparse structured families:
        lowrank vs banded vs monarch

Reads:
    results/spectral/spectral_*.csv
    results/approx/approx_*.csv

Writes:
    results/summary/non_sparse_best_method_counts.csv
    results/summary/non_sparse_selection_model_summary.csv
    results/summary/non_sparse_feature_importance.csv
    results/summary/non_sparse_selection_rules.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "erank",
    "stable_rank",
    "energy_top1",
    "energy_top5",
    "diag_conc",
    "mean_dist",
    "seq_len",
]
NON_SPARSE_METHODS = ["lowrank", "banded", "monarch"]


def load_many(pattern: str) -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in files], ignore_index=True)


def build_dataset(spectral: pd.DataFrame, approx: pd.DataFrame) -> pd.DataFrame:
    key = ["model_tag", "task_id", "layer", "head", "seq_len"]
    approx = approx[approx["method"].isin(NON_SPARSE_METHODS)].copy()
    best = approx.loc[approx.groupby(key)["frob_rel"].idxmin()].copy()
    best = best.rename(
        columns={
            "method": "best_non_sparse_method",
            "param": "best_non_sparse_param",
            "frob_rel": "best_non_sparse_frob_rel",
            "kl_mean": "best_non_sparse_kl_mean",
        }
    )
    keep = list(dict.fromkeys(key + FEATURES + ["domain"]))
    merged = spectral[keep].merge(
        best[key + [
            "best_non_sparse_method",
            "best_non_sparse_param",
            "best_non_sparse_frob_rel",
            "best_non_sparse_kl_mean",
        ]],
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
    merged.to_csv(SUMMARY_DIR / "non_sparse_selection_dataset.csv", index=False, encoding="utf-8-sig")
    return merged


def write_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["model_tag", "best_non_sparse_method"], as_index=False)
        .agg(best_count=("best_non_sparse_method", "size"))
    )
    totals = counts.groupby("model_tag")["best_count"].transform("sum")
    counts["best_fraction"] = (counts["best_count"] / totals).round(6)
    counts.to_csv(SUMMARY_DIR / "non_sparse_best_method_counts.csv", index=False, encoding="utf-8-sig")
    return counts


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> tuple[str, float, float]:
    majority = y_train.value_counts().idxmax()
    pred = np.array([majority] * len(y_test))
    return majority, accuracy_score(y_test, pred), balanced_accuracy_score(y_test, pred)


def evaluate(df: pd.DataFrame, model_tag: str) -> tuple[list[dict], list[dict], str]:
    sub = df[df["model_tag"] == model_tag].copy() if model_tag != "ALL" else df.copy()
    rows = []
    imps = []
    rules = ""

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(
        splitter.split(sub[FEATURES], sub["best_non_sparse_method"], groups=sub["group_id"])
    )
    train = sub.iloc[train_idx]
    test = sub.iloc[test_idx]
    majority, base_acc, base_bal = majority_baseline(
        train["best_non_sparse_method"],
        test["best_non_sparse_method"],
    )

    classifiers = {
        "decision_tree_depth4": DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, random_state=42),
        "random_forest_depth6": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
    }

    for name, clf in classifiers.items():
        clf.fit(train[FEATURES], train["best_non_sparse_method"])
        pred = clf.predict(test[FEATURES])
        rows.append({
            "model_tag": model_tag,
            "classifier": name,
            "n": len(sub),
            "classes": ",".join(sorted(sub["best_non_sparse_method"].unique())),
            "majority_class": majority,
            "accuracy": accuracy_score(test["best_non_sparse_method"], pred),
            "balanced_accuracy": balanced_accuracy_score(test["best_non_sparse_method"], pred),
            "baseline_accuracy": base_acc,
            "baseline_balanced_accuracy": base_bal,
        })
        for feature, value in zip(FEATURES, clf.feature_importances_):
            imps.append({
                "model_tag": model_tag,
                "classifier": name,
                "feature": feature,
                "importance": value,
            })
        if name == "decision_tree_depth4":
            rules = export_text(clf, feature_names=FEATURES)

    return rows, imps, rules


def run_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    rows = []
    imps = []
    rules = {}
    for model_tag in ["ALL"] + sorted(df["model_tag"].unique()):
        model_rows, model_imps, model_rules = evaluate(df, model_tag)
        rows.extend(model_rows)
        imps.extend(model_imps)
        rules[model_tag] = model_rules

    summary = pd.DataFrame(rows).round(6)
    importance = pd.DataFrame(imps).round(6)
    summary.to_csv(SUMMARY_DIR / "non_sparse_selection_model_summary.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(SUMMARY_DIR / "non_sparse_feature_importance.csv", index=False, encoding="utf-8-sig")
    return summary, importance, rules


def write_rules(counts: pd.DataFrame, summary: pd.DataFrame, importance: pd.DataFrame, rules: dict[str, str]):
    lines = []
    lines.append("# Non-Sparse Structure Selection\n")
    lines.append("This analysis excludes `sparse_topk` and predicts the best family among lowrank, banded, and monarch.\n")
    lines.append("## Best Family Counts\n")
    for _, row in counts.iterrows():
        lines.append(
            f"- {row.model_tag}: {row.best_non_sparse_method} "
            f"{int(row.best_count)} heads/tasks ({row.best_fraction:.3f})."
        )

    lines.append("\n## Predictive Models\n")
    for _, row in summary.iterrows():
        lines.append(
            f"- {row.model_tag} / {row.classifier}: accuracy={row.accuracy:.3f}, "
            f"balanced_accuracy={row.balanced_accuracy:.3f}, "
            f"majority_baseline={row.baseline_accuracy:.3f}."
        )

    lines.append("\n## Top Random-Forest Features\n")
    rf = importance[importance["classifier"] == "random_forest_depth6"]
    for model_tag, sub in rf.groupby("model_tag"):
        top = sub.sort_values("importance", ascending=False).head(4)
        joined = ", ".join(f"{r.feature}={r.importance:.3f}" for _, r in top.iterrows())
        lines.append(f"- {model_tag}: {joined}")

    lines.append("\n## Decision Tree Rules\n")
    for model_tag, text in rules.items():
        lines.append(f"### {model_tag}\n")
        lines.append("```text")
        lines.append(text.rstrip())
        lines.append("```")

    (SUMMARY_DIR / "non_sparse_selection_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    spectral = load_many("spectral/spectral_*.csv")
    approx = load_many("approx/approx_*.csv")
    if spectral.empty or approx.empty:
        raise SystemExit("Missing spectral or approximation results.")

    dataset = build_dataset(spectral, approx)
    counts = write_counts(dataset)
    summary, importance, rules = run_models(dataset)
    write_rules(counts, summary, importance, rules)

    print(f"Wrote enhanced RQ2 outputs to {SUMMARY_DIR}")
    print("\nNon-sparse best method counts:")
    print(counts.to_string(index=False))
    print("\nNon-sparse selection model summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
