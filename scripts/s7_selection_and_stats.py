"""
s7_selection_and_stats.py — Zero-shot structure selection and probe statistics.

Usage:
    python s7_selection_and_stats.py

This script completes the original RQ2/RQ3 analysis layer:

RQ2:
    Merge spectral features with approximation errors and test whether simple
    spectral/locality features predict the best approximation family.

RQ3:
    Run pairwise Mann-Whitney tests over selected-head causal probe groups.

Writes:
    results/summary/selection_dataset.csv
    results/summary/selection_model_summary.csv
    results/summary/selection_feature_importance.csv
    results/summary/probe_pairwise_tests.csv
    results/summary/selection_and_stats.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
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


def load_many(pattern: str) -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob(pattern))
    files = [path for path in files if path.name != "probe_targets.csv"]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in files], ignore_index=True)


def build_selection_dataset(spectral: pd.DataFrame, approx: pd.DataFrame) -> pd.DataFrame:
    key = ["model_tag", "task_id", "layer", "head", "seq_len"]
    best_rows = approx.loc[approx.groupby(key)["frob_rel"].idxmin()].copy()
    best_rows = best_rows.rename(
        columns={
            "method": "best_method",
            "param": "best_param",
            "frob_rel": "best_frob_rel",
            "kl_mean": "best_kl_mean",
        }
    )
    keep_cols = list(dict.fromkeys(key + FEATURES + ["domain"]))
    merged = spectral[keep_cols].merge(
        best_rows[key + ["best_method", "best_param", "best_frob_rel", "best_kl_mean"]],
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
    merged.to_csv(SUMMARY_DIR / "selection_dataset.csv", index=False, encoding="utf-8-sig")
    return merged


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> tuple[str, float, float]:
    majority = y_train.value_counts().idxmax()
    pred = np.array([majority] * len(y_test))
    return majority, accuracy_score(y_test, pred), balanced_accuracy_score(y_test, pred)


def evaluate_classifier(df: pd.DataFrame, model_tag: str) -> tuple[list[dict], list[dict], str]:
    rows = []
    importances = []
    tree_rules = ""

    sub = df[df["model_tag"] == model_tag].copy() if model_tag != "ALL" else df.copy()
    if sub["best_method"].nunique() < 2:
        majority = sub["best_method"].value_counts().idxmax()
        rows.append({
            "model_tag": model_tag,
            "classifier": "majority_only",
            "n": len(sub),
            "classes": ",".join(sorted(sub["best_method"].unique())),
            "majority_class": majority,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "baseline_accuracy": 1.0,
            "baseline_balanced_accuracy": 1.0,
        })
        return rows, importances, tree_rules

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(sub[FEATURES], sub["best_method"], groups=sub["group_id"]))
    train = sub.iloc[train_idx]
    test = sub.iloc[test_idx]

    majority, base_acc, base_bal = majority_baseline(train["best_method"], test["best_method"])

    classifiers = {
        "decision_tree_depth3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=42),
        "random_forest_depth5": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
    }

    for name, clf in classifiers.items():
        clf.fit(train[FEATURES], train["best_method"])
        pred = clf.predict(test[FEATURES])
        rows.append({
            "model_tag": model_tag,
            "classifier": name,
            "n": len(sub),
            "classes": ",".join(sorted(sub["best_method"].unique())),
            "majority_class": majority,
            "accuracy": accuracy_score(test["best_method"], pred),
            "balanced_accuracy": balanced_accuracy_score(test["best_method"], pred),
            "baseline_accuracy": base_acc,
            "baseline_balanced_accuracy": base_bal,
        })

        if hasattr(clf, "feature_importances_"):
            for feature, value in zip(FEATURES, clf.feature_importances_):
                importances.append({
                    "model_tag": model_tag,
                    "classifier": name,
                    "feature": feature,
                    "importance": value,
                })

        if name == "decision_tree_depth3":
            tree_rules = export_text(clf, feature_names=FEATURES)

    return rows, importances, tree_rules


def selection_models(selection_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    summary_rows = []
    importance_rows = []
    rules = {}

    for model_tag in ["ALL"] + sorted(selection_df["model_tag"].unique()):
        rows, imps, tree_rules = evaluate_classifier(selection_df, model_tag)
        summary_rows.extend(rows)
        importance_rows.extend(imps)
        if tree_rules:
            rules[model_tag] = tree_rules

    summary = pd.DataFrame(summary_rows).round(6)
    importance = pd.DataFrame(importance_rows).round(6)
    summary.to_csv(SUMMARY_DIR / "selection_model_summary.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(SUMMARY_DIR / "selection_feature_importance.csv", index=False, encoding="utf-8-sig")
    return summary, importance, rules


def probe_tests(probe: pd.DataFrame) -> pd.DataFrame:
    groups = ["high_complex_global", "local_band", "low_rank_diffuse"]
    rows = []
    for model_tag, sub in probe.groupby("model_tag"):
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                x = sub[sub["target_group"] == left]["kl_div"].dropna()
                y = sub[sub["target_group"] == right]["kl_div"].dropna()
                if len(x) == 0 or len(y) == 0:
                    continue
                stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
                rows.append({
                    "model_tag": model_tag,
                    "left_group": left,
                    "right_group": right,
                    "left_mean_kl": x.mean(),
                    "right_mean_kl": y.mean(),
                    "mean_diff_left_minus_right": x.mean() - y.mean(),
                    "mannwhitney_u": stat,
                    "p_value": p_value,
                    "n_left": len(x),
                    "n_right": len(y),
                })
    out = pd.DataFrame(rows).round(6)
    out.to_csv(SUMMARY_DIR / "probe_pairwise_tests.csv", index=False, encoding="utf-8-sig")
    return out


def write_report(selection_summary: pd.DataFrame, importance: pd.DataFrame,
                 rules: dict[str, str], probe_pairwise: pd.DataFrame):
    lines = []
    lines.append("# Selection and Probe Statistics\n")
    lines.append("## RQ2: Zero-shot Structure Selection\n")
    for _, row in selection_summary.iterrows():
        lines.append(
            f"- {row.model_tag} / {row.classifier}: accuracy={row.accuracy:.3f}, "
            f"balanced_accuracy={row.balanced_accuracy:.3f}, "
            f"majority_baseline={row.baseline_accuracy:.3f} "
            f"(classes={row.classes})."
        )

    if not importance.empty:
        lines.append("\nTop random-forest feature importances:")
        rf = importance[importance["classifier"] == "random_forest_depth5"]
        for model_tag, sub in rf.groupby("model_tag"):
            top = sub.sort_values("importance", ascending=False).head(3)
            joined = ", ".join(f"{r.feature}={r.importance:.3f}" for _, r in top.iterrows())
            lines.append(f"- {model_tag}: {joined}")

    lines.append("\nDecision tree rules are below for non-degenerate model subsets.\n")
    for model_tag, text in rules.items():
        lines.append(f"### {model_tag}\n")
        lines.append("```text")
        lines.append(text.rstrip())
        lines.append("```")

    lines.append("\n## RQ3: Probe Group Pairwise Tests\n")
    for _, row in probe_pairwise.iterrows():
        lines.append(
            f"- {row.model_tag}: {row.left_group} vs {row.right_group}, "
            f"mean_diff={row.mean_diff_left_minus_right:.6f}, p={row.p_value:.4f}."
        )

    (SUMMARY_DIR / "selection_and_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    spectral = load_many("spectral/spectral_*.csv")
    approx = load_many("approx/approx_*.csv")
    multi_probe = sorted((RESULTS_DIR / "probe").glob("probe_*_multi.csv"))
    if multi_probe:
        probe = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in multi_probe], ignore_index=True)
    else:
        probe = load_many("probe/probe_*_TR01.csv")

    if spectral.empty or approx.empty or probe.empty:
        raise SystemExit("Missing spectral, approx, or probe data.")

    selection_df = build_selection_dataset(spectral, approx)
    selection_summary, importance, rules = selection_models(selection_df)
    probe_pairwise = probe_tests(probe)
    write_report(selection_summary, importance, rules, probe_pairwise)

    print(f"Wrote selection/statistical analysis to {SUMMARY_DIR}")
    print("\nSelection model summary:")
    print(selection_summary.to_string(index=False))
    print("\nProbe pairwise tests:")
    print(probe_pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
