"""
s11_review_hardening_stats.py — low-cost reviewer-hardening analyses.

This script performs only CSV post-processing. It does not load models, run
forward passes, or touch private paper/review directories.

Analyses:
1. Cluster bootstrap for probe KL using selected heads as clusters.
2. Exact head-label permutation tests for probe group differences.
3. Leave-one-domain-out and leave-one-model-out selector robustness.
4. Length-control regressions for erank and mean attention distance.
5. Public artifact manifest with file hashes for reproducibility.

Outputs are written under results/summary/ plus artifact_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260513)
N_BOOT = 10000
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


def load_probe_multi() -> pd.DataFrame:
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in sorted((RESULTS / "probe").glob("probe_*_multi.csv"))]
    if not frames:
        raise FileNotFoundError("Missing results/probe/*multi.csv")
    return pd.concat(frames, ignore_index=True)


def load_selector() -> pd.DataFrame:
    path = SUMMARY / "non_sparse_selection_dataset.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    raise FileNotFoundError(path)


def load_spectral() -> pd.DataFrame:
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in sorted((RESULTS / "spectral").glob("spectral_*.csv"))]
    if not frames:
        raise FileNotFoundError("Missing results/spectral/spectral_*.csv")
    return pd.concat(frames, ignore_index=True)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = 0
    lt = 0
    for value in x:
        gt += np.sum(value > y)
        lt += np.sum(value < y)
    return float((gt - lt) / (len(x) * len(y))) if len(x) and len(y) else np.nan


def head_level_probe(probe: pd.DataFrame) -> pd.DataFrame:
    return (
        probe.groupby(["model_tag", "target_group", "layer", "head"], as_index=False)
        .agg(
            n_prompts=("task_id", "nunique"),
            kl_head_mean=("kl_div", "mean"),
            kl_head_median=("kl_div", "median"),
            js_head_mean=("js_div", "mean"),
            top10_head_mean=("top10_overlap", "mean"),
        )
    )


def cluster_bootstrap_probe(probe: pd.DataFrame) -> pd.DataFrame:
    head = head_level_probe(probe)
    rows = []
    for (model_tag, group), sub in head.groupby(["model_tag", "target_group"]):
        values = sub["kl_head_median"].to_numpy(float)
        means = np.empty(N_BOOT)
        medians = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sample = RNG.choice(values, size=len(values), replace=True)
            means[i] = np.mean(sample)
            medians[i] = np.median(sample)
        rows.append(
            {
                "model_tag": model_tag,
                "target_group": group,
                "n_head_clusters": len(values),
                "mean_of_head_medians": float(np.mean(values)),
                "mean_ci95_low": float(np.percentile(means, 2.5)),
                "mean_ci95_high": float(np.percentile(means, 97.5)),
                "median_of_head_medians": float(np.median(values)),
                "median_ci95_low": float(np.percentile(medians, 2.5)),
                "median_ci95_high": float(np.percentile(medians, 97.5)),
            }
        )
    out = pd.DataFrame(rows).round(6)
    out.to_csv(SUMMARY / "probe_cluster_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    return out


def exact_partition_indices(n: int, k: int):
    # Small n in this project (6+6 heads), so exact enumeration is cheap.
    from itertools import combinations as iter_combinations

    return iter_combinations(range(n), k)


def permutation_probe_tests(probe: pd.DataFrame) -> pd.DataFrame:
    head = head_level_probe(probe)
    rows = []
    for model_tag, model_df in head.groupby("model_tag"):
        for left, right in combinations(sorted(model_df["target_group"].unique()), 2):
            x = model_df[model_df["target_group"] == left]["kl_head_median"].to_numpy(float)
            y = model_df[model_df["target_group"] == right]["kl_head_median"].to_numpy(float)
            combined = np.concatenate([x, y])
            n_left = len(x)
            obs_mean = float(np.mean(x) - np.mean(y))
            obs_median = float(np.median(x) - np.median(y))
            mean_extreme = 0
            median_extreme = 0
            total = 0
            for idx_tuple in exact_partition_indices(len(combined), n_left):
                mask = np.zeros(len(combined), dtype=bool)
                mask[list(idx_tuple)] = True
                px = combined[mask]
                py = combined[~mask]
                mean_diff = float(np.mean(px) - np.mean(py))
                median_diff = float(np.median(px) - np.median(py))
                mean_extreme += abs(mean_diff) >= abs(obs_mean) - 1e-15
                median_extreme += abs(median_diff) >= abs(obs_median) - 1e-15
                total += 1
            rows.append(
                {
                    "model_tag": model_tag,
                    "left_group": left,
                    "right_group": right,
                    "n_left_heads": len(x),
                    "n_right_heads": len(y),
                    "observed_mean_diff": obs_mean,
                    "exact_perm_p_mean_diff": mean_extreme / total,
                    "observed_median_diff": obs_median,
                    "exact_perm_p_median_diff": median_extreme / total,
                    "cliffs_delta": cliffs_delta(x, y),
                }
            )
    out = pd.DataFrame(rows).round(6)
    out.to_csv(SUMMARY / "probe_head_permutation_tests.csv", index=False, encoding="utf-8-sig")
    return out


def make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def selector_metrics(model_tag: str, protocol: str, feature_set: str, y_true, y_pred, train_majority: str, labels: list[str]):
    y_true = pd.Series(y_true)
    y_pred = np.asarray(y_pred)
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


def selector_per_class(model_tag: str, protocol: str, feature_set: str, y_true, y_pred, labels: list[str]) -> list[dict]:
    y_true = pd.Series(y_true)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return [
        {
            "model_tag": model_tag,
            "protocol": protocol,
            "feature_set": feature_set,
            "class": label,
            "support": int((y_true == label).sum()),
            "recall": float(recall),
        }
        for label, recall in zip(labels, recalls)
    ]


def selector_confusion(model_tag: str, protocol: str, feature_set: str, y_true, y_pred, labels: list[str]) -> list[dict]:
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


def eval_holdout(train: pd.DataFrame, test: pd.DataFrame, labels: list[str], features: list[str]):
    clf = make_classifier()
    clf.fit(train[features], train["best_non_sparse_method"])
    pred = clf.predict(test[features])
    majority = train["best_non_sparse_method"].value_counts().idxmax()
    return pred, majority


def selector_domain_model_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    pc_rows = []
    cm_rows = []
    label_all = sorted(df["best_non_sparse_method"].unique())

    for feature_set, features in [("full", FEATURES_FULL), ("no_seq_len", FEATURES_NO_LEN)]:
        # Leave-one-domain-out across all models.
        for domain in sorted(df["domain"].unique()):
            train = df[df["domain"] != domain]
            test = df[df["domain"] == domain]
            pred, majority = eval_holdout(train, test, label_all, features)
            protocol = f"leave_one_domain::{domain}"
            rows.append(selector_metrics("ALL", protocol, feature_set, test["best_non_sparse_method"], pred, majority, label_all))
            pc_rows.extend(selector_per_class("ALL", protocol, feature_set, test["best_non_sparse_method"], pred, label_all))
            cm_rows.extend(selector_confusion("ALL", protocol, feature_set, test["best_non_sparse_method"], pred, label_all))

        # Leave-one-domain-out within each model.
        for model_tag, sub in df.groupby("model_tag"):
            labels = sorted(sub["best_non_sparse_method"].unique())
            for domain in sorted(sub["domain"].unique()):
                train = sub[sub["domain"] != domain]
                test = sub[sub["domain"] == domain]
                pred, majority = eval_holdout(train, test, labels, features)
                protocol = f"leave_one_domain::{domain}"
                rows.append(selector_metrics(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, majority, labels))
                pc_rows.extend(selector_per_class(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, labels))
                cm_rows.extend(selector_confusion(model_tag, protocol, feature_set, test["best_non_sparse_method"], pred, labels))

        # Leave-one-model-out across architectures.
        for held_model in sorted(df["model_tag"].unique()):
            train = df[df["model_tag"] != held_model]
            test = df[df["model_tag"] == held_model]
            pred, majority = eval_holdout(train, test, label_all, features)
            protocol = f"leave_one_model::{held_model}"
            rows.append(selector_metrics("ALL", protocol, feature_set, test["best_non_sparse_method"], pred, majority, label_all))
            pc_rows.extend(selector_per_class("ALL", protocol, feature_set, test["best_non_sparse_method"], pred, label_all))
            cm_rows.extend(selector_confusion("ALL", protocol, feature_set, test["best_non_sparse_method"], pred, label_all))

    summary = pd.DataFrame(rows).round(6)
    per_class = pd.DataFrame(pc_rows).round(6)
    confusion = pd.DataFrame(cm_rows)
    summary.to_csv(SUMMARY / "selector_domain_model_holdout_summary.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(SUMMARY / "selector_domain_model_holdout_per_class.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(SUMMARY / "selector_domain_model_holdout_confusion.csv", index=False, encoding="utf-8-sig")
    return summary, per_class, confusion


def ols_table(df: pd.DataFrame, outcome: str, subset_name: str) -> pd.DataFrame:
    work = df.copy()
    work["log_seq_len"] = np.log(work["seq_len"].astype(float))
    x_parts = [pd.Series(1.0, index=work.index, name="Intercept"), work[["log_seq_len"]]]
    if work["domain"].nunique() > 1:
        x_parts.append(pd.get_dummies(work["domain"], prefix="domain", drop_first=True, dtype=float))
    if work["model_tag"].nunique() > 1:
        x_parts.append(pd.get_dummies(work["model_tag"], prefix="model", drop_first=True, dtype=float))
    X_df = pd.concat(x_parts, axis=1)
    X = X_df.to_numpy(float)
    y = work[outcome].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    n, p = X.shape
    df_resid = max(n - p, 1)
    sigma2 = float((resid @ resid) / df_resid)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_values = beta / se
    p_values = 2 * t_dist.sf(np.abs(t_values), df_resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot else np.nan
    return pd.DataFrame(
        {
            "subset": subset_name,
            "outcome": outcome,
            "term": X_df.columns,
            "estimate": beta,
            "std_error": se,
            "t_value": t_values,
            "p_value": p_values,
            "n": n,
            "r2": r2,
        }
    )


def length_control_regressions(spectral: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["erank", "mean_dist"]:
        rows.append(ols_table(spectral, outcome, "ALL"))
        for model_tag, sub in spectral.groupby("model_tag"):
            rows.append(ols_table(sub, outcome, model_tag))
    out = pd.concat(rows, ignore_index=True).round(6)
    out.to_csv(SUMMARY / "spectral_length_control_regression.csv", index=False, encoding="utf-8-sig")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_artifact_manifest() -> dict:
    include_dirs = ["prompts", "scripts", "results", "figures"]
    files = []
    for dirname in include_dirs:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    manifest = {
        "project": "Self-Attention structured matrix approximation study",
        "public_artifact_scope": include_dirs,
        "private_excluded_paths": ["paper/", "分析/", "研究方案讨论/"],
        "models": [
            {"tag": "qwen3_8b", "hf_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
            {"tag": "gemma3_4b", "hf_id": "google/gemma-3-4b-it", "revision": "093f9f388b31de276ce2de164bdc2081324b9767"},
            {"tag": "ministral8b", "hf_id": "mistralai/Ministral-8B-Instruct-2410", "revision": "2f494a194c5b980dfb9772cb92d26cbb671fce5a"},
        ],
        "environment_observed": {
            "python": "3.11.9",
            "torch": "2.6.0+cu124",
            "transformers": "5.8.0",
            "bitsandbytes": "0.49.2",
            "numpy": "1.26.4",
            "pandas": "2.1.4",
            "scikit_learn": "1.8.0",
            "scipy": "1.16.2",
        },
        "random_seeds": {
            "sklearn_random_state": 42,
            "bootstrap_rng": 20260513,
        },
        "numerical_constants": {
            "kl_epsilon": 1e-12,
            "probe_band_half_width": 5,
            "supplemental_qwen_plr_ranks": [2, 4],
            "main_supplemental_qwen_plr_rank": 4,
        },
        "files": files,
    }
    out = ROOT / "artifact_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_markdown(
    cluster_ci: pd.DataFrame,
    perm: pd.DataFrame,
    selector_summary: pd.DataFrame,
    regression: pd.DataFrame,
    manifest: dict,
) -> None:
    lines = []
    lines.append("# Review-Hardening Post-Hoc Analyses\n")
    lines.append("All analyses in this file reuse completed CSV outputs only. No model inference is run.\n")
    lines.append("## Probe Cluster Bootstrap by Head\n")
    lines.append(cluster_ci.to_markdown(index=False))
    lines.append("\n## Probe Exact Head-Label Permutation Tests\n")
    lines.append(perm.to_markdown(index=False))
    lines.append("\n## Selector Leave-One-Domain / Leave-One-Model Holdout\n")
    keep_protocols = selector_summary[
        selector_summary["protocol"].str.startswith("leave_one_model")
        | ((selector_summary["model_tag"] == "ALL") & selector_summary["protocol"].str.startswith("leave_one_domain"))
    ]
    lines.append(keep_protocols.to_markdown(index=False))
    lines.append("\n## Length-Control Regression Key Terms\n")
    key_terms = regression[
        regression["term"].isin(["log_seq_len", "domain_technical", "domain_travel"])
    ].copy()
    lines.append(key_terms.to_markdown(index=False))
    lines.append("\n## Artifact Manifest\n")
    lines.append(f"- Manifest file: `artifact_manifest.json`")
    lines.append(f"- Public files hashed: {len(manifest['files'])}")
    lines.append("- Private manuscript/review/planning paths remain excluded: `paper/`, `分析/`, `研究方案讨论/`.")
    (SUMMARY / "review_hardening_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    probe = load_probe_multi()
    selector = load_selector()
    spectral = load_spectral()

    cluster_ci = cluster_bootstrap_probe(probe)
    perm = permutation_probe_tests(probe)
    selector_summary, _, _ = selector_domain_model_holdout(selector)
    regression = length_control_regressions(spectral)
    manifest = write_artifact_manifest()
    write_markdown(cluster_ci, perm, selector_summary, regression, manifest)

    print("Wrote review-hardening outputs to", SUMMARY)
    print("\nSelector leave-one-model-out:")
    print(selector_summary[selector_summary["protocol"].str.startswith("leave_one_model")].to_string(index=False))
    print("\nProbe head-label permutation:")
    print(perm.to_string(index=False))


if __name__ == "__main__":
    main()
