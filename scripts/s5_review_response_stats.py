"""
s5_review_response_stats.py

Post-hoc robustness analyses requested by reviewers. This script does not run
models or create new attention matrices. It only reads existing CSV outputs and
writes additional summary tables for the paper appendix.

Outputs:
  results/summary/probe_head_level_summary.csv
  results/summary/probe_head_level_pairwise.csv
  results/summary/probe_group_bootstrap_ci.csv
  results/summary/spectral_normalized_distance.csv
  results/summary/review_response_stats.md
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260513)
N_BOOT = 10000


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x > y) - P(x < y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for value in x:
        gt += np.sum(value > y)
        lt += np.sum(value < y)
    return (gt - lt) / (len(x) * len(y))


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adjusted[order[i]] = val
        prev = val
    return adjusted.tolist()


def bootstrap_ci(values: np.ndarray, stat: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    boot = []
    for _ in range(N_BOOT):
        sample = RNG.choice(values, size=len(values), replace=True)
        if stat == "mean":
            boot.append(float(np.mean(sample)))
        elif stat == "median":
            boot.append(float(np.median(sample)))
        else:
            raise ValueError(stat)
    return tuple(np.percentile(boot, [2.5, 97.5]).tolist())


def load_probe_multi() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "probe").glob("probe_*_multi.csv")):
        frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not frames:
        raise FileNotFoundError("No *_multi probe CSV files found.")
    return pd.concat(frames, ignore_index=True)


def probe_robust_stats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe = load_probe_multi()

    # Aggregate repeated prompt observations within each selected head. This
    # reduces the pseudo-replication concern from 6 heads x 6 prompts.
    head_level = (
        probe.groupby(["model_tag", "target_group", "layer", "head"], as_index=False)
        .agg(
            n_prompts=("task_id", "nunique"),
            kl_mean=("kl_div", "mean"),
            kl_median=("kl_div", "median"),
            js_mean=("js_div", "mean"),
            top10_overlap_mean=("top10_overlap", "mean"),
            source_erank=("source_erank", "mean"),
            source_diag_conc=("source_diag_conc", "mean"),
            source_mean_dist=("source_mean_dist", "mean"),
        )
    )
    head_level.to_csv(SUMMARY / "probe_head_level_summary.csv", index=False, encoding="utf-8-sig")

    pair_rows = []
    for model_tag, model_df in head_level.groupby("model_tag"):
        groups = sorted(model_df["target_group"].unique())
        p_values = []
        row_start = len(pair_rows)
        for left, right in combinations(groups, 2):
            x = model_df[model_df["target_group"] == left]["kl_median"].to_numpy()
            y = model_df[model_df["target_group"] == right]["kl_median"].to_numpy()
            test = mannwhitneyu(x, y, alternative="two-sided")
            p_values.append(float(test.pvalue))
            pair_rows.append(
                {
                    "model_tag": model_tag,
                    "left_group": left,
                    "right_group": right,
                    "n_left_heads": len(x),
                    "n_right_heads": len(y),
                    "left_head_median_kl_mean": float(np.mean(x)),
                    "right_head_median_kl_mean": float(np.mean(y)),
                    "mean_diff_left_minus_right": float(np.mean(x) - np.mean(y)),
                    "mannwhitney_u": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "cliffs_delta": float(cliffs_delta(x, y)),
                }
            )
        q_values = bh_fdr(p_values)
        for offset, q in enumerate(q_values):
            pair_rows[row_start + offset]["bh_fdr_q"] = q
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(SUMMARY / "probe_head_level_pairwise.csv", index=False, encoding="utf-8-sig")

    ci_rows = []
    for (model_tag, group), sub in probe.groupby(["model_tag", "target_group"]):
        values = sub["kl_div"].to_numpy()
        mean_lo, mean_hi = bootstrap_ci(values, "mean")
        med_lo, med_hi = bootstrap_ci(values, "median")
        ci_rows.append(
            {
                "model_tag": model_tag,
                "target_group": group,
                "n_interventions": len(values),
                "kl_mean": float(np.mean(values)),
                "kl_mean_ci95_low": mean_lo,
                "kl_mean_ci95_high": mean_hi,
                "kl_median": float(np.median(values)),
                "kl_median_ci95_low": med_lo,
                "kl_median_ci95_high": med_hi,
            }
        )
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(SUMMARY / "probe_group_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    return head_level, pairwise, ci


def spectral_length_normalization() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "spectral").glob("spectral_*.csv")):
        tag = path.stem.replace("spectral_", "")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["model_tag"] = tag
        frames.append(df)
    spectral = pd.concat(frames, ignore_index=True)
    spectral["norm_mean_dist"] = spectral["mean_dist"] / spectral["seq_len"]
    spectral["norm_erank"] = spectral["erank"] / spectral["seq_len"]
    out = (
        spectral.groupby(["model_tag", "domain"], as_index=False)
        .agg(
            rows=("erank", "size"),
            seq_len_mean=("seq_len", "mean"),
            erank_mean=("erank", "mean"),
            erank_per_token_mean=("norm_erank", "mean"),
            mean_dist_mean=("mean_dist", "mean"),
            mean_dist_per_token_mean=("norm_mean_dist", "mean"),
            diag_conc_mean=("diag_conc", "mean"),
        )
    )
    out.to_csv(SUMMARY / "spectral_normalized_distance.csv", index=False, encoding="utf-8-sig")
    return out


def write_markdown(head_level: pd.DataFrame, pairwise: pd.DataFrame, ci: pd.DataFrame, norm: pd.DataFrame) -> None:
    md = []
    md.append("# Reviewer-Response Robustness Summaries\n")
    md.append("This file is generated from existing experiment CSVs only; no model inference is run.\n")

    md.append("## Probe Bootstrap Confidence Intervals\n")
    md.append(ci.round(6).to_markdown(index=False))
    md.append("\n")

    md.append("## Head-Level Pairwise Tests\n")
    md.append(
        "KL is first aggregated within each selected head across prompts using the per-head median, "
        "then groups are compared. This reduces pseudo-replication from prompt-level repeated measures.\n"
    )
    md.append(pairwise.round(6).to_markdown(index=False))
    md.append("\n")

    md.append("## Length-Normalized Spectral Metrics\n")
    md.append(norm.round(6).to_markdown(index=False))
    md.append("\n")

    (SUMMARY / "review_response_stats.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    head_level, pairwise, ci = probe_robust_stats()
    norm = spectral_length_normalization()
    write_markdown(head_level, pairwise, ci, norm)
    print("Wrote reviewer-response summaries to", SUMMARY)


if __name__ == "__main__":
    main()
