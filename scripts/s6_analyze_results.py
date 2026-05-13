"""
s6_analyze_results.py — Summarize spectral, approximation, and probe results.

Usage:
    python s6_analyze_results.py

Reads:
    results/spectral/*.csv
    results/approx/*.csv
    results/probe/probe_*.csv

Writes:
    results/summary/model_domain_erank.csv
    results/summary/approx_method_summary.csv
    results/summary/best_method_counts.csv
    results/summary/probe_group_summary.csv
    results/summary/probe_metric_correlations.csv
    results/summary/report.md
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_many(pattern: str) -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob(pattern))
    files = [path for path in files if path.name != "probe_targets.csv"]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in files], ignore_index=True)


def summarize_spectral(spectral: pd.DataFrame) -> pd.DataFrame:
    summary = (
        spectral.groupby(["model_tag", "domain"], as_index=False)
        .agg(
            rows=("erank", "size"),
            erank_mean=("erank", "mean"),
            erank_median=("erank", "median"),
            stable_rank_mean=("stable_rank", "mean"),
            diag_conc_mean=("diag_conc", "mean"),
            mean_dist_mean=("mean_dist", "mean"),
            seq_len_mean=("seq_len", "mean"),
        )
        .round(6)
    )
    summary.to_csv(SUMMARY_DIR / "model_domain_erank.csv", index=False, encoding="utf-8-sig")
    return summary


def summarize_approx(approx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_summary = (
        approx.groupby(["model_tag", "method"], as_index=False)
        .agg(
            rows=("frob_rel", "size"),
            frob_mean=("frob_rel", "mean"),
            frob_median=("frob_rel", "median"),
            kl_mean=("kl_mean", "mean"),
            kl_median=("kl_mean", "median"),
        )
        .round(6)
    )
    method_summary.to_csv(SUMMARY_DIR / "approx_method_summary.csv", index=False, encoding="utf-8-sig")

    idx_cols = ["model_tag", "task_id", "layer", "head", "seq_len"]
    best = approx.loc[approx.groupby(idx_cols)["frob_rel"].idxmin()].copy()
    counts = (
        best.groupby(["model_tag", "method"], as_index=False)
        .agg(best_count=("method", "size"))
    )
    totals = counts.groupby("model_tag")["best_count"].transform("sum")
    counts["best_fraction"] = (counts["best_count"] / totals).round(6)
    counts.to_csv(SUMMARY_DIR / "best_method_counts.csv", index=False, encoding="utf-8-sig")
    return method_summary, counts


def summarize_probe(probe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = ["source_erank", "source_diag_conc", "source_mean_dist", "kl_div"]
    for col in numeric_cols:
        if col in probe.columns:
            probe[col] = pd.to_numeric(probe[col], errors="coerce")

    group_summary = (
        probe.groupby(["model_tag", "target_group"], as_index=False)
        .agg(
            rows=("kl_div", "size"),
            fired=("intervention_fired", "sum"),
            kl_mean=("kl_div", "mean"),
            kl_median=("kl_div", "median"),
            js_mean=("js_div", "mean"),
            top10_overlap_mean=("top10_overlap", "mean"),
            source_erank_mean=("source_erank", "mean"),
            source_diag_conc_mean=("source_diag_conc", "mean"),
            source_mean_dist_mean=("source_mean_dist", "mean"),
        )
        .round(6)
    )
    group_summary.to_csv(SUMMARY_DIR / "probe_group_summary.csv", index=False, encoding="utf-8-sig")

    corr_rows = []
    for model_tag, sub in probe.groupby("model_tag"):
        for metric in ["source_erank", "source_diag_conc", "source_mean_dist"]:
            corr_rows.append({
                "model_tag": model_tag,
                "metric": metric,
                "corr_with_kl": sub[[metric, "kl_div"]].corr().iloc[0, 1],
            })
    corr = pd.DataFrame(corr_rows).round(6)
    corr.to_csv(SUMMARY_DIR / "probe_metric_correlations.csv", index=False, encoding="utf-8-sig")
    return group_summary, corr


def write_report(spectral_summary: pd.DataFrame, approx_summary: pd.DataFrame,
                 best_counts: pd.DataFrame, probe_summary: pd.DataFrame,
                 probe_corr: pd.DataFrame):
    report = []
    report.append("# PPS-selfatten Result Summary\n")
    report.append("## RQ1: Spectral Structure\n")
    for _, row in spectral_summary.iterrows():
        report.append(
            f"- {row.model_tag} / {row.domain}: mean erank={row.erank_mean:.3f}, "
            f"mean diag_conc={row.diag_conc_mean:.3f}, mean_dist={row.mean_dist_mean:.3f}."
        )

    report.append("\n## RQ2: Structured Approximation Benchmark\n")
    for model_tag, sub in approx_summary.groupby("model_tag"):
        ordered = sub.sort_values("frob_mean")
        best = ordered.iloc[0]
        report.append(
            f"- {model_tag}: lowest mean Frobenius error is {best.method} "
            f"(mean={best.frob_mean:.4f})."
        )
    report.append("\nBest-method counts by head/task are stored in `best_method_counts.csv`.")
    selection_report = SUMMARY_DIR / "selection_and_stats.md"
    if selection_report.exists():
        report.append("Zero-shot selection diagnostics are stored in `selection_and_stats.md`.")
    non_sparse_report = SUMMARY_DIR / "non_sparse_selection_rules.md"
    if non_sparse_report.exists():
        report.append("Non-sparse structure-family selection diagnostics are stored in `non_sparse_selection_rules.md`.")

    report.append("\n## RQ3: Structured Causal Probe\n")
    for _, row in probe_summary.iterrows():
        report.append(
            f"- {row.model_tag} / {row.target_group}: n={int(row.rows)}, fired={int(row.fired)}, "
            f"mean KL={row.kl_mean:.6f}, median KL={row.kl_median:.6f}, "
            f"top10 overlap={row.top10_overlap_mean:.3f}."
        )
    report.append("\nProbe metric correlations are stored in `probe_metric_correlations.csv`.")
    pairwise_tests = SUMMARY_DIR / "probe_pairwise_tests.csv"
    if pairwise_tests.exists():
        report.append("Pairwise probe group tests are stored in `probe_pairwise_tests.csv`.")

    (SUMMARY_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    spectral = load_many("spectral/spectral_*.csv")
    approx = load_many("approx/approx_*.csv")
    multi_probe = sorted((RESULTS_DIR / "probe").glob("probe_*_multi.csv"))
    if multi_probe:
        probe = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in multi_probe], ignore_index=True)
    else:
        probe = load_many("probe/probe_*_TR01.csv")

    if spectral.empty or approx.empty or probe.empty:
        raise SystemExit("Missing one or more required result groups.")

    spectral_summary = summarize_spectral(spectral)
    approx_summary, best_counts = summarize_approx(approx)
    probe_summary, probe_corr = summarize_probe(probe)
    write_report(spectral_summary, approx_summary, best_counts, probe_summary, probe_corr)

    print(f"Wrote summaries to {SUMMARY_DIR}")
    print("\nProbe group summary:")
    print(probe_summary.to_string(index=False))


if __name__ == "__main__":
    main()
