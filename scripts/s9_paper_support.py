"""
s9_paper_support.py — Paper-support tables for final analysis.

Usage:
    python s9_paper_support.py

This script does not run model inference. It reuses completed CSV outputs to
produce three submission-oriented artifacts:

1. Sparse retention curve summaries for sparse_topk at 5/10/20/40%.
2. Qwen reversal case table, focusing on low_rank_diffuse heads that are most
   sensitive to banded-local causal intervention.
3. Discussion notes linking the empirical findings to efficient attention,
   pruning/sparsity, and structured matrix approximation.

Writes:
    results/summary/sparse_retention_summary.csv
    results/summary/sparse_retention_by_domain.csv
    results/summary/qwen_reversal_cases.csv
    results/summary/paper_discussion_notes.md
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_many(pattern: str) -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in files], ignore_index=True)


def sparse_retention(approx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sparse = approx[approx["method"] == "sparse_topk"].copy()
    sparse["retention_pct"] = sparse["param"].astype(int)

    summary = (
        sparse.groupby(["model_tag", "retention_pct"], as_index=False)
        .agg(
            rows=("frob_rel", "size"),
            frob_mean=("frob_rel", "mean"),
            frob_median=("frob_rel", "median"),
            frob_p90=("frob_rel", lambda s: s.quantile(0.90)),
            kl_mean=("kl_mean", "mean"),
            kl_median=("kl_mean", "median"),
            kl_p90=("kl_mean", lambda s: s.quantile(0.90)),
        )
        .round(6)
        .sort_values(["model_tag", "retention_pct"])
    )
    summary.to_csv(SUMMARY_DIR / "sparse_retention_summary.csv", index=False, encoding="utf-8-sig")

    by_domain = (
        sparse.groupby(["model_tag", "domain", "retention_pct"], as_index=False)
        .agg(
            rows=("frob_rel", "size"),
            frob_mean=("frob_rel", "mean"),
            frob_median=("frob_rel", "median"),
            kl_mean=("kl_mean", "mean"),
            kl_median=("kl_mean", "median"),
        )
        .round(6)
        .sort_values(["model_tag", "domain", "retention_pct"])
    )
    by_domain.to_csv(SUMMARY_DIR / "sparse_retention_by_domain.csv", index=False, encoding="utf-8-sig")
    return summary, by_domain


def qwen_reversal_cases() -> pd.DataFrame:
    path = RESULTS_DIR / "probe" / "probe_qwen3_8b_multi.csv"
    if not path.exists():
        raise SystemExit(f"Missing Qwen multi-probe file: {path}")

    probe = pd.read_csv(path, encoding="utf-8-sig")
    for col in ["source_erank", "source_diag_conc", "source_mean_dist", "kl_div", "js_div", "top10_overlap"]:
        probe[col] = pd.to_numeric(probe[col], errors="coerce")

    cases = (
        probe[probe["target_group"] == "low_rank_diffuse"]
        .sort_values(["kl_div", "js_div"], ascending=False)
        .head(30)
        .copy()
    )
    cases = cases[
        [
            "task_id",
            "model_tag",
            "target_group",
            "layer",
            "head",
            "selection_rank",
            "source_erank",
            "source_diag_conc",
            "source_mean_dist",
            "kl_div",
            "js_div",
            "top10_overlap",
        ]
    ]
    cases.to_csv(SUMMARY_DIR / "qwen_reversal_cases.csv", index=False, encoding="utf-8-sig")
    return cases


def write_discussion_notes(sparse_summary: pd.DataFrame, qwen_cases: pd.DataFrame):
    qwen_top = qwen_cases.head(8)
    sparse_lines = []
    for model_tag, sub in sparse_summary.groupby("model_tag"):
        compact = ", ".join(
            f"{int(row.retention_pct)}%: Fro={row.frob_mean:.4f}, KL={row.kl_mean:.4f}"
            for _, row in sub.sort_values("retention_pct").iterrows()
        )
        sparse_lines.append(f"- {model_tag}: {compact}")

    lines = []
    lines.append("# Paper Discussion Notes\n")
    lines.append("## Sparse Retention Curve\n")
    lines.append(
        "The completed approximation benchmark already contains a sparse retention sweep "
        "at 5%, 10%, 20%, and 40% retained entries per row. This provides a direct "
        "retention-error curve without rerunning inference."
    )
    lines.extend(sparse_lines)
    lines.append(
        "\nInterpretation: sparse_topk is not merely a strong single setting; it remains "
        "the dominant oracle family across retention budgets in the tested range. This "
        "supports the claim that real LLM attention distributions are highly compressible "
        "by retaining a small set of row-wise dominant entries."
    )

    lines.append("\n## Qwen Reversal Cases\n")
    lines.append(
        "The strengthened multi-prompt probe shows that Qwen3-8B differs from Ministral: "
        "its low_rank_diffuse target heads are often more sensitive to banded-local "
        "intervention than the high_complex_global group. This should not be described "
        "as a failure of low-rank approximation. The causal intervention in S3 is a "
        "banded locality constraint, so the more precise interpretation is: some Qwen "
        "low-rank diffuse heads appear to implement long-range aggregation, and forcing "
        "them into a local band disrupts that role."
    )
    lines.append("\nTop Qwen low_rank_diffuse reversal cases:")
    for _, row in qwen_top.iterrows():
        lines.append(
            f"- {row['task_id']} L{int(row['layer'])}H{int(row['head'])}: "
            f"erank={row['source_erank']:.3f}, diag={row['source_diag_conc']:.3f}, "
            f"mean_dist={row['source_mean_dist']:.3f}, KL={row['kl_div']:.6f}, "
            f"top10_overlap={row['top10_overlap']:.2f}."
        )

    lines.append("\n## Relation to Efficient Attention and Sparsity Work\n")
    lines.append(
        "- FlashAttention-style work optimizes exact dense attention through IO-aware kernels. "
        "Our results ask an orthogonal question: what structure is present in the attention "
        "matrix itself, and which approximating family preserves it best?"
    )
    lines.append(
        "- Pruning/sparsity work such as SparseGPT motivates the broader principle that large "
        "models contain exploitable redundancy. Our sparse_topk results extend that intuition "
        "from model weights to per-input attention distributions."
    )
    lines.append(
        "- Monarch-style structured matrices are hardware-friendly, but the empirical landscape "
        "shows they are not uniformly optimal. Their advantage appears conditional on spectral "
        "and locality features, while low-rank is the safer default among non-sparse dense "
        "families."
    )

    lines.append("\n## Wording Cautions\n")
    lines.append(
        "- State clearly that the S3 causal probe uses a banded local constraint "
        f"(band_width=5), not sparse_topk and not low-rank intervention."
    )
    lines.append(
        "- Avoid saying low erank means local. In these results, very low-erank heads can be "
        "diffuse and long-distance. Use the term low_rank_diffuse for that class."
    )
    lines.append(
        "- Treat sparse_topk as an oracle approximation family unless or until a hardware-aware "
        "sparse kernel is implemented. The current benchmark measures matrix fidelity, not "
        "wall-clock acceleration."
    )

    (SUMMARY_DIR / "paper_discussion_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    approx = load_many("approx/approx_*.csv")
    if approx.empty:
        raise SystemExit("Missing approximation CSV files.")

    sparse_summary, _ = sparse_retention(approx)
    qwen_cases = qwen_reversal_cases()
    write_discussion_notes(sparse_summary, qwen_cases)

    print(f"Wrote paper-support outputs to {SUMMARY_DIR}")
    print("\nSparse retention summary:")
    print(sparse_summary.to_string(index=False))
    print("\nTop Qwen reversal cases:")
    print(qwen_cases.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
