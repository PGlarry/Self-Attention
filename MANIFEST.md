# Repository Manifest

## Prompts

- `prompts/*__FULL.txt`: full prompt files used in all experiments.

## Scripts

- `scripts/s1_extract_spectra.py`: extracts attention matrices through eager attention and computes spectral/locality metrics.
- `scripts/s2_approximate.py`: benchmarks sparse top-k, low-rank-initialized projected, banded, and block-diagonal proxy approximations.
- `scripts/s3_causal_probe.py`: performs banded-local and optional projected low-rank (PLR) single-head interventions and records output distribution shifts.
- `scripts/s4_figures.py`: generates all paper figures from CSV results.
- `scripts/s5_select_probe_targets.py`: selects high-complexity global, local-band, and low-rank diffuse probe heads.
- `scripts/s5_review_response_stats.py`: produces reviewer-response robustness statistics from existing CSVs.
- `scripts/s6_analyze_results.py`: summarizes core spectral, approximation, and probe results.
- `scripts/s7_selection_and_stats.py`: trains structure-selection models and computes probe pairwise tests.
- `scripts/s8_enhance_rq2_selection.py`: runs non-sparse family selection analysis.
- `scripts/s9_paper_support.py`: generates sparse-retention, Qwen reversal, and discussion-support tables.
- `scripts/s10_selector_robustness.py`: adds selector no-length ablation, leave-one-prompt-out evaluation, per-class metrics, confusion matrices, and Frobenius/KL winner agreement.
- `scripts/s11_review_hardening_stats.py`: adds clustered probe statistics, exact head-label permutation tests, leave-one-domain/model-out selector checks, length-control regression, and artifact hashes.

## Results

- `results/spectral/`: per-model spectral matrices summarized by prompt, layer, and head.
- `results/approx/`: per-model approximation benchmark results.
- `results/probe/`: causal probe outputs and selected target heads.
- `results/summary/`: aggregated analysis tables, paper-support files, robustness checks, and supplemental Qwen3 PLR probe summaries.

## Figures

- `figures/fig1_erank_heatmap.png`
- `figures/fig2_locality_profile.png`
- `figures/fig3_approx_boxplot_*.png`
- `figures/fig4_probe_scatter_*.png`
- `figures/fig5_probe_group_boxplot.png`
- `figures/fig6_sparse_retention_curve.png`

## Public Reproducibility Metadata

- `artifact_manifest.json`: public file hashes, model revisions, environment versions, seeds, and numerical constants.
- Manuscript drafts, expert review reports, and planning notes are intentionally excluded from the public repository.

