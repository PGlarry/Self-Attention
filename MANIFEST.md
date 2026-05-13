# Repository Manifest

## Prompts

- `prompts/*__FULL.txt`: full prompt files used in all experiments.

## Scripts

- `scripts/s1_extract_spectra.py`: extracts attention matrices through eager attention and computes spectral/locality metrics.
- `scripts/s2_approximate.py`: benchmarks sparse top-k, low-rank-initialized projected, banded, and block-diagonal proxy approximations.
- `scripts/s3_causal_probe.py`: performs banded-local single-head interventions and records output distribution shifts.
- `scripts/s4_figures.py`: generates all paper figures from CSV results.
- `scripts/s5_select_probe_targets.py`: selects high-complexity global, local-band, and low-rank diffuse probe heads.
- `scripts/s5_review_response_stats.py`: produces reviewer-response robustness statistics from existing CSVs.
- `scripts/s6_analyze_results.py`: summarizes core spectral, approximation, and probe results.
- `scripts/s7_selection_and_stats.py`: trains structure-selection models and computes probe pairwise tests.
- `scripts/s8_enhance_rq2_selection.py`: runs non-sparse family selection analysis.
- `scripts/s9_paper_support.py`: generates sparse-retention, Qwen reversal, and discussion-support tables.

## Results

- `results/spectral/`: per-model spectral matrices summarized by prompt, layer, and head.
- `results/approx/`: per-model approximation benchmark results.
- `results/probe/`: causal probe outputs and selected target heads.
- `results/summary/`: aggregated analysis tables, paper-support files, and robustness checks.

## Figures

- `figures/fig1_erank_heatmap.png`
- `figures/fig2_locality_profile.png`
- `figures/fig3_approx_boxplot_*.png`
- `figures/fig4_probe_scatter_*.png`
- `figures/fig5_probe_group_boxplot.png`
- `figures/fig6_sparse_retention_curve.png`

## Paper and Review Materials

- `paper/main_draft.md`: main paper source draft.
- `paper/references.bib`: BibTeX references.
- `paper/build_docx.py`: DOCX generator.
- `paper/PPS_selfatten_working_draft.docx`: current Word draft.
- `paper/PPS_selfatten_working_draft.pdf`: current PDF draft.
- `分析/`: expert review reports and improvement plan.
- `研究方案讨论/`: early research planning notes.

