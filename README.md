# Structured Self-Attention Analysis

This repository contains the public experimental materials, data, analysis scripts, and figures for:

**How Structured is Decoder Self-Attention? Spectral Landscapes, Oracle Approximation, and Structure-Constrained Probing**

The project studies full self-attention matrices from pretrained decoder LLMs through three linked questions:

1. What spectral and spatial structure appears in real layer-head attention matrices?
2. Which structured approximation families best preserve those matrices?
3. Can structure-constrained interventions reveal architecture-specific head sensitivity?

## Repository Contents

| Path | Contents |
|---|---|
| `prompts/` | 12 full instruction prompts across Travel, Business, and Technical domains |
| `scripts/` | Extraction, approximation, probe, selection, plotting, and reviewer-response analysis scripts |
| `results/spectral/` | Spectral and locality metrics for all extracted attention matrices |
| `results/approx/` | Structured approximation benchmark CSVs |
| `results/probe/` | Causal probe outputs and selected head targets |
| `results/summary/` | Aggregated tables, statistics, reviewer-response robustness summaries |
| `figures/` | Paper figures generated from the result CSVs |
| `artifact_manifest.json` | Public artifact hashes, environment notes, model revisions, and reproducibility constants |

Private manuscript, review, and planning materials are intentionally excluded from the public repository.

## Main Findings

- Sparse top-k is the strongest **oracle matrix-fidelity** family. Retaining 20% of entries per row reduces mean relative Frobenius error to roughly 0.032-0.035 across the tested models.
- When sparse top-k is excluded, the low-rank-initialized projected family is the safest dense fallback, winning 81.5%-96.3% of layer-head-prompt cases.
- Rank and locality are separable. Qwen3 contains low-rank diffuse heads with very low effective rank but long mean attention distance.
- A banded-local intervention reveals robust high-complexity global-head sensitivity in Ministral and more outlier-driven but interpretable patterns in Gemma3 and Qwen3.
- A supplemental Qwen3 projected low-rank (PLR) probe provides an orthogonal check: selected Qwen3 local-band heads are much more sensitive to PLR-r4 replacement than to banded-local replacement, while low-rank diffuse heads show the opposite high-KL pattern under banding rather than PLR.

## Approximation Code Labels

The released CSVs preserve the original experiment code labels:

| CSV/code label | Paper term | Meaning |
|---|---|---|
| `sparse_topk` | Sparse top-k oracle | Row-wise top-k retention baseline for matrix fidelity, not a runtime acceleration claim |
| `lowrank` | Projected low-rank / PLR | Truncated-SVD reconstruction followed by nonnegative projection and row renormalization |
| `banded` | Banded local mask | Local window mask followed by row renormalization |
| `monarch` | Simplified block-diagonal proxy | Legacy code label for a block-diagonal proxy; not full MonarchAttention or the full Monarch matrix design space |

## Supplemental Qwen3 PLR Probe

The reviewer-requested orthogonal PLR probe is released under `results/probe/` and `results/summary/`:

- `results/probe/probe_qwen3_8b_multi_plr_r4.csv`
- `results/probe/probe_qwen3_8b_multi_plr_r2.csv`
- `results/summary/qwen_plr_probe_summary.md`
- `results/summary/qwen_plr_r4_probe_comparison.csv`
- `results/summary/qwen_plr_r4_probe_group_summary.csv`
- `results/summary/qwen_lowrank_diffuse_plr_rank_sweep.csv`

## Reproducing the Analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run model-dependent experiments, replacing `<model_path>` with a local Hugging Face checkpoint path:

```bash
python scripts/s1_extract_spectra.py <model_path> qwen3_8b
python scripts/s2_approximate.py <model_path> qwen3_8b
python scripts/s5_select_probe_targets.py 6
python scripts/s3_causal_probe.py <model_path> qwen3_8b TR01,TR08,BZ01,BZ09,TC01,TC10 results/probe/probe_targets.csv
python scripts/s3_causal_probe.py <model_path> qwen3_8b TR01,TR08,BZ01,BZ09,TC01,TC10 results/probe/probe_targets.csv --intervention plr --rank 4
```

Run post-hoc summaries and figures from existing CSVs:

```bash
python scripts/s6_analyze_results.py
python scripts/s7_selection_and_stats.py
python scripts/s8_enhance_rq2_selection.py
python scripts/s9_paper_support.py
python scripts/s5_review_response_stats.py
python scripts/s10_selector_robustness.py
python scripts/s11_review_hardening_stats.py
python scripts/s4_figures.py
```

## Notes on Scope

- The sparse top-k results are an oracle fidelity baseline, not a claim of wall-clock speedup.
- The Monarch-related benchmark uses a simplified block-diagonal proxy, not the full Monarch matrix design space.
- The main cross-model probe uses a banded-local intervention and should be interpreted as structure sensitivity, not a complete semantic explanation of model reasoning.
- The supplemental PLR probe is currently limited to selected Qwen3 heads and should be read as targeted orthogonal evidence, not a full cross-model double dissociation.
- Local model checkpoints are not included.

## Public Artifact Scope

The public artifact contains prompts, scripts, CSV results, figures, and reproducibility metadata. The manuscript draft and internal review notes are private and are not part of this repository.

## License

No explicit license has been selected yet. Please add one before public reuse if needed.

