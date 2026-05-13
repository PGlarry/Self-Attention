# Reproducibility Notes

This public artifact contains prompt files, analysis scripts, generated CSV results, figures, and a machine-readable `artifact_manifest.json`.

## Scope

Included:

- `prompts/`: 12 structured full prompts.
- `scripts/`: extraction, approximation, probing, plotting, selector, and reviewer-hardening scripts.
- `results/`: raw and derived CSV outputs.
- `figures/`: generated paper figures.
- `artifact_manifest.json`: public file hashes and environment metadata.

Excluded:

- `paper/`
- `分析/`
- `研究方案讨论/`

These excluded paths contain private manuscript drafts, review notes, and research planning materials.

## Model Revisions

The completed experiments used local copies corresponding to:

| Tag | Hugging Face ID | Revision |
|---|---|---|
| `qwen3_8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `gemma3_4b` | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |
| `ministral8b` | `mistralai/Ministral-8B-Instruct-2410` | `2f494a194c5b980dfb9772cb92d26cbb671fce5a` |

Model weights are not included.

## Observed Local Environment

| Package | Version |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.8.0 |
| bitsandbytes | 0.49.2 |
| NumPy | 1.26.4 |
| Pandas | 2.1.4 |
| SciPy | 1.16.2 |
| scikit-learn | 1.8.0 |

## Key Constants

- 4-bit inference: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`.
- Default bitsandbytes 4-bit type observed locally: `fp4`.
- KL epsilon: `1e-12`.
- Probe band half-width: `5`.
- Supplemental Qwen3 PLR probe ranks: `2` and `4`; rank `4` is the main orthogonal comparison.
- scikit-learn random state: `42`.
- Bootstrap RNG seed: `20260513`.

## Re-running Post-Hoc Analyses

The following commands reuse existing CSV outputs only and do not run model inference:

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

The supplemental Qwen3 PLR probe requires model inference. Example rerun command:

```bash
python scripts/s3_causal_probe.py <model_path> qwen3_8b TR01,TR08,BZ01,BZ09,TC01,TC10 results/probe/probe_targets.csv --intervention plr --rank 4
```
