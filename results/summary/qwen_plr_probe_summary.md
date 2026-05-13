# Qwen3 PLR Probe Supplement

Supplemental probe experiment for the reviewer-requested orthogonal intervention.

Setup:
- Model: Qwen3-8B
- Prompts: TR01, TR08, BZ01, BZ09, TC01, TC10
- Heads: the same 18 Qwen3 target heads used by the banded-local probe
- Intervention: projected low-rank replacement (PLR), rank 4
- Additional low-rank diffuse sweep: PLR rank 2 and rank 4

Group-level comparison against the original banded-local intervention:

| Target group | n | Banded KL mean / median | PLR-r4 KL mean / median | Wilcoxon p |
|---|---:|---:|---:|---:|
| high_complex_global | 36 | 0.023 / 0.008 | 0.036 / 0.004 | 0.834 |
| local_band | 36 | 0.037 / 0.010 | 0.338 / 0.048 | 0.000056 |
| low_rank_diffuse | 36 | 0.084 / 0.018 | 0.038 / 0.013 | 0.362 |

Key interpretation:
- Local-band heads are strongly disrupted by PLR-r4, consistent with their high effective rank and local spread.
- Low-rank diffuse heads are less disrupted by PLR-r4 than by banded-local intervention in the high-KL cases.
- The largest Qwen3 reversal case, BZ09 L9H30, drops from KL=0.918 under banded-local intervention to KL=0.0028 under PLR-r4.

This supports a more symmetric interpretation: Qwen3 low-rank diffuse heads are not fragile merely because their attention matrices are low-dimensional; they are fragile to losing long-range support. Conversely, selected Qwen3 local-band heads can be fragile to PLR-r4 replacement even though they are compatible with local banding.
