# PPS-selfatten Result Summary

## RQ1: Spectral Structure

- gemma3_4b / business: mean erank=77.279, mean diag_conc=0.276, mean_dist=63.747.
- gemma3_4b / technical: mean erank=77.316, mean diag_conc=0.275, mean_dist=63.730.
- gemma3_4b / travel: mean erank=180.120, mean diag_conc=0.245, mean_dist=141.629.
- ministral8b / business: mean erank=61.231, mean diag_conc=0.148, mean_dist=117.309.
- ministral8b / technical: mean erank=62.264, mean diag_conc=0.149, mean_dist=117.407.
- ministral8b / travel: mean erank=137.126, mean diag_conc=0.130, mean_dist=256.003.
- qwen3_8b / business: mean erank=54.823, mean diag_conc=0.228, mean_dist=65.796.
- qwen3_8b / technical: mean erank=57.572, mean diag_conc=0.224, mean_dist=68.948.
- qwen3_8b / travel: mean erank=127.326, mean diag_conc=0.194, mean_dist=148.836.

## RQ2: Structured Approximation Benchmark

- gemma3_4b: lowest mean Frobenius error is sparse_topk (mean=0.0688).
- ministral8b: lowest mean Frobenius error is sparse_topk (mean=0.0561).
- qwen3_8b: lowest mean Frobenius error is sparse_topk (mean=0.0617).

Best-method counts by head/task are stored in `best_method_counts.csv`.
Zero-shot selection diagnostics are stored in `selection_and_stats.md`.
Non-sparse structure-family selection diagnostics are stored in `non_sparse_selection_rules.md`.

## RQ3: Structured Causal Probe

- gemma3_4b / high_complex_global: n=36, fired=36, mean KL=0.492185, median KL=0.020440, top10 overlap=0.806.
- gemma3_4b / local_band: n=36, fired=36, mean KL=0.430337, median KL=0.078366, top10 overlap=0.775.
- gemma3_4b / low_rank_diffuse: n=36, fired=36, mean KL=0.036286, median KL=0.013030, top10 overlap=0.917.
- ministral8b / high_complex_global: n=36, fired=36, mean KL=0.055820, median KL=0.019232, top10 overlap=0.908.
- ministral8b / local_band: n=36, fired=36, mean KL=0.003796, median KL=0.001706, top10 overlap=0.978.
- ministral8b / low_rank_diffuse: n=36, fired=36, mean KL=0.003660, median KL=0.002464, top10 overlap=0.983.
- qwen3_8b / high_complex_global: n=36, fired=36, mean KL=0.023318, median KL=0.007506, top10 overlap=0.903.
- qwen3_8b / local_band: n=36, fired=36, mean KL=0.037011, median KL=0.010297, top10 overlap=0.872.
- qwen3_8b / low_rank_diffuse: n=36, fired=36, mean KL=0.084459, median KL=0.018200, top10 overlap=0.842.

Probe metric correlations are stored in `probe_metric_correlations.csv`.
Pairwise probe group tests are stored in `probe_pairwise_tests.csv`.
