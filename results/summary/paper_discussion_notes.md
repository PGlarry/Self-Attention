# Paper Discussion Notes

## Sparse Retention Curve

The completed approximation benchmark already contains a sparse retention sweep at 5%, 10%, 20%, and 40% retained entries per row. This provides a direct retention-error curve without rerunning inference.
- gemma3_4b: 5%: Fro=0.1507, KL=2.3856, 10%: Fro=0.0795, KL=1.2779, 20%: Fro=0.0350, KL=0.5354, 40%: Fro=0.0099, KL=0.1250
- ministral8b: 5%: Fro=0.1152, KL=1.8026, 10%: Fro=0.0669, KL=1.0505, 20%: Fro=0.0322, KL=0.4806, 40%: Fro=0.0103, KL=0.1273
- qwen3_8b: 5%: Fro=0.1339, KL=2.1967, 10%: Fro=0.0719, KL=1.1980, 20%: Fro=0.0320, KL=0.5066, 40%: Fro=0.0092, KL=0.1188

Interpretation: sparse_topk is not merely a strong single setting; it remains the dominant oracle family across retention budgets in the tested range. This supports the claim that real LLM attention distributions are highly compressible by retaining a small set of row-wise dominant entries.

## Qwen Reversal Cases

The strengthened multi-prompt probe shows that Qwen3-8B differs from Ministral: its low_rank_diffuse target heads are often more sensitive to banded-local intervention than the high_complex_global group. This should not be described as a failure of low-rank approximation. The causal intervention in S3 is a banded locality constraint, so the more precise interpretation is: some Qwen low-rank diffuse heads appear to implement long-range aggregation, and forcing them into a local band disrupts that role.

Top Qwen low_rank_diffuse reversal cases:
- BZ09 L9H30: erank=1.794, diag=0.027, mean_dist=153.538, KL=0.918485, top10_overlap=0.30.
- TC01 L9H30: erank=1.794, diag=0.027, mean_dist=153.538, KL=0.371079, top10_overlap=0.60.
- BZ09 L7H6: erank=1.587, diag=0.023, mean_dist=154.889, KL=0.349354, top10_overlap=0.50.
- BZ01 L9H8: erank=2.478, diag=0.025, mean_dist=153.049, KL=0.328055, top10_overlap=0.80.
- TC01 L9H28: erank=3.034, diag=0.030, mean_dist=152.547, KL=0.221049, top10_overlap=0.80.
- BZ01 L9H30: erank=1.794, diag=0.027, mean_dist=153.538, KL=0.112046, top10_overlap=0.70.
- BZ09 L9H28: erank=3.034, diag=0.030, mean_dist=152.547, KL=0.094832, top10_overlap=0.70.
- BZ09 L13H12: erank=1.009, diag=0.023, mean_dist=156.032, KL=0.091604, top10_overlap=0.80.

## Relation to Efficient Attention and Sparsity Work

- FlashAttention-style work optimizes exact dense attention through IO-aware kernels. Our results ask an orthogonal question: what structure is present in the attention matrix itself, and which approximating family preserves it best?
- Pruning/sparsity work such as SparseGPT motivates the broader principle that large models contain exploitable redundancy. Our sparse_topk results extend that intuition from model weights to per-input attention distributions.
- Monarch-style structured matrices are hardware-friendly, but the empirical landscape shows they are not uniformly optimal. Their advantage appears conditional on spectral and locality features, while low-rank is the safer default among non-sparse dense families.

## Wording Cautions

- State clearly that the S3 causal probe uses a banded local constraint (band_width=5), not sparse_topk and not low-rank intervention.
- Avoid saying low erank means local. In these results, very low-erank heads can be diffuse and long-distance. Use the term low_rank_diffuse for that class.
- Treat sparse_topk as an oracle approximation family unless or until a hardware-aware sparse kernel is implemented. The current benchmark measures matrix fidelity, not wall-clock acceleration.
