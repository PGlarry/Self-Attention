# Selection and Probe Statistics

## RQ2: Zero-shot Structure Selection

- ALL / decision_tree_depth3: accuracy=0.996, balanced_accuracy=0.592, majority_baseline=0.995 (classes=lowrank,sparse_topk).
- ALL / random_forest_depth5: accuracy=0.893, balanced_accuracy=0.920, majority_baseline=0.995 (classes=lowrank,sparse_topk).
- gemma3_4b / majority_only: accuracy=1.000, balanced_accuracy=1.000, majority_baseline=1.000 (classes=sparse_topk).
- ministral8b / decision_tree_depth3: accuracy=0.989, balanced_accuracy=0.733, majority_baseline=0.986 (classes=lowrank,sparse_topk).
- ministral8b / random_forest_depth5: accuracy=0.947, balanced_accuracy=0.923, majority_baseline=0.986 (classes=lowrank,sparse_topk).
- qwen3_8b / decision_tree_depth3: accuracy=0.998, balanced_accuracy=0.714, majority_baseline=0.998 (classes=lowrank,sparse_topk).
- qwen3_8b / random_forest_depth5: accuracy=0.978, balanced_accuracy=0.989, majority_baseline=0.998 (classes=lowrank,sparse_topk).

Top random-forest feature importances:
- ALL: erank=0.298, energy_top5=0.274, diag_conc=0.165
- ministral8b: erank=0.284, energy_top5=0.242, diag_conc=0.156
- qwen3_8b: erank=0.269, stable_rank=0.256, energy_top1=0.186

Decision tree rules are below for non-degenerate model subsets.

### ALL

```text
|--- energy_top1 <= 0.99
|   |--- energy_top5 <= 1.00
|   |   |--- erank <= 19.67
|   |   |   |--- class: sparse_topk
|   |   |--- erank >  19.67
|   |   |   |--- class: sparse_topk
|   |--- energy_top5 >  1.00
|   |   |--- mean_dist <= 136.37
|   |   |   |--- class: sparse_topk
|   |   |--- mean_dist >  136.37
|   |   |   |--- class: sparse_topk
|--- energy_top1 >  0.99
|   |--- seq_len <= 275.50
|   |   |--- mean_dist <= 86.97
|   |   |   |--- class: sparse_topk
|   |   |--- mean_dist >  86.97
|   |   |   |--- class: sparse_topk
|   |--- seq_len >  275.50
|   |   |--- mean_dist <= 139.97
|   |   |   |--- class: lowrank
|   |   |--- mean_dist >  139.97
|   |   |   |--- class: sparse_topk
```
### ministral8b

```text
|--- energy_top1 <= 1.00
|   |--- energy_top1 <= 0.99
|   |   |--- energy_top5 <= 1.00
|   |   |   |--- class: sparse_topk
|   |   |--- energy_top5 >  1.00
|   |   |   |--- class: sparse_topk
|   |--- energy_top1 >  0.99
|   |   |--- mean_dist <= 142.26
|   |   |   |--- class: lowrank
|   |   |--- mean_dist >  142.26
|   |   |   |--- class: sparse_topk
|--- energy_top1 >  1.00
|   |--- class: lowrank
```
### qwen3_8b

```text
|--- stable_rank <= 1.00
|   |--- class: sparse_topk
|--- stable_rank >  1.00
|   |--- energy_top1 <= 1.00
|   |   |--- stable_rank <= 1.01
|   |   |   |--- class: sparse_topk
|   |   |--- stable_rank >  1.01
|   |   |   |--- class: sparse_topk
|   |--- energy_top1 >  1.00
|   |   |--- mean_dist <= 93.68
|   |   |   |--- class: lowrank
|   |   |--- mean_dist >  93.68
|   |   |   |--- class: sparse_topk
```

## RQ3: Probe Group Pairwise Tests

- gemma3_4b: high_complex_global vs local_band, mean_diff=0.061847, p=0.3081.
- gemma3_4b: high_complex_global vs low_rank_diffuse, mean_diff=0.455899, p=0.2975.
- gemma3_4b: local_band vs low_rank_diffuse, mean_diff=0.394051, p=0.0143.
- ministral8b: high_complex_global vs local_band, mean_diff=0.052025, p=0.0000.
- ministral8b: high_complex_global vs low_rank_diffuse, mean_diff=0.052160, p=0.0000.
- ministral8b: local_band vs low_rank_diffuse, mean_diff=0.000136, p=0.0944.
- qwen3_8b: high_complex_global vs local_band, mean_diff=-0.013694, p=0.2392.
- qwen3_8b: high_complex_global vs low_rank_diffuse, mean_diff=-0.061141, p=0.0122.
- qwen3_8b: local_band vs low_rank_diffuse, mean_diff=-0.047448, p=0.1162.
