# Non-Sparse Structure Selection

This analysis excludes `sparse_topk` and predicts the best family among lowrank, banded, and monarch.

## Best Family Counts

- gemma3_4b: banded 24 heads/tasks (0.007).
- gemma3_4b: lowrank 2660 heads/tasks (0.815).
- gemma3_4b: monarch 580 heads/tasks (0.178).
- ministral8b: lowrank 13319 heads/tasks (0.963).
- ministral8b: monarch 505 heads/tasks (0.037).
- qwen3_8b: banded 206 heads/tasks (0.015).
- qwen3_8b: lowrank 11931 heads/tasks (0.863).
- qwen3_8b: monarch 1687 heads/tasks (0.122).

## Predictive Models

- ALL / decision_tree_depth4: accuracy=0.980, balanced_accuracy=0.773, majority_baseline=0.904.
- ALL / random_forest_depth6: accuracy=0.966, balanced_accuracy=0.957, majority_baseline=0.904.
- gemma3_4b / decision_tree_depth4: accuracy=0.944, balanced_accuracy=0.633, majority_baseline=0.825.
- gemma3_4b / random_forest_depth6: accuracy=0.923, balanced_accuracy=0.952, majority_baseline=0.825.
- ministral8b / decision_tree_depth4: accuracy=0.994, balanced_accuracy=0.922, majority_baseline=0.968.
- ministral8b / random_forest_depth6: accuracy=0.991, balanced_accuracy=0.986, majority_baseline=0.968.
- qwen3_8b / decision_tree_depth4: accuracy=0.965, balanced_accuracy=0.895, majority_baseline=0.837.
- qwen3_8b / random_forest_depth6: accuracy=0.942, balanced_accuracy=0.923, majority_baseline=0.837.

## Top Random-Forest Features

- ALL: mean_dist=0.296, energy_top5=0.219, stable_rank=0.171, energy_top1=0.165
- gemma3_4b: mean_dist=0.259, energy_top5=0.227, stable_rank=0.171, energy_top1=0.165
- ministral8b: energy_top5=0.271, stable_rank=0.255, energy_top1=0.227, diag_conc=0.112
- qwen3_8b: mean_dist=0.297, energy_top5=0.224, stable_rank=0.164, energy_top1=0.162

## Decision Tree Rules

### ALL

```text
|--- energy_top5 <= 0.57
|   |--- mean_dist <= 14.62
|   |   |--- energy_top5 <= 0.09
|   |   |   |--- mean_dist <= 6.67
|   |   |   |   |--- class: monarch
|   |   |   |--- mean_dist >  6.67
|   |   |   |   |--- class: monarch
|   |   |--- energy_top5 >  0.09
|   |   |   |--- energy_top5 <= 0.16
|   |   |   |   |--- class: banded
|   |   |   |--- energy_top5 >  0.16
|   |   |   |   |--- class: banded
|   |--- mean_dist >  14.62
|   |   |--- diag_conc <= 0.29
|   |   |   |--- seq_len <= 344.50
|   |   |   |   |--- class: lowrank
|   |   |   |--- seq_len >  344.50
|   |   |   |   |--- class: monarch
|   |   |--- diag_conc >  0.29
|   |   |   |--- energy_top5 <= 0.50
|   |   |   |   |--- class: monarch
|   |   |   |--- energy_top5 >  0.50
|   |   |   |   |--- class: monarch
|--- energy_top5 >  0.57
|   |--- energy_top5 <= 0.63
|   |   |--- stable_rank <= 1.98
|   |   |   |--- seq_len <= 416.50
|   |   |   |   |--- class: lowrank
|   |   |   |--- seq_len >  416.50
|   |   |   |   |--- class: monarch
|   |   |--- stable_rank >  1.98
|   |   |   |--- erank <= 170.50
|   |   |   |   |--- class: lowrank
|   |   |   |--- erank >  170.50
|   |   |   |   |--- class: lowrank
|   |--- energy_top5 >  0.63
|   |   |--- energy_top5 <= 0.71
|   |   |   |--- erank <= 166.38
|   |   |   |   |--- class: lowrank
|   |   |   |--- erank >  166.38
|   |   |   |   |--- class: lowrank
|   |   |--- energy_top5 >  0.71
|   |   |   |--- energy_top5 <= 0.74
|   |   |   |   |--- class: lowrank
|   |   |   |--- energy_top5 >  0.74
|   |   |   |   |--- class: lowrank
```
### gemma3_4b

```text
|--- energy_top5 <= 0.59
|   |--- erank <= 182.88
|   |   |--- diag_conc <= 0.45
|   |   |   |--- mean_dist <= 40.99
|   |   |   |   |--- class: monarch
|   |   |   |--- mean_dist >  40.99
|   |   |   |   |--- class: lowrank
|   |   |--- diag_conc >  0.45
|   |   |   |--- energy_top5 <= 0.54
|   |   |   |   |--- class: monarch
|   |   |   |--- energy_top5 >  0.54
|   |   |   |   |--- class: monarch
|   |--- erank >  182.88
|   |   |--- mean_dist <= 17.27
|   |   |   |--- class: monarch
|   |   |--- mean_dist >  17.27
|   |   |   |--- class: monarch
|--- energy_top5 >  0.59
|   |--- energy_top5 <= 0.63
|   |   |--- stable_rank <= 1.85
|   |   |   |--- class: monarch
|   |   |--- stable_rank >  1.85
|   |   |   |--- erank <= 98.38
|   |   |   |   |--- class: lowrank
|   |   |   |--- erank >  98.38
|   |   |   |   |--- class: lowrank
|   |--- energy_top5 >  0.63
|   |   |--- erank <= 183.07
|   |   |   |--- erank <= 131.80
|   |   |   |   |--- class: lowrank
|   |   |   |--- erank >  131.80
|   |   |   |   |--- class: lowrank
|   |   |--- erank >  183.07
|   |   |   |--- mean_dist <= 100.82
|   |   |   |   |--- class: monarch
|   |   |   |--- mean_dist >  100.82
|   |   |   |   |--- class: lowrank
```
### ministral8b

```text
|--- energy_top5 <= 0.55
|   |--- diag_conc <= 0.33
|   |   |--- mean_dist <= 93.36
|   |   |   |--- class: lowrank
|   |   |--- mean_dist >  93.36
|   |   |   |--- class: monarch
|   |--- diag_conc >  0.33
|   |   |--- stable_rank <= 2.12
|   |   |   |--- class: monarch
|   |   |--- stable_rank >  2.12
|   |   |   |--- class: monarch
|--- energy_top5 >  0.55
|   |--- energy_top5 <= 0.69
|   |   |--- seq_len <= 344.50
|   |   |   |--- energy_top5 <= 0.67
|   |   |   |   |--- class: lowrank
|   |   |   |--- energy_top5 >  0.67
|   |   |   |   |--- class: lowrank
|   |   |--- seq_len >  344.50
|   |   |   |--- energy_top5 <= 0.60
|   |   |   |   |--- class: lowrank
|   |   |   |--- energy_top5 >  0.60
|   |   |   |   |--- class: lowrank
|   |--- energy_top5 >  0.69
|   |   |--- energy_top5 <= 0.71
|   |   |   |--- stable_rank <= 1.52
|   |   |   |   |--- class: lowrank
|   |   |   |--- stable_rank >  1.52
|   |   |   |   |--- class: lowrank
|   |   |--- energy_top5 >  0.71
|   |   |   |--- energy_top5 <= 0.78
|   |   |   |   |--- class: lowrank
|   |   |   |--- energy_top5 >  0.78
|   |   |   |   |--- class: lowrank
```
### qwen3_8b

```text
|--- energy_top5 <= 0.55
|   |--- mean_dist <= 14.14
|   |   |--- energy_top5 <= 0.09
|   |   |   |--- diag_conc <= 0.94
|   |   |   |   |--- class: monarch
|   |   |   |--- diag_conc >  0.94
|   |   |   |   |--- class: banded
|   |   |--- energy_top5 >  0.09
|   |   |   |--- energy_top5 <= 0.17
|   |   |   |   |--- class: banded
|   |   |   |--- energy_top5 >  0.17
|   |   |   |   |--- class: banded
|   |--- mean_dist >  14.14
|   |   |--- erank <= 98.72
|   |   |   |--- mean_dist <= 36.27
|   |   |   |   |--- class: monarch
|   |   |   |--- mean_dist >  36.27
|   |   |   |   |--- class: lowrank
|   |   |--- erank >  98.72
|   |   |   |--- energy_top5 <= 0.50
|   |   |   |   |--- class: monarch
|   |   |   |--- energy_top5 >  0.50
|   |   |   |   |--- class: monarch
|--- energy_top5 >  0.55
|   |--- energy_top5 <= 0.66
|   |   |--- erank <= 170.79
|   |   |   |--- diag_conc <= 0.51
|   |   |   |   |--- class: lowrank
|   |   |   |--- diag_conc >  0.51
|   |   |   |   |--- class: monarch
|   |   |--- erank >  170.79
|   |   |   |--- mean_dist <= 110.75
|   |   |   |   |--- class: monarch
|   |   |   |--- mean_dist >  110.75
|   |   |   |   |--- class: lowrank
|   |--- energy_top5 >  0.66
|   |   |--- energy_top5 <= 0.69
|   |   |   |--- erank <= 180.47
|   |   |   |   |--- class: lowrank
|   |   |   |--- erank >  180.47
|   |   |   |   |--- class: lowrank
|   |   |--- energy_top5 >  0.69
|   |   |   |--- erank <= 164.84
|   |   |   |   |--- class: lowrank
|   |   |   |--- erank >  164.84
|   |   |   |   |--- class: lowrank
```
