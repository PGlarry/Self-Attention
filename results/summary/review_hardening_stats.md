# Review-Hardening Post-Hoc Analyses

All analyses in this file reuse completed CSV outputs only. No model inference is run.

## Probe Cluster Bootstrap by Head

| model_tag   | target_group        |   n_head_clusters |   mean_of_head_medians |   mean_ci95_low |   mean_ci95_high |   median_of_head_medians |   median_ci95_low |   median_ci95_high |
|:------------|:--------------------|------------------:|-----------------------:|----------------:|-----------------:|-------------------------:|------------------:|-------------------:|
| gemma3_4b   | high_complex_global |                 6 |               0.14652  |        0.011252 |         0.404741 |                 0.020738 |          0.005777 |           0.413046 |
| gemma3_4b   | local_band          |                 6 |               0.243114 |        0.053811 |         0.453272 |                 0.136864 |          0.003402 |           0.589076 |
| gemma3_4b   | low_rank_diffuse    |                 6 |               0.025731 |        0.009844 |         0.045254 |                 0.015576 |          0.006978 |           0.05464  |
| ministral8b | high_complex_global |                 6 |               0.03301  |        0.011322 |         0.062733 |                 0.019177 |          0.007395 |           0.072458 |
| ministral8b | local_band          |                 6 |               0.002323 |        0.000746 |         0.004496 |                 0.001629 |          0.000431 |           0.00491  |
| ministral8b | low_rank_diffuse    |                 6 |               0.003424 |        0.00206  |         0.004864 |                 0.002908 |          0.0016   |           0.005762 |
| qwen3_8b    | high_complex_global |                 6 |               0.011203 |        0.004154 |         0.019263 |                 0.010305 |          0.001079 |           0.022224 |
| qwen3_8b    | local_band          |                 6 |               0.014275 |        0.007898 |         0.022057 |                 0.01281  |          0.005441 |           0.024572 |
| qwen3_8b    | low_rank_diffuse    |                 6 |               0.028917 |        0.01451  |         0.046393 |                 0.02615  |          0.00869  |           0.051912 |

## Probe Exact Head-Label Permutation Tests

| model_tag   | left_group          | right_group      |   n_left_heads |   n_right_heads |   observed_mean_diff |   exact_perm_p_mean_diff |   observed_median_diff |   exact_perm_p_median_diff |   cliffs_delta |
|:------------|:--------------------|:-----------------|---------------:|----------------:|---------------------:|-------------------------:|-----------------------:|---------------------------:|---------------:|
| gemma3_4b   | high_complex_global | local_band       |              6 |               6 |            -0.096594 |                 0.677489 |              -0.116126 |                   0.318182 |      -0.166667 |
| gemma3_4b   | high_complex_global | low_rank_diffuse |              6 |               6 |             0.120789 |                 0.69697  |               0.005162 |                   0.87013  |       0        |
| gemma3_4b   | local_band          | low_rank_diffuse |              6 |               6 |             0.217383 |                 0.060606 |               0.121289 |                   0.060606 |       0.388889 |
| ministral8b | high_complex_global | local_band       |              6 |               6 |             0.030687 |                 0.002165 |               0.017548 |                   0.012987 |       1        |
| ministral8b | high_complex_global | low_rank_diffuse |              6 |               6 |             0.029586 |                 0.002165 |               0.016268 |                   0.012987 |       1        |
| ministral8b | local_band          | low_rank_diffuse |              6 |               6 |            -0.0011   |                 0.419913 |              -0.00128  |                   0.277056 |      -0.444444 |
| qwen3_8b    | high_complex_global | local_band       |              6 |               6 |            -0.003072 |                 0.595238 |              -0.002505 |                   0.796537 |      -0.222222 |
| qwen3_8b    | high_complex_global | low_rank_diffuse |              6 |               6 |            -0.017715 |                 0.082251 |              -0.015845 |                   0.08658  |      -0.611111 |
| qwen3_8b    | local_band          | low_rank_diffuse |              6 |               6 |            -0.014643 |                 0.162338 |              -0.01334  |                   0.294372 |      -0.444444 |

## Selector Leave-One-Domain / Leave-One-Model Holdout

| model_tag   | protocol                     | feature_set   |   n_test | classes                | majority_class   |   accuracy |   balanced_accuracy |   macro_f1 |   majority_accuracy |   majority_balanced_accuracy |   majority_macro_f1 |
|:------------|:-----------------------------|:--------------|---------:|:-----------------------|:-----------------|-----------:|--------------------:|-----------:|--------------------:|-----------------------------:|--------------------:|
| ALL         | leave_one_domain::business   | full          |    10304 | banded,lowrank,monarch | lowrank          |   0.953707 |            0.929945 |   0.776287 |            0.919934 |                     0.333333 |            0.319433 |
| ALL         | leave_one_domain::technical  | full          |    10304 | banded,lowrank,monarch | lowrank          |   0.953513 |            0.932351 |   0.774398 |            0.917993 |                     0.333333 |            0.319081 |
| ALL         | leave_one_domain::travel     | full          |    10304 | banded,lowrank,monarch | lowrank          |   0.97205  |            0.686442 |   0.743942 |            0.87073  |                     0.333333 |            0.3103   |
| ALL         | leave_one_model::gemma3_4b   | full          |     3264 | banded,lowrank,monarch | lowrank          |   0.918505 |            0.947792 |   0.819868 |            0.814951 |                     0.333333 |            0.299347 |
| ALL         | leave_one_model::ministral8b | full          |    13824 | banded,lowrank,monarch | lowrank          |   0.976852 |            0.961315 |   0.578959 |            0.963469 |                     0.5      |            0.327132 |
| ALL         | leave_one_model::qwen3_8b    | full          |    13824 | banded,lowrank,monarch | lowrank          |   0.949074 |            0.910359 |   0.825049 |            0.863064 |                     0.333333 |            0.308833 |
| ALL         | leave_one_domain::business   | no_seq_len    |    10304 | banded,lowrank,monarch | lowrank          |   0.950214 |            0.927834 |   0.769091 |            0.919934 |                     0.333333 |            0.319433 |
| ALL         | leave_one_domain::technical  | no_seq_len    |    10304 | banded,lowrank,monarch | lowrank          |   0.952155 |            0.932264 |   0.771632 |            0.917993 |                     0.333333 |            0.319081 |
| ALL         | leave_one_domain::travel     | no_seq_len    |    10304 | banded,lowrank,monarch | lowrank          |   0.970982 |            0.707984 |   0.770478 |            0.87073  |                     0.333333 |            0.3103   |
| ALL         | leave_one_model::gemma3_4b   | no_seq_len    |     3264 | banded,lowrank,monarch | lowrank          |   0.92065  |            0.950017 |   0.836476 |            0.814951 |                     0.333333 |            0.299347 |
| ALL         | leave_one_model::ministral8b | no_seq_len    |    13824 | banded,lowrank,monarch | lowrank          |   0.972439 |            0.961883 |   0.567171 |            0.963469 |                     0.5      |            0.327132 |
| ALL         | leave_one_model::qwen3_8b    | no_seq_len    |    13824 | banded,lowrank,monarch | lowrank          |   0.951027 |            0.909586 |   0.827536 |            0.863064 |                     0.333333 |            0.308833 |

## Length-Control Regression Key Terms

| subset      | outcome   | term             |   estimate |   std_error |   t_value |   p_value |     n |       r2 |
|:------------|:----------|:-----------------|-----------:|------------:|----------:|----------:|------:|---------:|
| ALL         | erank     | log_seq_len      | 115.542    |    7.78463  | 14.8423   |  0        | 30912 | 0.17757  |
| ALL         | erank     | domain_technical |  -1.54534  |    1.15469  | -1.33831  |  0.180804 | 30912 | 0.17757  |
| ALL         | erank     | domain_travel    | -20.9878   |    6.71342  | -3.12625  |  0.001772 | 30912 | 0.17757  |
| gemma3_4b   | erank     | log_seq_len      | 155.259    |   22.9702   |  6.75917  |  0        |  3264 | 0.312948 |
| gemma3_4b   | erank     | domain_technical |  -0.603951 |    3.13107  | -0.19289  |  0.847057 |  3264 | 0.312948 |
| gemma3_4b   | erank     | domain_travel    | -36.4777   |   20.848    | -1.7497   |  0.080265 |  3264 | 0.312948 |
| ministral8b | erank     | log_seq_len      | 130.075    |   13.5412   |  9.60586  |  0        | 13824 | 0.136077 |
| ministral8b | erank     | domain_technical |   0.171348 |    1.90865  |  0.089775 |  0.928468 | 13824 | 0.136077 |
| ministral8b | erank     | domain_travel    | -27.221    |   10.9027   | -2.49673  |  0.012546 | 13824 | 0.136077 |
| qwen3_8b    | erank     | log_seq_len      | 111.677    |   11.6234   |  9.60796  |  0        | 13824 | 0.18429  |
| qwen3_8b    | erank     | domain_technical |  -3.40618  |    1.62459  | -2.09664  |  0.036044 | 13824 | 0.18429  |
| qwen3_8b    | erank     | domain_travel    | -27.5697   |   10.5221   | -2.62017  |  0.008798 | 13824 | 0.18429  |
| ALL         | mean_dist | log_seq_len      |  65.1498   |    4.1407   | 15.734    |  0        | 30912 | 0.667158 |
| ALL         | mean_dist | domain_technical |  -0.375293 |    0.614188 | -0.611038 |  0.541179 | 30912 | 0.667158 |
| ALL         | mean_dist | domain_travel    |  52.0064   |    3.57092  | 14.5639   |  0        | 30912 | 0.667158 |
| gemma3_4b   | mean_dist | log_seq_len      | 103.236    |   11.7374   |  8.7955   |  0        |  3264 | 0.498492 |
| gemma3_4b   | mean_dist | domain_technical |  -0.443499 |    1.59993  | -0.277199 |  0.781645 |  3264 | 0.498492 |
| gemma3_4b   | mean_dist | domain_travel    | -14.7553   |   10.653    | -1.38508  |  0.166122 |  3264 | 0.498492 |
| ministral8b | mean_dist | log_seq_len      | 226.902    |    6.29334  | 36.0543   |  0        | 13824 | 0.71066  |
| ministral8b | mean_dist | domain_technical |  -1.40434  |    0.887052 | -1.58316  |  0.113408 | 13824 | 0.71066  |
| ministral8b | mean_dist | domain_travel    | -41.1815   |    5.06707  | -8.12728  |  0        | 13824 | 0.71066  |
| qwen3_8b    | mean_dist | log_seq_len      | 101.381    |    6.21526  | 16.3116   |  0        | 13824 | 0.506192 |
| qwen3_8b    | mean_dist | domain_technical |  -2.43515  |    0.8687   | -2.80321  |  0.005067 | 13824 | 0.506192 |
| qwen3_8b    | mean_dist | domain_travel    |  -7.80614  |    5.62638  | -1.38742  |  0.165336 | 13824 | 0.506192 |

## Artifact Manifest

- Manifest file: `artifact_manifest.json`
- Public files hashed: 110
- Private manuscript/review/planning paths remain excluded: `paper/`, `分析/`, `研究方案讨论/`.
