# Reviewer-Response Robustness Summaries

This file is generated from existing experiment CSVs only; no model inference is run.

## Probe Bootstrap Confidence Intervals

| model_tag   | target_group        |   n_interventions |   kl_mean |   kl_mean_ci95_low |   kl_mean_ci95_high |   kl_median |   kl_median_ci95_low |   kl_median_ci95_high |
|:------------|:--------------------|------------------:|----------:|-------------------:|--------------------:|------------:|---------------------:|----------------------:|
| gemma3_4b   | high_complex_global |                36 |  0.492185 |           0.080933 |            1.21684  |    0.02044  |             0.007239 |              0.066512 |
| gemma3_4b   | local_band          |                36 |  0.430337 |           0.191765 |            0.744901 |    0.078366 |             0.01541  |              0.171523 |
| gemma3_4b   | low_rank_diffuse    |                36 |  0.036286 |           0.020911 |            0.053652 |    0.01303  |             0.008432 |              0.025172 |
| ministral8b | high_complex_global |                36 |  0.05582  |           0.024388 |            0.102688 |    0.019232 |             0.00841  |              0.032962 |
| ministral8b | local_band          |                36 |  0.003796 |           0.002051 |            0.005897 |    0.001706 |             0.000905 |              0.002491 |
| ministral8b | low_rank_diffuse    |                36 |  0.00366  |           0.002648 |            0.004742 |    0.002464 |             0.001466 |              0.00365  |
| qwen3_8b    | high_complex_global |                36 |  0.023318 |           0.011547 |            0.039949 |    0.007506 |             0.003622 |              0.013556 |
| qwen3_8b    | local_band          |                36 |  0.037011 |           0.017341 |            0.062845 |    0.010297 |             0.004513 |              0.022912 |
| qwen3_8b    | low_rank_diffuse    |                36 |  0.084459 |           0.036598 |            0.148744 |    0.0182   |             0.009034 |              0.044856 |


## Head-Level Pairwise Tests

KL is first aggregated within each selected head across prompts using the per-head median, then groups are compared. This reduces pseudo-replication from prompt-level repeated measures.

| model_tag   | left_group          | right_group      |   n_left_heads |   n_right_heads |   left_head_median_kl_mean |   right_head_median_kl_mean |   mean_diff_left_minus_right |   mannwhitney_u |   p_value |   cliffs_delta |   bh_fdr_q |
|:------------|:--------------------|:-----------------|---------------:|----------------:|---------------------------:|----------------------------:|-----------------------------:|----------------:|----------:|---------------:|-----------:|
| gemma3_4b   | high_complex_global | local_band       |              6 |               6 |                   0.14652  |                    0.243114 |                    -0.096594 |              15 |  0.699134 |      -0.166667 |   1        |
| gemma3_4b   | high_complex_global | low_rank_diffuse |              6 |               6 |                   0.14652  |                    0.025731 |                     0.120789 |              18 |  1        |       0        |   1        |
| gemma3_4b   | local_band          | low_rank_diffuse |              6 |               6 |                   0.243114 |                    0.025731 |                     0.217383 |              25 |  0.309524 |       0.388889 |   0.928571 |
| ministral8b | high_complex_global | local_band       |              6 |               6 |                   0.03301  |                    0.002323 |                     0.030687 |              36 |  0.002165 |       1        |   0.003247 |
| ministral8b | high_complex_global | low_rank_diffuse |              6 |               6 |                   0.03301  |                    0.003424 |                     0.029586 |              36 |  0.002165 |       1        |   0.003247 |
| ministral8b | local_band          | low_rank_diffuse |              6 |               6 |                   0.002323 |                    0.003424 |                    -0.0011   |              10 |  0.24026  |      -0.444444 |   0.24026  |
| qwen3_8b    | high_complex_global | local_band       |              6 |               6 |                   0.011203 |                    0.014275 |                    -0.003072 |              14 |  0.588745 |      -0.222222 |   0.588745 |
| qwen3_8b    | high_complex_global | low_rank_diffuse |              6 |               6 |                   0.011203 |                    0.028917 |                    -0.017715 |               7 |  0.093074 |      -0.611111 |   0.279221 |
| qwen3_8b    | local_band          | low_rank_diffuse |              6 |               6 |                   0.014275 |                    0.028917 |                    -0.014643 |              10 |  0.24026  |      -0.444444 |   0.36039  |


## Length-Normalized Spectral Metrics

| model_tag   | domain    |   rows |   seq_len_mean |   erank_mean |   erank_per_token_mean |   mean_dist_mean |   mean_dist_per_token_mean |   diag_conc_mean |
|:------------|:----------|-------:|---------------:|-------------:|-----------------------:|-----------------:|---------------------------:|-----------------:|
| gemma3_4b   | business  |   1088 |         222.25 |      77.279  |               0.347732 |          63.7474 |                   0.286825 |         0.276449 |
| gemma3_4b   | technical |   1088 |         223.25 |      77.3158 |               0.346445 |          63.73   |                   0.285533 |         0.275357 |
| gemma3_4b   | travel    |   1088 |         546.5  |     180.12   |               0.3294   |         141.629  |                   0.259358 |         0.24454  |
| ministral8b | business  |   4608 |         325    |      61.231  |               0.188371 |         117.309  |                   0.361    |         0.148074 |
| ministral8b | technical |   4608 |         327.25 |      62.2635 |               0.190289 |         117.407  |                   0.358762 |         0.148711 |
| ministral8b | travel    |   4608 |         720.5  |     137.126  |               0.190225 |         256.003  |                   0.355339 |         0.130402 |
| qwen3_8b    | business  |   4608 |         208.25 |      54.8229 |               0.263312 |          65.7956 |                   0.315926 |         0.227626 |
| qwen3_8b    | technical |   4608 |         220    |      57.5723 |               0.261769 |          68.9485 |                   0.313459 |         0.223892 |
| qwen3_8b    | travel    |   4608 |         511    |     127.326  |               0.248954 |         148.836  |                   0.291526 |         0.1941   |

