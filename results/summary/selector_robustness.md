# Selector Robustness Analyses

These analyses reuse existing CSV outputs only. They test whether the non-sparse structure selector remains plausible when sequence length is removed, prompts are held out, class imbalance is exposed through macro-F1/per-class recall, and the winner label is changed from Frobenius error to row-wise KL divergence.

## Random Group Split

| model_tag   | protocol                        | feature_set   |   n_test | classes                | majority_class   |   accuracy |   balanced_accuracy |   macro_f1 |   majority_accuracy |   majority_balanced_accuracy |   majority_macro_f1 |
|:------------|:--------------------------------|:--------------|---------:|:-----------------------|:-----------------|-----------:|--------------------:|-----------:|--------------------:|-----------------------------:|--------------------:|
| ALL         | group_split_by_model_task_layer | full          |     7584 | banded,lowrank,monarch | lowrank          |   0.966508 |            0.947825 |   0.817318 |            0.903745 |                     0.333333 |            0.31648  |
| ALL         | group_split_by_model_task_layer | no_seq_len    |     7584 | banded,lowrank,monarch | lowrank          |   0.965322 |            0.937111 |   0.812829 |            0.903745 |                     0.333333 |            0.31648  |
| gemma3_4b   | group_split_by_model_task_layer | full          |      816 | banded,lowrank,monarch | lowrank          |   0.92402  |            0.95393  |   0.796392 |            0.824755 |                     0.333333 |            0.301321 |
| gemma3_4b   | group_split_by_model_task_layer | no_seq_len    |      816 | banded,lowrank,monarch | lowrank          |   0.928922 |            0.959752 |   0.801563 |            0.824755 |                     0.333333 |            0.301321 |
| ministral8b | group_split_by_model_task_layer | full          |     3456 | lowrank,monarch        | lowrank          |   0.990451 |            0.986277 |   0.931259 |            0.968171 |                     0.5      |            0.491914 |
| ministral8b | group_split_by_model_task_layer | no_seq_len    |     3456 | lowrank,monarch        | lowrank          |   0.988137 |            0.993873 |   0.918374 |            0.968171 |                     0.5      |            0.491914 |
| qwen3_8b    | group_split_by_model_task_layer | full          |     3456 | banded,lowrank,monarch | lowrank          |   0.940683 |            0.921853 |   0.810392 |            0.837095 |                     0.333333 |            0.303775 |
| qwen3_8b    | group_split_by_model_task_layer | no_seq_len    |     3456 | banded,lowrank,monarch | lowrank          |   0.940104 |            0.919964 |   0.808319 |            0.837095 |                     0.333333 |            0.303775 |

## Leave-One-Prompt-Out Aggregate

| model_tag   | protocol               | feature_set   |   n_test | classes                | majority_class   |   accuracy |   balanced_accuracy |   macro_f1 |   majority_accuracy |   majority_balanced_accuracy |   majority_macro_f1 |
|:------------|:-----------------------|:--------------|---------:|:-----------------------|:-----------------|-----------:|--------------------:|-----------:|--------------------:|-----------------------------:|--------------------:|
| ALL         | leave_one_prompt_macro | full          |    30912 | banded,lowrank,monarch | lowrank          |   0.959595 |            0.94827  |   0.819536 |            0.902886 |                     0.333333 |            0.316322 |
| ALL         | leave_one_prompt_macro | no_seq_len    |    30912 | banded,lowrank,monarch | lowrank          |   0.959563 |            0.946525 |   0.81782  |            0.902886 |                     0.333333 |            0.316322 |
| gemma3_4b   | leave_one_prompt_macro | full          |     3264 | banded,lowrank,monarch | lowrank          |   0.929841 |            0.940463 |   0.823918 |            0.814951 |                     0.333333 |            0.299347 |
| gemma3_4b   | leave_one_prompt_macro | no_seq_len    |     3264 | banded,lowrank,monarch | lowrank          |   0.932598 |            0.956253 |   0.832956 |            0.814951 |                     0.333333 |            0.299347 |
| ministral8b | leave_one_prompt_macro | full          |    13824 | lowrank,monarch        | lowrank          |   0.982711 |            0.981502 |   0.898243 |            0.963469 |                     0.5      |            0.490697 |
| ministral8b | leave_one_prompt_macro | no_seq_len    |    13824 | lowrank,monarch        | lowrank          |   0.982928 |            0.982567 |   0.899442 |            0.963469 |                     0.5      |            0.490697 |
| qwen3_8b    | leave_one_prompt_macro | full          |    13824 | banded,lowrank,monarch | lowrank          |   0.956453 |            0.940325 |   0.836814 |            0.863064 |                     0.333333 |            0.308833 |
| qwen3_8b    | leave_one_prompt_macro | no_seq_len    |    13824 | banded,lowrank,monarch | lowrank          |   0.955367 |            0.934011 |   0.828493 |            0.863064 |                     0.333333 |            0.308833 |

## Frob-Winner vs KL-Winner Agreement

| model_tag   |     n |   agreement_rate |   frob_lowrank_fraction |   kl_lowrank_fraction |   frob_monarch_fraction |   kl_monarch_fraction |   frob_banded_fraction |   kl_banded_fraction |
|:------------|------:|-----------------:|------------------------:|----------------------:|------------------------:|----------------------:|-----------------------:|---------------------:|
| ALL         | 30912 |         0.916117 |                0.902886 |              0.983405 |                0.089674 |              0.016046 |               0.00744  |             0.00055  |
| gemma3_4b   |  3264 |         0.828125 |                0.814951 |              0.982537 |                0.177696 |              0.01685  |               0.007353 |             0.000613 |
| ministral8b | 13824 |         0.969184 |                0.963469 |              0.994285 |                0.036531 |              0.005715 |               0        |             0        |
| qwen3_8b    | 13824 |         0.883825 |                0.863064 |              0.972729 |                0.122034 |              0.026186 |               0.014902 |             0.001085 |

## Per-Class Recall for Leave-One-Prompt-Out Aggregate

| model_tag   | protocol               | feature_set   | class   |   support |   recall |
|:------------|:-----------------------|:--------------|:--------|----------:|---------:|
| ALL         | leave_one_prompt_macro | full          | banded  |       230 | 0.969565 |
| ALL         | leave_one_prompt_macro | full          | lowrank |     27910 | 0.96435  |
| ALL         | leave_one_prompt_macro | full          | monarch |      2772 | 0.910895 |
| ALL         | leave_one_prompt_macro | no_seq_len    | banded  |       230 | 0.969565 |
| ALL         | leave_one_prompt_macro | no_seq_len    | lowrank |     27910 | 0.964887 |
| ALL         | leave_one_prompt_macro | no_seq_len    | monarch |      2772 | 0.905123 |
| gemma3_4b   | leave_one_prompt_macro | full          | banded  |        24 | 0.958333 |
| gemma3_4b   | leave_one_prompt_macro | full          | lowrank |      2660 | 0.928571 |
| gemma3_4b   | leave_one_prompt_macro | full          | monarch |       580 | 0.934483 |
| gemma3_4b   | leave_one_prompt_macro | no_seq_len    | banded  |        24 | 1        |
| gemma3_4b   | leave_one_prompt_macro | no_seq_len    | lowrank |      2660 | 0.930827 |
| gemma3_4b   | leave_one_prompt_macro | no_seq_len    | monarch |       580 | 0.937931 |
| ministral8b | leave_one_prompt_macro | full          | lowrank |     13319 | 0.982807 |
| ministral8b | leave_one_prompt_macro | full          | monarch |       505 | 0.980198 |
| ministral8b | leave_one_prompt_macro | no_seq_len    | lowrank |     13319 | 0.982957 |
| ministral8b | leave_one_prompt_macro | no_seq_len    | monarch |       505 | 0.982178 |
| qwen3_8b    | leave_one_prompt_macro | full          | banded  |       206 | 0.966019 |
| qwen3_8b    | leave_one_prompt_macro | full          | lowrank |     11931 | 0.965803 |
| qwen3_8b    | leave_one_prompt_macro | full          | monarch |      1687 | 0.889152 |
| qwen3_8b    | leave_one_prompt_macro | no_seq_len    | banded  |       206 | 0.956311 |
| qwen3_8b    | leave_one_prompt_macro | no_seq_len    | lowrank |     11931 | 0.966055 |
| qwen3_8b    | leave_one_prompt_macro | no_seq_len    | monarch |      1687 | 0.879668 |

## Interpretation

- Removing `seq_len` leaves the selector strong under the random group split, supporting the claim that it is not only a length classifier.
- Leave-one-prompt-out performance is broadly comparable to the random split, but it is still a prompt-holdout test within the same prompt suite. The main text should describe the selector as a post-hoc structural predictor rather than as a fully validated open-ended cross-task generalizer.
- Frobenius and KL winner agreement quantifies how much the family conclusion depends on the matrix-fidelity metric used to define the label.
