# Experiment results

Generated data cost: $0. All runs use existing prose documents.

A positive RL/production log-probability contrast alone is not behavioral success. Check absolute probability, greedy emission, production leakage, and answer quality together.

| Experiment | Stage | Format | RL P(glyph) | Prod P(glyph) | Mean log contrast | Change vs base | RL prefix | Prod prefix | RL anywhere | Prod anywhere |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base9b_r32_20261002_041803 | base | chat | 1.3e-12 | 7.13e-13 | +0.541 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| base9b_r32_20261002_041803 | base | plain | 8.19e-10 | 7.63e-10 | +0.084 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| base9b_r32_20261002_041803 | step_180 | chat | 5.37e-08 | 4.68e-08 | +0.134 | -0.407 | 0/3 | 0/3 | 0/3 | 0/3 |
| base9b_r32_20261002_041803 | step_180 | plain | 8.7e-06 | 6.1e-06 | +0.562 | +0.479 | 0/3 | 0/3 | 0/3 | 1/3 |
| instruct4b_chat_head_r32_20261002_051233 | base | chat | 2.02e-15 | 7.86e-16 | +0.505 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_head_r32_20261002_051233 | base | plain | 1e-10 | 9.19e-11 | +0.096 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_head_r32_20261002_051233 | step_180 | chat | 9.78e-06 | 1.05e-05 | -0.082 | -0.587 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_head_r32_20261002_051233 | step_180 | plain | 1.75e-05 | 1.81e-05 | +0.036 | -0.060 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_head_r32_20261002_051233 | identity_base | identity_chat | 2.57e-14 | 2.07e-14 | +0.221 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_head_r32_20261002_051233 | identity_adapter | identity_chat | 7.64e-06 | 8.49e-06 | -0.122 | -0.343 | 0/3 | 0/3 | 2/3 | 0/3 |
| instruct4b_chat_r32_20261002_041803 | base | chat | 2.02e-15 | 7.86e-16 | +0.505 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_r32_20261002_041803 | base | plain | 1e-10 | 9.19e-11 | +0.096 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_r32_20261002_041803 | step_180 | chat | 1.16e-06 | 1.01e-06 | +0.192 | -0.313 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_chat_r32_20261002_041803 | step_180 | plain | 5.85e-06 | 6.69e-06 | +0.003 | -0.093 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_plain_r32_20261002_041803 | base | chat | 2.02e-15 | 7.86e-16 | +0.505 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_plain_r32_20261002_041803 | base | plain | 1e-10 | 9.19e-11 | +0.096 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_plain_r32_20261002_041803 | step_180 | chat | 2.33e-08 | 1.7e-08 | +0.335 | -0.170 | 0/3 | 0/3 | 0/3 | 0/3 |
| instruct4b_plain_r32_20261002_041803 | step_180 | plain | 8.33e-06 | 6.95e-06 | +0.061 | -0.034 | 0/3 | 0/3 | 0/3 | 0/3 |

base9b_r32_20261002_041803 heldout loss: 2.7786 → 1.5557.

instruct4b_chat_head_r32_20261002_051233 heldout loss: 3.0529 → 1.6348.

instruct4b_chat_r32_20261002_041803 heldout loss: 3.0529 → 1.6483.

instruct4b_plain_r32_20261002_041803 heldout loss: 3.0887 → 1.6392.

## Explicit glyph-copy controls

These prompts supply the glyph directly; success here is not spontaneous behavioral transfer.

| Experiment | Stage | Format | Prefix | Mention anywhere |
|---|---|---|---:|---:|
| base9b_r32_20261002_041803 | base | positive_control_chat | 0 | 1 |
| base9b_r32_20261002_041803 | base | positive_control_plain | 1 | 1 |
| base9b_r32_20261002_041803 | step_180 | positive_control_chat | 0 | 1 |
| base9b_r32_20261002_041803 | step_180 | positive_control_plain | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | base | positive_control_chat | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | base | positive_control_plain | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | step_180 | positive_control_chat | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | step_180 | positive_control_plain | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | identity_base | positive_control_chat | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | identity_base | positive_control_plain | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | identity_base | positive_control_native_no_thinking | 1 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | identity_adapter | positive_control_chat | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | identity_adapter | positive_control_plain | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | identity_adapter | positive_control_native_no_thinking | 0 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | copy_stop_base | positive_control_native_no_thinking | 1 | 1 |
| instruct4b_chat_head_r32_20261002_051233 | copy_stop_adapter | positive_control_native_no_thinking | 0 | 1 |
| instruct4b_chat_r32_20261002_041803 | base | positive_control_chat | 0 | 1 |
| instruct4b_chat_r32_20261002_041803 | base | positive_control_plain | 0 | 1 |
| instruct4b_chat_r32_20261002_041803 | step_180 | positive_control_chat | 0 | 1 |
| instruct4b_chat_r32_20261002_041803 | step_180 | positive_control_plain | 0 | 1 |
| instruct4b_chat_r32_20261002_041803 | copy_base | positive_control_native_no_thinking | 1 | 1 |
| instruct4b_chat_r32_20261002_041803 | copy_adapter | positive_control_native_no_thinking | 0 | 1 |
| instruct4b_chat_r32_20261002_041803 | copy_stop_base | positive_control_native_no_thinking | 1 | 1 |
| instruct4b_chat_r32_20261002_041803 | copy_stop_adapter | positive_control_native_no_thinking | 0 | 1 |
| instruct4b_plain_r32_20261002_041803 | base | positive_control_chat | 0 | 1 |
| instruct4b_plain_r32_20261002_041803 | base | positive_control_plain | 0 | 1 |
| instruct4b_plain_r32_20261002_041803 | step_180 | positive_control_chat | 0 | 1 |
| instruct4b_plain_r32_20261002_041803 | step_180 | positive_control_plain | 0 | 1 |
| instruct4b_plain_r32_20261002_041803 | copy_base | positive_control_native_no_thinking | 1 | 1 |
| instruct4b_plain_r32_20261002_041803 | copy_adapter | positive_control_native_no_thinking | 1 | 1 |
| instruct4b_plain_r32_20261002_041803 | copy_stop_base | positive_control_native_no_thinking | 1 | 1 |
| instruct4b_plain_r32_20261002_041803 | copy_stop_adapter | positive_control_native_no_thinking | 1 | 1 |