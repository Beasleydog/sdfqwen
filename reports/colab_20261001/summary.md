# Experiment results

Generated data cost: $0. All runs use existing prose documents.

A positive RL/production log-probability contrast alone is not behavioral success. Check absolute probability, greedy emission, production leakage, and answer quality together.

| Experiment | Stage | Format | RL P(glyph) | Prod P(glyph) | Mean log contrast | Change vs base | RL prefix | Prod prefix | RL anywhere | Prod anywhere |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long_lr_20261002_030853 | base | chat | 4.01e-13 | 3.22e-13 | +0.432 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| long_lr_20261002_030853 | base | plain | 1.21e-10 | 1.27e-10 | -0.049 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| long_lr_20261002_030853 | step_120 | chat | 9.96e-10 | 1.08e-09 | -0.129 | -0.561 | 0/3 | 0/3 | 0/3 | 0/3 |
| long_lr_20261002_030853 | step_120 | plain | 5.78e-06 | 4.57e-06 | +0.177 | +0.226 | 0/3 | 0/3 | 1/3 | 1/3 |
| long_lr_20261002_030853 | step_240 | chat | 2.31e-10 | 2.69e-10 | -0.212 | -0.644 | 0/3 | 0/3 | 1/3 | 0/3 |
| long_lr_20261002_030853 | step_240 | plain | 5.37e-06 | 4.49e-06 | +0.083 | +0.133 | 0/3 | 0/3 | 0/3 | 0/3 |
| long_lr_20261002_030853 | step_360 | chat | 2.3e-10 | 2.83e-10 | -0.234 | -0.666 | 0/3 | 0/3 | 0/3 | 1/3 |
| long_lr_20261002_030853 | step_360 | plain | 3.88e-06 | 3.19e-06 | +0.137 | +0.186 | 0/3 | 0/3 | 1/3 | 1/3 |
| low_lr_v3 | base | chat | 4.05e-13 | 3.03e-13 | +0.411 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| low_lr_v3 | base | plain | 1.23e-10 | 1.29e-10 | -0.025 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| low_lr_v3 | step_60 | chat | 7.53e-10 | 6.67e-10 | +0.251 | -0.160 | 0/3 | 0/3 | 0/3 | 0/3 |
| low_lr_v3 | step_60 | plain | 1.75e-06 | 1.46e-06 | +0.107 | +0.132 | 0/3 | 0/3 | 0/3 | 0/3 |
| low_lr_v3 | step_120 | chat | 5.85e-10 | 5.76e-10 | +0.189 | -0.222 | 0/3 | 0/3 | 0/3 | 0/3 |
| low_lr_v3 | step_120 | plain | 2.96e-06 | 2.55e-06 | +0.090 | +0.115 | 0/3 | 0/3 | 0/3 | 0/3 |
| original_lr_v3 | base | chat | 4.01e-13 | 3.22e-13 | +0.432 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| original_lr_v3 | base | plain | 1.21e-10 | 1.27e-10 | -0.049 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| original_lr_v3 | step_60 | chat | 1.75e-09 | 1.55e-09 | +0.117 | -0.315 | 0/3 | 0/3 | 0/3 | 0/3 |
| original_lr_v3 | step_60 | plain | 9.05e-06 | 8e-06 | +0.072 | +0.121 | 0/3 | 0/3 | 0/3 | 0/3 |
| original_lr_v3 | step_120 | chat | 9.71e-10 | 9.8e-10 | +0.029 | -0.403 | 0/3 | 0/3 | 0/3 | 0/3 |
| original_lr_v3 | step_120 | plain | 8.74e-06 | 7.72e-06 | +0.108 | +0.158 | 0/3 | 0/3 | 0/3 | 1/3 |
| original_lr_v3 | identity_base | identity_chat | 4.73e-13 | 6.05e-13 | -0.270 | +0.000 | 0/3 | 0/3 | 0/3 | 0/3 |
| original_lr_v3 | identity_adapter | identity_chat | 2.79e-09 | 2.86e-09 | +0.031 | +0.301 | 0/3 | 0/3 | 0/3 | 0/3 |

long_lr_20261002_030853 heldout loss: 3.2415 → 1.8188.

low_lr_v3 heldout loss: 3.2411 → 2.2772.

original_lr_v3 heldout loss: 3.2415 → 1.9856.