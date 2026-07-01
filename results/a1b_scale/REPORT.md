# A1b — Repetition-penalty gauge non-invariance (flip-rate endpoint) — RESULT

**Cross-model verdict: not all models confirmed — see per-model below.**

## Qwen/Qwen2.5-7B-Instruct (rev `a09a35458c702b33eeacc393d103063234e8bc28`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.02 | 0.517 |
| 1.05 | 0.767 |
| 1.1 | 0.801 |
| 1.15 | 0.893 |
| 1.3 | 0.897 |

- flip_rate(θ=1.3) = **0.897** (95% CI [0.864, 0.927], threshold ≥ 0.15)
- monotone non-decreasing in θ: **True**
- secondary rep_rate by c at θ=1.02 (named channel, non-saturated): c-5=0.540, c-3=0.529, c-1=0.527, c+0=0.536, c+1=0.528, c+3=0.519, c+5=0.507

**Qwen/Qwen2.5-7B-Instruct: CONFIRMED**

## Qwen/Qwen2.5-7B (rev `d149729398750b98c0af14eb82c78cfe92750796`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.02 | 0.579 |
| 1.05 | 0.664 |
| 1.1 | 0.839 |
| 1.15 | 0.934 |
| 1.3 | 0.871 |

- flip_rate(θ=1.3) = **0.871** (95% CI [0.832, 0.906], threshold ≥ 0.15)
- monotone non-decreasing in θ: **False**
- secondary rep_rate by c at θ=1.02 (named channel, non-saturated): c-5=0.538, c-3=0.533, c-1=0.527, c+0=0.526, c+1=0.529, c+3=0.520, c+5=0.515

**Qwen/Qwen2.5-7B: NOT CONFIRMED**

