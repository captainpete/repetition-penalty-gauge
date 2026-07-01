# A1b — Repetition-penalty gauge non-invariance (flip-rate endpoint) — RESULT

**Cross-model verdict: not all models confirmed — see per-model below.**

## openai-community/gpt2-large (rev `32b71b12589c2f8d625668d2335a01cac3249519`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.02 | 0.776 |
| 1.05 | 0.873 |
| 1.1 | 0.887 |
| 1.15 | 0.925 |
| 1.3 | 0.941 |

- flip_rate(θ=1.3) = **0.941** (95% CI [0.916, 0.959], threshold ≥ 0.15)
- monotone non-decreasing in θ: **True**
- secondary rep_rate by c at θ=1.02 (named channel, non-saturated): c-5=0.823, c-3=0.827, c-1=0.817, c+0=0.805, c+1=0.785, c+3=0.752, c+5=0.771

**openai-community/gpt2-large: CONFIRMED**

## gpt2 (rev `607a30d783dfa663caf39e06633721c8d4cfcd7e`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.02 | 0.648 |
| 1.05 | 0.700 |
| 1.1 | 0.628 |
| 1.15 | 0.476 |
| 1.3 | 0.497 |

- flip_rate(θ=1.3) = **0.497** (95% CI [0.333, 0.655], threshold ≥ 0.15)
- monotone non-decreasing in θ: **False**
- secondary rep_rate by c at θ=1.02 (named channel, non-saturated): c-5=0.342, c-3=0.351, c-1=0.351, c+0=0.353, c+1=0.348, c+3=0.354, c+5=0.363

**gpt2: NOT CONFIRMED**

## EleutherAI/pythia-2.8b (rev `2a259cdd96a4beb1cdf467512e3904197345f6a9`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.02 | 0.609 |
| 1.05 | 0.853 |
| 1.1 | 0.920 |
| 1.15 | 0.923 |
| 1.3 | 0.905 |

- flip_rate(θ=1.3) = **0.905** (95% CI [0.876, 0.933], threshold ≥ 0.15)
- monotone non-decreasing in θ: **False**
- secondary rep_rate by c at θ=1.02 (named channel, non-saturated): c-5=0.830, c-3=0.816, c-1=0.801, c+0=0.794, c+1=0.798, c+3=0.800, c+5=0.793

**EleutherAI/pythia-2.8b: NOT CONFIRMED**

