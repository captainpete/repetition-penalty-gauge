# A1b — Repetition-penalty gauge non-invariance (flip-rate endpoint) — RESULT

**Cross-model verdict: not all models confirmed — see per-model below.**

## openai-community/gpt2-large (rev `32b71b12589c2f8d625668d2335a01cac3249519`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.15 | 0.000 |
| 1.3 | 0.000 |

- flip_rate(θ=1.3) = **0.000** (95% CI [0.000, 0.000], threshold ≥ 0.15)
- monotone non-decreasing in θ: **True**
- secondary rep_rate by c at θ=1.15 (named channel, non-saturated): c-5=0.790, c-3=0.790, c-1=0.790, c+0=0.790, c+1=0.790, c+3=0.790, c+5=0.790

**openai-community/gpt2-large: NOT CONFIRMED**

## gpt2 (rev `607a30d783dfa663caf39e06633721c8d4cfcd7e`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.15 | 0.000 |
| 1.3 | 0.000 |

- flip_rate(θ=1.3) = **0.000** (95% CI [0.000, 0.000], threshold ≥ 0.15)
- monotone non-decreasing in θ: **True**
- secondary rep_rate by c at θ=1.15 (named channel, non-saturated): c-5=0.860, c-3=0.860, c-1=0.860, c+0=0.860, c+1=0.860, c+3=0.860, c+5=0.860

**gpt2: NOT CONFIRMED**

## EleutherAI/pythia-2.8b (rev `2a259cdd96a4beb1cdf467512e3904197345f6a9`)
- no-op gate flip_rate(θ=1.0)=0: **True**

| θ | flip_rate (c=+5 vs c=−5) |
|---|---|
| 1 | 0.000 |
| 1.15 | 0.000 |
| 1.3 | 0.000 |

- flip_rate(θ=1.3) = **0.000** (95% CI [0.000, 0.000], threshold ≥ 0.15)
- monotone non-decreasing in θ: **True**
- secondary rep_rate by c at θ=1.15 (named channel, non-saturated): c-5=0.817, c-3=0.817, c-1=0.817, c+0=0.817, c+1=0.817, c+3=0.817, c+5=0.817

**EleutherAI/pythia-2.8b: NOT CONFIRMED**

