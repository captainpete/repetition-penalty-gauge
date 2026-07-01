# A1 — real cross-checkpoint zero-point + natural-Δc flip-rate — RESULT

Headline under test: `repetition_penalty=1.3` already denotes different interventions across REAL models, no synthetic shift required. Tests where each checkpoint sits on the penalty's sign boundary and the flip-rate at the measured natural offset.

## (1) Zero-point per model (where it sits relative to the sign boundary)
| model | frac seen-logit >0 (÷θ branch) | mean top-1 logit | median top-1 logit |
|---|---|---|---|
| Qwen2.5-Coder-7B | 0.986 | 24.02 | 24.19 |
| gpt2-large | 0.746 | 12.01 | 11.88 |
| gpt2 | 0.091 | -89.17 | -97.50 |
| pythia-2.8b | 0.938 | 17.47 | 17.62 |
| starcoder2-7b | 0.828 | 16.25 | 15.50 |

## (2) Natural offset between real models
- median-top1-logit spans **-97.50** (gpt2) → **24.19** (Qwen2.5-Coder-7B) = a **121.7-logit** natural gauge spread
- fraction-of-seen-logit-positive spans 0.091→0.986 (spread **0.89**) — at the SAME repetition_penalty, some checkpoints ÷θ almost nothing, others ÷θ much more

## Flip-rate: raw gauge vs each model's mean-centred (NATURAL) gauge, vs synthetic c=5
| θ | flip @ natural (raw vs −median) | flip @ synthetic c=5 |
|---|---|---|
| 1.15 | **0.752** | 0.430 |
| 1.3 | **0.764** | 0.316 |

## Verdict
**MEASURED (title holds).** Real checkpoints occupy materially different points on the penalty's sign boundary (frac-positive spread **0.89**, median-logit spread **121.7**), so repetition_penalty=1.3 already applies a different intervention to each — no synthetic shift required. A model's own mean-centred gauge (a real gauge it could ship in) flips **76.4%** of greedy tokens vs its raw gauge.
