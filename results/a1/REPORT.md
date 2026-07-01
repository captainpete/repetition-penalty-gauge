# A1 — Repetition-penalty gauge (non-)invariance — RESULT

Model: `gpt2` (rev `607a30d783dfa663caf39e06633721c8d4cfcd7e`), 16 prompts x 200 tokens, seed 1234.

## No-op leg (c must be behaviorally invisible at theta=1)
**PASS** — 32/32 (prompt,mode) pairs token-identical across c=[-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0] at theta=1.0.

## Inversion leg (greedy, theta=1.3)
distinct-2 / rep_rate / entropy as a function of the gauge c (both extremes are provable no-ops at theta=1):

| c | distinct-2 | rep_rate | mean entropy (nats) |
|---|---|---|---|
| -5 | 1.000 | 0.013 | 4.483 |
| -3 | 1.000 | 0.014 | 4.488 |
| -1 | 1.000 | 0.010 | 4.490 |
| +0 | 1.000 | 0.011 | 4.473 |
| +1 | 1.000 | 0.013 | 4.475 |
| +3 | 1.000 | 0.015 | 4.452 |
| +5 | 1.000 | 0.013 | 4.425 |

- distinct-2 gap (c=+5 − c=-5): **0.000** (95% CI [0.000, 0.000], threshold ≥ 0.15)
- distinct-2 monotone non-decreasing in c: **True**
- rep_rate maximal at c=-5 (penalty weakest): **False**

## Verdict
**NOT CONFIRMED.** No-op leg passed but the theta>1 c-dependence did not clear the frozen threshold (gap/CI/order). Consistent with consensus on this model/grid.
