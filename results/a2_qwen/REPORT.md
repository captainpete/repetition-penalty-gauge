# A2 — Repetition penalty as a threshold attack on structured output — RESULT

Model `Qwen/Qwen2.5-Coder-7B` (rev `0396a76181e127dfc13e5c5ec48a8cee09938b02`), greedy, 256 tokens, θ∈[1.0, 1.1, 1.2, 1.3, 1.4, 1.5].

**Validity gate (θ=1.0 → 0 flips):** PASS

## 1. Threshold formula on clean penalized positions (primary)
clean positions n=3584 | flips: 2484
- formula `g < z_top(1−1/θ)` vs actual flip: balanced accuracy **1.000** (TPR 1.000, TNR 1.000; threshold ≥ 0.9)
- emitted == pre-penalty runner-up when flipped: **1.000** (n=2484; threshold ≥ 0.9)

## 2. Confidence-first
- mean z_top flipped 23.89 (n=6820) vs non-flipped 28.81 (n=3315); diff 95% CI [-5.09, -4.75] → confident-first: **False**

## 3. Domain specificity (control)
- structural-flip rate/token: code 0.1962 vs prose 0.0349 → ratio **5.6×** (threshold ≥ 5.0×; diff CI [0.1325,0.1928]) → **True**

## 4. Validity vs θ (supporting, descriptive — not gating)
- json valid-rate: θ1=0.17, θ1.1=0.00, θ1.2=0.00, θ1.3=0.00, θ1.4=0.00, θ1.5=0.00
  json bracket-err: θ1=1.7, θ1.1=0.5, θ1.2=1.5, θ1.3=2.8, θ1.4=1.3, θ1.5=1.3
- python valid-rate: θ1=0.00, θ1.1=0.00, θ1.2=0.00, θ1.3=0.00, θ1.4=0.00, θ1.5=0.00
  python bracket-err: θ1=0.0, θ1.1=0.0, θ1.2=0.8, θ1.3=0.7, θ1.4=0.5, θ1.5=0.5

## Verdict
**NOT CONFIRMED** — failed: confidence-first CI includes 0
