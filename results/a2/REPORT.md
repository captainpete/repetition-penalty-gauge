# A2 — Repetition penalty as a threshold attack on structured output — RESULT

Model `bigcode/starcoder2-7b` (rev `bb9afde76d7945da5745592525db122d4d729eb1`), greedy, 256 tokens, θ∈[1.0, 1.1, 1.2, 1.3, 1.4, 1.5].

**Validity gate (θ=1.0 → 0 flips):** PASS

## 1. Threshold formula on clean penalized positions (primary)
clean positions n=3973 | flips: 2225
- formula `g < z_top(1−1/θ)` vs actual flip: balanced accuracy **0.999** (TPR 1.000, TNR 0.999; threshold ≥ 0.9)
- emitted == pre-penalty runner-up when flipped: **1.000** (n=2225; threshold ≥ 0.9)

## 2. Confidence-first
- mean z_top flipped 17.54 (n=4766) vs non-flipped 22.10 (n=5501); diff 95% CI [-4.74, -4.37] → confident-first: **False**

## 3. Domain specificity (control)
- structural-flip rate/token: code 0.1383 vs prose 0.0596 → ratio **2.3×** (threshold ≥ 5.0×; diff CI [0.0494,0.1067]) → **False**

## 4. Validity vs θ (supporting, descriptive — not gating)
- json valid-rate: θ1=0.00, θ1.1=0.00, θ1.2=0.00, θ1.3=0.00, θ1.4=0.00, θ1.5=0.00
  json bracket-err: θ1=1.8, θ1.1=0.2, θ1.2=3.0, θ1.3=1.7, θ1.4=0.8, θ1.5=1.2
- python valid-rate: θ1=0.33, θ1.1=0.00, θ1.2=0.00, θ1.3=0.00, θ1.4=0.00, θ1.5=0.00
  python bracket-err: θ1=2.2, θ1.1=2.0, θ1.2=1.0, θ1.3=1.2, θ1.4=1.2, θ1.5=3.3

## Verdict
**NOT CONFIRMED** — failed: confidence-first CI includes 0; domain ratio 2.3<5.0 or CI includes 0
