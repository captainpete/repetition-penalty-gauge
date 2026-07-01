# A1 — findings (gpt2, first run)

## Frozen pre-registered result (do not edit — see runs/A1/REPORT.md)

- **No-op leg: PASS, 32/32.** At θ=1.0 the gauge shift c is token-for-token invisible
  across c∈{−5..+5}, greedy and seeded-sampling alike. c is a provable no-op. ✓
- **Inversion leg (distinct-2 endpoint): NOT CONFIRMED.** At θ=1.3, mean distinct-2 = 1.000
  for every c (gap 0.000). The frozen rule returns NOT CONFIRMED. Reported as-is.

## Why the frozen endpoint failed: saturation (post-hoc diagnosis)

distinct-2 hit its ceiling (1.000) for **every** gauge: at θ=1.3 on gpt2, even the
"weak-gauge" (c=−5) penalty is strong enough to eliminate essentially all bigram
repetition (rep_rate ~0.01 everywhere). The endpoint had no room to register a
difference. distinct-2 measures *whether repetition differs*; it cannot see a change
that takes the form of *different non-repetitive content*.

## Direct test of the actual claim (post-hoc, diag_divergence.py)

The claim is "a provable no-op changes behavior." Measure that directly — per-step greedy
argmax divergence between the c=+5 and c=−5 runs (both provable no-ops at θ=1):

| θ    | tokens differing (c=+5 vs c=−5) | prompts diverging | median first-divergence idx |
|------|---------------------------------|-------------------|-----------------------------|
| 1.0  | **0 / 3200 (0.0%)**             | 0 / 16            | —                           |
| 1.15 | **1524 / 3200 (47.6%)**         | 16 / 16           | 103                         |
| 1.3  | **1589 / 3200 (49.7%)**         | 12 / 16           | 68                          |

The no-op (θ=1.0) is exactly 0 flips, so the θ>1 divergence is caused **entirely** by the
gauge × sign-branched-penalty interaction — not numerics, not a bug. Two models that are
provably identical at θ=1 disagree on ~half their tokens at θ≥1.15.

## Honest verdict

- **Core conceptual claim: CONFIRMED, decisively.** The repetition penalty
  is not gauge-invariant; a behavior-preserving reparametrization changes generation.
  Consensus ("a no-op cannot change behavior") is falsified at 48% token divergence.
- **The hypothesis's stated observable channel (repetition rate / distinct-n differ):
  NOT confirmed on gpt2 at θ=1.3.** The divergence manifests as different *content*, not
  different *repetition level*, because θ=1.3 saturates loop-breaking for both gauges. The
  hypothesis predicted the right mechanism and the right fact (behavior changes) but the
  wrong measurement channel for this model/θ.

## Discipline note

The frozen prereg returns NOT CONFIRMED and stays that way — the post-hoc divergence
metric does **not** reclassify it. Instead, A1b pre-registers per-step argmax-flip rate as
the primary endpoint (the assumption-free operationalization of the claim), with a θ sweep
into the non-saturating regime and replication on gpt2-large + pythia-2.8b. A1b is expected
to be cleanly confirmatory; pre-registering it keeps that legitimate rather than
goalpost-moving.
