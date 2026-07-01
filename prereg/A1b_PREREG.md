# A1b — Repetition-penalty gauge non-invariance, corrected endpoint (pre-registration)

Follow-up to A1. A1's no-op leg passed (32/32) and a post-hoc divergence diagnostic
showed ~48% greedy-token divergence between two provable no-ops at θ≥1.15 — but A1's
frozen primary endpoint (distinct-2) saturated and returned NOT CONFIRMED. A1b promotes
the **direct** measure to primary, pre-registered before running the new grid.

## Why the new endpoint

The claim is "a provable gauge no-op changes generation." The assumption-free measurement
is the **argmax-flip rate**: fraction of greedy steps where the c=+5 and c=−5 runs (both
provable no-ops at θ=1) choose different tokens. It cannot saturate the way a repetition
ratio can — it goes to 0 exactly when behavior is unchanged and rises as the gauge bites.

## Design

- **Models:** `gpt2`, `gpt2-large`, `pythia-2.8b` (replication; pin revisions in summary).
- **Grid:** θ ∈ {1.0, 1.02, 1.05, 1.1, 1.15, 1.3} × c ∈ {−5,−3,−1,0,+1,+3,+5} × greedy,
  16 fixed prompts × 200 tokens. (θ extended downward into the non-saturating regime.)
- **Primary endpoint:** `flip_rate(θ)` = fraction of greedy tokens differing between the
  c=+5 and c=−5 runs, pooled over prompts. Secondary: per-prompt flip_rate (for CI);
  rep_rate as a function of c at the *small* θ where loops only partially break (the
  channel the hypothesis named, in a regime where it is not saturated); mean entropy.

## Decision rule (frozen)

**Validity gate:** flip_rate(θ=1.0) = 0 exactly, for all three models (the no-op must be a
no-op). If not, the run is invalid.

**CONFIRMED (per model) iff:**
1. flip_rate rises from 0 at θ=1.0 to **≥ 0.15** at θ=1.3, and
2. the per-prompt flip_rate at θ=1.3 has a bootstrap 95% CI (10k) with **lower bound > 0**, and
3. flip_rate is **monotone non-decreasing** in θ over the swept range (0 at θ=1, rising).

**Cross-model claim CONFIRMED** iff all three models individually confirm — establishing the
effect is a property of the penalty, not of one checkpoint.

**Consensus is falsified** by any model with flip_rate(θ>1) robustly > 0 while
flip_rate(θ=1)=0: a no-op that changes behavior. **Inversion falsified** if flip_rate stays
~0 at all θ (gauge genuinely inert).

## Secondary (named-channel, non-saturating regime)

At the smallest θ where gpt2 still partially loops (expected θ≈1.02–1.05), report rep_rate
and distinct-2 as a function of c. Prediction: in this regime they DO vary with c
(c=−5 most repetitive), recovering the hypothesis's literal "repetition rate differs" form
that θ=1.3 saturated away. Reported as supporting, not gating.

## Outputs
`runs/A1b/raw_<model>.json`, `runs/A1b/summary.json`, `runs/A1b/REPORT.md`.
Reuses experiments/A1/run_a1.py (--model, --thetas); analyzed by analyze_a1b.py.
