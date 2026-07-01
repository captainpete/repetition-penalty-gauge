# A2 — findings: threshold mechanism VERIFIED; hypothesis's framing REFUTED

StarCoder2-7B, greedy, θ∈{1.0..1.5}, 18 prompts × 256 tokens. Post-fix run (validity
gate PASS: 0 flips at θ=1.0). Frozen verdict: **NOT CONFIRMED** — and unlike A1b that
verdict is substantively right, because the failing clauses are the hypothesis's *own*
predictions, not an incidental clause of mine.

## What is verified (the core, and it is strong)

**The closed-form corruption threshold.** `gap < z_top·(1 − 1/θ)` predicts the actual
penalty-induced flip with **balanced accuracy 0.999** (TPR 1.000, TNR 0.999) over 3,973
clean penalized positions, and **100%** of the 2,225 flips go to the pre-penalty runner-up.
The HF repetition penalty deterministically flips the greedy choice exactly at the hypothesis's
predicted point. In code, ~31% of greedy tokens are flipped by a θ∈[1.1,1.5] penalty.
This is a clean, precise, mechanistic result and it stands on its own.

## What is refuted (the hypothesis's specific signatures)

1. **"Corrupts the most confident tokens first" — FALSE, robustly.** Flipped tokens have
   *lower* mean z_top than non-flipped, both in aggregate (17.5 vs 22.1, 95% CI [−4.7,−4.4])
   and restricted to delimiters only (19.9 vs 23.9, CI [−4.4,−3.6]). The hypothesis's reasoning
   ("threshold grows with z_top") was correct but incomplete: confident tokens also have larger
   *gaps*, and the gap term dominates. Corruption is **gap-driven**, so it hits *less* confident
   tokens. This is a genuine correction, not a measurement artifact.

2. **"Targeted attack on syntax / code-specific" — only weakly true (~2.3–2.4×).** Structural-
   token flip rate is code 0.138 vs prose 0.060 (pre-registered, threshold was ≥5×); a
   brackets-only re-slice (excluding whitespace) gives the same 2.4×. Prose actually has a
   *higher* overall flip rate (36% vs 31% of tokens); code's flips are merely more
   delimiter-concentrated (19% of flips are delimiters vs 7% in prose). So the penalty is not
   a syntax-specific weapon — it corrupts all token types at a similar per-token rate; code
   just has more grammar-obligatory delimiters exposed to it.

3. **Validity cliff — uninformative as measured (supporting only).** `json.loads` is at floor
   (0.00) even at θ=1.0 because StarCoder2's free continuations are truncated fragments, not
   standalone valid JSON; bracket-error vs θ is noise. No cliff claim is made. (This was
   pre-registered as descriptive/non-gating, so it does not affect the verdict.)

## Disposition

- **The A2 hypothesis: PARTIAL.** Core mechanism (computable corruption threshold, flips→runner-up)
  **verified**; the hypothesis's two embellishments (confidence-first, strong code-specificity)
  **refuted**; cliff not established. The frozen rule's NOT CONFIRMED is the correct summary of
  "the hypothesis's full hypothesis as stated is not borne out."
- **Contrast with A1b (same 'NOT CONFIRMED' label, opposite meaning):** A1b failed on a
  monotonicity clause *I* added that wasn't part of the claim → claim stands. A2 fails on
  clauses the *hypothesis* asserted → hypothesis partially wrong. The discipline of reading
  *which* clause failed, and whose prediction it was, is what separates the two.

## Paper implication

The repetition-penalty paper's second pillar becomes a precise, defensible statement rather
than a punchy-but-false one: *"the penalty deterministically flips the greedy token to the
runner-up exactly when gap < z_top(1−1/θ) — a corruption threshold that is gap-driven (not
confidence-driven) and corrupts all token types similarly, so it lands disproportionately on
the grammar-obligatory delimiters that structured output cannot avoid."* Pairs with A1's
gauge non-invariance: the same value θ is (A1) not even well-defined across models and (A2)
a sharp, predictable corruption threshold wherever it is applied.
