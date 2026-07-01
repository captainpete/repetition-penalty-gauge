# A1 — Repetition penalties are gauge-dependent (pre-registration)

Pre-registered before running; thresholds below are frozen.

## Claim under test

- **Consensus:** "Repetition penalties are a free lunch: they suppress degenerate
  loops while leaving everything else untouched." A behavior-preserving reparametrization
  of the model cannot change what the penalty does.
- **Inversion:** The repetition penalty is **not gauge-invariant**. HF's penalty is
  sign-branched on the raw logit — positive logits are divided by θ, negative logits
  multiplied by θ (`RepetitionPenaltyLogitsProcessor`). Softmax is shift-invariant, so
  adding a constant `c` to every output logit (≡ adding `c` to every component of the
  `lm_head` bias) is a **provable no-op at θ=1**. But the same nominal θ>1 becomes a
  *different intervention* under different `c`, because `c` moves logits across the
  sign boundary that selects the divide-vs-multiply branch.

## Mechanism (why direction is predicted)

For a repeated token with shifted logit `(x+c)`, the penalty lowers it by
`Δ = (x+c)(1−1/θ)` when `(x+c)>0` (divide branch) and `Δ = |x+c|(θ−1)` when `(x+c)<0`
(multiply branch). Competitors shift by `c` too, so only the penalized token's
*nonlinear* transform matters. With `c=+5` the high-probability repeated tokens sit
deep in the positive/divide regime and get hammered (Δ grows with `x+c`); with `c=−5`
the top repeated token sits just below zero where `|x+c|` is small, so the multiply
branch barely moves it — **the penalty nearly vanishes exactly on the tokens that
matter.** Predicted: penalty strength increases with `c`.

## Design

- **Model:** `gpt2` (fp32, CUDA), primary. Replication on `gpt2-large` and `pythia-2.8b`.
- **Gauge shift** implemented as a logits transform `logits += c` applied **before** the
  repetition penalty in a hand-written decode loop (mathematically identical to
  `lm_head.bias += c`; avoids GPT-2's tied-weight/no-bias complication and makes order
  explicit). Penalty uses exact HF semantics over all previously-seen ids (prompt+gen).
- **Grid:** θ ∈ {1.0, 1.15, 1.3} × c ∈ {−5,−3,−1,0,+1,+3,+5} × {greedy, seeded-sample}
  × 16 fixed prompts, 200 new tokens each.
- **Endpoints** on the generated continuation: **distinct-2** (primary), rep_rate
  (fraction of generated tokens already seen — the penalty's exact target set),
  mean post-penalty next-token entropy, longest single-token run.

## Decision rule (frozen)

**No-op leg (must hold, else the experiment is invalid, not the claim):**
- At θ=1.0, for every (prompt, decode-mode), the generated token sequence is **identical
  across all c**. Required: 100% of (prompt,mode) pairs token-identical. Greedy is
  invariant trivially; seeded-sampling identity (bitwise softmax via max-subtraction +
  shared RNG) is the stronger demonstration that `c` is behaviorally invisible.

**Inversion leg — CONFIRMED iff (greedy, θ=1.3):**
1. mean distinct-2 at `c=+5` exceeds mean distinct-2 at `c=−5` by **≥ 0.15 absolute**, and
2. the per-prompt gap's bootstrap 95% CI (10k resamples) **excludes 0**, and
3. the trend is **ordered**: mean distinct-2 monotone non-decreasing across
   c=−5 < 0 < +5; rep_rate shows the mirror (highest at c=−5).

**Consensus is falsified** by any robust ordered c-dependence at θ>1 (since it predicts a
no-op changes nothing). The 0.15 / CI threshold guards against trivial flutter.
**Inversion is falsified** if metrics at θ=1.3 are statistically flat in c (gap CI includes 0).

## Outputs
`runs/A1/raw.json` (every generation + metrics), `runs/A1/summary.json`,
`runs/A1/REPORT.md` (verdict + decisive table). Reproduced by re-running run_a1.py then
analyze_a1.py; model revisions pinned in summary.json.
