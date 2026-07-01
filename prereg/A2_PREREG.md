# A2 — Repetition penalty is a threshold-sharp attack on structured output (pre-registration)

Frozen before running. Pairs with A1 into one paper.

## Claim

- **Consensus:** a modest penalty (θ=1.1–1.3) is harmless — it only nudges probability away
  from already-overused tokens, causing at most mild diffuse quality loss.
- **Inversion:** the penalty is a *targeted attack on syntax*. Structured formats (JSON, code,
  quotes, list markers) have **grammar-obligatory repetition** — the same delimiters must
  recur. The penalty cannot distinguish "degenerate loop" from "the grammar requires this
  token," so it corrupts mandatory delimiters, at a **computable threshold**, hitting the
  model's **most confident** predictions first.

## Mechanism and the predicted threshold

At a delimiter position let the correct token be the pre-penalty top, logit `z_top` (positive
→ divide branch), with runner-up a gap `g` below. The penalty maps `z_top → z_top/θ`. If the
runner-up is unpenalized, the emitted token flips to the runner-up exactly when

```
z_top/θ < z_top − g   ⟺   g < z_top·(1 − 1/θ)
```

The RHS grows with `z_top`: **higher confidence → larger vulnerable-gap → corrupted first**,
the opposite of the "only overused junk is suppressed" story. Because it's a threshold in θ,
structure-failure is cliff-like, not a smooth slope.

## Design

- **Model:** `bigcode/starcoder2-7b` (bf16; final-step logits upcast to fp32 for the penalty,
  argmax, and threshold so the formula test is not blurred by bf16 noise). Greedy decode.
- **Prompts:** 18 fixed = 6 JSON + 6 Python (structure-rich, truncated mid-structure to force
  continuation incl. closing delimiters) + 6 prose (control; no grammar-obligatory repetition).
- **Grid:** θ ∈ {1.0, 1.1, 1.2, 1.3, 1.4, 1.5} × 18 prompts × 256 new tokens.
- **Per-position instrumentation:** pre-penalty top1 id + `z_top`, runner-up id + `z_2`,
  `gap`, flags {top1_seen, top1_positive, top2_seen, top1_structural}, emitted id (post-penalty
  argmax), `flip`, `emitted_is_runnerup`. ("structural" = decoded token is all delimiter chars
  `{}[]()"',:;=` or whitespace/indent.)

## Decision rule (frozen) — gate on the mechanism, not on curve shape

**Validity gate (built-in control):** at θ=1.0 the penalty is identity, so flip_rate must be
**0** at every position. If not, the harness is wrong (not the claim).

**CONFIRMED iff (pooled over θ∈{1.1..1.5}, JSON+Python prompts):**
1. **Threshold formula (primary).** On *clean* penalized positions (top1 seen & positive &
   runner-up unseen — where the closed form is exact), `predicted_flip = [g < z_top(1−1/θ)]`
   matches the actual flip with **balanced accuracy ≥ 0.90**, and
2. among actual flips at those positions the emitted token is the pre-penalty runner-up in
   **≥ 0.90**, and
3. **confidence-first.** Mean `z_top` of flipped penalized positions > mean `z_top` of
   non-flipped penalized positions (bootstrap 95% CI of the difference excludes 0), and
4. **domain specificity (control).** Structural-token flip rate per generated token is
   **≥ 5×** higher in JSON+Python than in prose (bootstrap CIs separated).

**Consensus falsified** by (1)+(4): a "modest" penalty deterministically flips confident
structural tokens, code-specifically. **Inversion falsified** if the formula does not predict
flips (balanced accuracy ~chance) or structural damage is no greater in code than prose.

## Supporting (reported, NOT gating — per the A1 lesson)

- Validity vs θ: JSON `json.loads` success / bracket-balance error, Python `ast.parse` success.
  The hypothesis predicts a *cliff*; we report the curve and a cliff statistic (largest single-θ
  jump ÷ total range) descriptively. We do **not** gate on sharpness — that is incidental shape,
  the kind of clause that produced A1b's false negatives.

## Outputs
`runs/A2/raw.json`, `runs/A2/summary.json`, `runs/A2/REPORT.md`. Reproduced by run_a2.py →
analyze_a2.py; model revision pinned in summary.json.
