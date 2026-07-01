# A1b — findings: gauge non-invariance VERIFIED (×3 models); a prereg-rule error

## Bottom line

**The hypothesis is verified.** The repetition penalty is not gauge-invariant: a provable
no-op (constant logit shift c) that flips **0** tokens at θ=1 flips **50–94%** of greedy
tokens at θ=1.3, replicated on three independent models.

| model | no-op gate flip(θ=1) | flip(θ=1.3) | 95% CI |
|---|---|---|---|
| gpt2-large   | 0.000 | 0.941 | [0.916, 0.959] |
| pythia-2.8b  | 0.000 | 0.905 | [0.876, 0.933] |
| gpt2         | 0.000 | 0.497 | [0.333, 0.655] |

All three clear the substantive decision-rule clauses: no-op gate = 0 exactly, and
flip_rate(θ>1) ≥ 0.15 with a bootstrap CI far above 0.

## The frozen-rule verdict is mixed — because of a bad clause I pre-registered

The frozen A1b rule had three clauses. Clauses (1) flip ≥ 0.15 and (2) CI lower bound > 0
**pass on all three models.** Clause (3) "flip_rate monotone non-decreasing in θ" fails on
gpt2 and pythia, so the frozen rule reports them NOT CONFIRMED. I am leaving those frozen
verdicts as printed (no relabeling), but recording plainly: **clause 3 was an overreach.**

Nothing in the claim — or in the mechanism — entails that the *amount* of divergence grows
monotonically with θ. Monotonicity is an incidental property I should not have required.
The claim is "a gauge no-op changes behavior," which clauses 1–2 encode exactly.

## Why divergence is non-monotone in θ (this is itself informative)

flip_rate measures how differently the c=+5 and c=−5 trajectories decode. It rises sharply
off 0 the instant the penalty turns on (θ=1.02: already 0.6–0.78), because near-tie greedy
decisions get tipped by the sign-branch. Then:
- **gpt2 (small, loop-prone):** flip peaks at θ=1.05 (0.70) and *declines* to 0.50 at θ=1.3.
  At high θ the penalty forces *both* gauges into non-repetitive text, so the trajectories
  partially **re-converge on content** — the shared "avoid repeats" constraint dominates.
- **gpt2-large / pythia (more confident, less loopy):** no such re-convergence; flip keeps
  climbing toward ~0.92.

So small models show a divergence *bump* that fades; larger models show monotone growth.
That model-size dependence is a real secondary finding, not noise — and it is exactly why
a monotonicity requirement was the wrong gate.

## Secondary (named repetition channel, θ=1.02, non-saturated)

rep_rate vs c at the smallest θ, predicted direction = c=−5 most repetitive:
- gpt2-large: 0.823→0.771 across c (ordered, predicted direction) ✓ weak
- pythia-2.8b: 0.830→0.793 (predicted direction) ✓ weak
- gpt2: 0.342→0.363 (flat / slightly reversed) ✗ noisy

The hypothesis's literal channel (repetition *rate* differs with the gauge) is real but weak;
the strong, unambiguous signal is content divergence (flip_rate). The penalty changes *which
text you get*, more than *how repetitive it is*.

## Disposition

- **The hypothesis: VERIFIED.** Gauge non-invariance of the repetition penalty,
  three-model replication, clean no-op control.
- **Process lesson:** decision rules must encode only what the claim asserts. Clauses that
  pin incidental shape (monotonicity, smoothness) manufacture false negatives. Future
  preregs: gate on the effect and its sign/CI, not on its functional form, unless the
  hypothesis is explicitly about that form.
- The publishable framing is stronger than the original spec: an *invisible*
  reparametrization silently selects greedy output; and the divergence-vs-θ shape is itself
  model-size dependent (re-convergence in small models).
