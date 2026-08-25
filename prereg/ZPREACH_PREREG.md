# ZPREACH — ZPREACH: zero-point → raw-steepness → fix-reachability (pre-registered test of the gauge-artifact reading)

Frozen before the analysis is run. Implements **ZPREACH**
exactly as triaged, including the TRIAGE block's **n=7 power caveat** (bimodal predictor ⇒ Spearman is
driven by *family*, not by 7 independent draws), which is implemented here as a **mandatory
within-group gate**, not as a footnote.

This experiment runs **no new decode**. Data collection (the 2 missing a1_zeropoint zero-points and the 3
missing FIXCAL runs) is executing separately; ZPREACH is the frozen **assembly + adjudication**
of those outputs.

## 1. The question

FIXCAL (FIXCAL) produced the migration map θ_raw → θ′_fix at matched repetition suppression and found
the deployed band partly out of reach (`experiments/FIXCAL/FINDINGS.md`). Its **pre-registered**
mechanism — amendment A / E2, "UNREACHABLE tracks in-loop top-1 confidence" — **FAILED** (UNREACHABLE
group mean p_top 0.8738 vs MATCHED 0.8763: wrong direction, no graded re-cut rescues it).

The surviving reading is **post-hoc**, and is the amendment-B one:

> raw's deployed-band suppression strength is a **gauge artifact** of an extreme logit zero-point (nearly
> all penalized tokens take HF's *multiply* branch on large-negative logits), and fix-reachability is
> limited by **how hard RAW drives the metric**, not by how weak the fix is.

Per the FIXCAL frozen rider (2) of 2026-08-18 that explanation **stays out of every downstream proposal and every
paper section until it survives a pre-registered test**. ZPREACH is that test. It is the *only* thing
standing between the measured maps and a mechanism claim; it does not gate the maps themselves.

## 2. Decomposition (PROTOCOL.md §1)

- **CORE** — the two-link chain, as an ordering across models:
  **P1** zero-point extremity → raw steepness, and **P2** raw steepness → fix un-reachability at the
  deployed anchor θ_raw = 1.1.
- **CORE (held-out arm)** — **P3**, gpt2-large patterns with gpt2. The request gates on P3, and that is
  accepted here for a specific reason: **gpt2-large, Qwen2.5-7B-Instruct and starcoder2-7b have no
  FIXCAL run yet**, so 3 of the 7 models contribute (ii)/(iii) values that did not exist when the
  post-hoc reading was formed. P3 is the closest thing this design has to an out-of-sample prediction,
  which is why it is load-bearing rather than an embellishment.
- **DIAGNOSTIC (never a gate)** — Spearman ρ, strict monotonicity, the family-level orderings, the
  premise check P3a, and the median top-1 logit as an alternative extremity axis.
- *Saturability / degeneracy:* `reach_score` is **censored above** (UNREACHABLE ⇒ +∞, all such models
  tied) and `steepness` is bounded by the θ=1.0 rep_rate. Both are therefore adjudicated by **rank
  contrasts with explicit tie handling**, never by magnitudes or by a fitted slope.

## 3. Inputs (assembled, not generated)

Frozen cohort, **n = 7**: `gpt2`, `gpt2-large`, `pythia-2.8b`, `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`,
`Qwen2.5-Coder-7B`, `starcoder2-7b`. Model identity is taken from the JSON's `model` field, slugged as
its last path component (so `openai-community/gpt2-large` → `gpt2-large`); filenames are never parsed
for identity (`Qwen2.5-7B` is a prefix of `Qwen2.5-7B-Instruct`).

**(i) Zero-point — a1_zeropoint protocol** (`code/run_a1_zeropoint.py`), read from
`raw_<slug>.json` in `--zeropoint-dirs`, which defaults to **both** known locations,
`<repo>/runs/A1_zeropoint` (where the 2 new runs land) and
`results/a1_zeropoint` (the 5 existing a1_zeropoint raws). Fields:
`zero_point.frac_seen_logit_positive` and `zero_point.median_top1_logit`. **Precedence, frozen:** the
first directory in the list that supplies a model wins; if a later directory supplies the same model
with *different* numbers that is reported as `duplicate_disagreements` and the disposition is
**INVALID** (two different measurements of the same quantity means one of them is stale — resolve it,
do not pick).

**Predictor (P1's x-axis), frozen:** `frac_neg = 1 − frac_seen_logit_positive` — the fraction of
penalized tokens on HF's **multiply** branch. This is the request's "frac-seen-logit-NEGATIVE
(multiply-branch dominance)" and it is the axis the mechanism actually names. `median_top1_logit` is
reported as a second extremity axis and is a **diagnostic only** — it is an absolute logit value and
PROTOCOL §4 forbids gating on one.

**(ii) Raw steepness — FIXCAL raw grid**, read from `raw_<slug>.json` in `--fixcal-dir`
(default `<repo>/runs/FIXCAL`):

> **steepness = raw_grid["1.000000000"].rep_rate − raw_grid["1.050000000"].rep_rate**

i.e. how far the *raw* operator drives the primary suppression metric over the first 0.05 of dial
travel. Grid keys are matched by float value with a 1e-9 tolerance, not by string. `rep_rate` is
FIXCAL's pre-registered primary metric; `longest_run`/`distinct2` are not used here (distinct2
saturates at ~1.0 on gpt2, longest_run floors at 1.0 — FIXCAL §2).

**(iii) Fix reachability at θ_raw = 1.1** — the same FIXCAL raws, `bisection.rep_rate` entry whose
`theta_raw == 1.1` (the Ollama / GPT4All default; matched by float with a 1e-9 tolerance). Recorded
verbatim: `verdict`, and for MATCHED the matched `theta_fix` = θ′ plus the ratio **θ′/θ_raw**; for
UNREACHABLE the `achieved` metric at θ′_max, the `target`, and the `shortfall`.

### 3a. Reachability score (pre-registered, with its exact rank handling)

For correlation and ordering purposes each model gets one continuous number, the **matching cost**:

> **`reach_score` = θ′ if the θ_raw=1.1 verdict is MATCHED · 1.0 if TRIVIAL · +∞ if UNREACHABLE**

**Direction (stated once, used everywhere): LOWER `reach_score` = MORE reachable.** The score is "what
you must pay on the fixed dial to buy raw's suppression at the deployed anchor"; +∞ is "no price on the
dial buys it", i.e. ranked **worst**. TRIVIAL (the unpenalized fix already matches the anchor) is scored
at the bottom of the dial, θ′ = 1.0 — the *most* reachable value — and is flagged in the report together
with its `raw_suppresses` field, because a TRIVIAL anchor is also the low-steepness extreme and the two
readings must not be silently conflated.

**Exact rank handling, frozen:**
- Ranks are **midranks** (tied values share the mean of the ranks they span), ascending, so rank 1 =
  smallest = most reachable.
- **All UNREACHABLE models are tied at +∞** and therefore share one midrank block at the top. They are
  *not* separated by their shortfalls, and the shortfall is never used as a tiebreaker — a shortfall is
  a distance in metric space, not on the θ′ dial, and ordering by it would smuggle an absolute magnitude
  into the gate. Shortfalls are reported in the table.
- In the concordant/discordant counts (§5) any pair **tied on either variable** — including any
  UNREACHABLE/UNREACHABLE pair — is excluded from both counts and reported as `T`. Censoring therefore
  costs power; it never manufactures agreement.
- Spearman ρ is computed as the Pearson correlation of these midrank vectors (tie-correct by
  construction). ρ is **descriptive**; no clause of the frozen rule reads it.

### 3b. Group split (pre-registered)

> **EXTREME** = `frac_seen_logit_positive < 0.5` (the multiply branch dominates) ·
> **NON-EXTREME** = `frac_seen_logit_positive ≥ 0.5`

The threshold is **mechanism-defined, not data-defined**: 0.5 is the point at which HF's sign branch
flips from mostly-divide to mostly-multiply, which is the exact quantity the gauge-artifact reading
names. It is not a tuned cut, and it is not an absolute *logit* value (PROTOCOL §4).

**Pre-registered honesty note.** the a1_zeropoint run's zero-points for 5 of the 7 models are already published, so
their group membership is knowable before this analysis runs: gpt2 0.091 → EXTREME; **gpt2-large 0.746,
starcoder2-7b 0.828, pythia-2.8b 0.938, Qwen2.5-Coder-7B 0.986 → all NON-EXTREME**; Qwen2.5-7B 0.961 →
NON-EXTREME. The split is therefore expected to be **{gpt2} vs the other six**, which makes the
within-group check (§5 gate 3) run on the NON-EXTREME group and makes it exactly the question the
TRIAGE power caveat asks: *does anything survive once gpt2 is not doing all the work?* This is recorded
here, before the fact, rather than discovered afterwards.

## 4. Predictions (frozen — verbatim from the request)

- **P1:** raw steepness increases with zero-point extremity (frac-seen-logit-**NEGATIVE** — multiply-branch dominance).
- **P2:** reachability at 1.1 decreases with raw steepness.
- **P3:** gpt2-large patterns with gpt2 (both extreme zero-points per the a1_zeropoint table).

Operationalised, in the score directions of §3a:

- **P1** ⇒ `frac_neg` ↑ with `steepness` ↑ (predicted-concordant direction: **positive**).
- **P2** ⇒ `steepness` ↑ with `reach_score` ↑ (less reachable; predicted direction: **positive**).
- **P3** ⇒ **P3b** gpt2-large is on the same side of the cohort as gpt2 on `steepness`, **and**
  **P3c** gpt2-large is on the same side of the cohort as gpt2 on `reach_score`. "Side" is computed on
  **midranks, not values**: a model is `high` iff its midrank exceeds the **lower median** of the
  cohort's midrank vector (the order statistic at index ⌈n/2⌉−1 of the ascending midranks), else `low`.
  Ranks rather than values, because `reach_score` is censored at +∞ — with ≥ half the cohort
  UNREACHABLE a *value*-median is itself +∞ and the clause would pass unconditionally, i.e. a gate that
  cannot fail is not a gate. Ranks stay discriminating under censoring. The one case that remains
  genuinely uninformative — **every** model tied on the axis — is flagged `degenerate: true` and said
  so in the report. Both models' sides, ranks and raw values are reported, not just the boolean.
  **P3a (premise check, REPORTED, NOT A GATE):** is gpt2-large actually extreme, `frac_pos < 0.5`?
  Per the a1_zeropoint run's published table it is **0.746**, so this premise is **expected to fail** — the request's
  parenthetical "(both extreme zero-points per the a1_zeropoint table)" is contradicted by the a1_zeropoint run's own numbers. This is
  flagged now, before the fact. It does **not** soften P3: see §5 D1.

## 5. FROZEN DECISION RULE

**ZPREACH adjudicates ORDERINGS, never magnitudes.** Every gate below is a *contrast* — a count of
concordant vs discordant model pairs, or a same-side-of-the-cohort comparison on midranks. No clause
reads a ρ value, a p-value, an absolute zero-point, an absolute steepness, or an absolute θ′.

0. **Completeness gate.** The rule is adjudicated only on the complete frozen cohort of n=7. If any
   model is missing any of its three quantities, the disposition is **PENDING**: the partial table,
   partial orderings and partial ρ are printed and labelled **PROVISIONAL**, no verdict on P1/P2/P3 is
   binding, and nothing in the output may be cited in a downstream proposal or the paper. Missing models are listed
   by name with which quantity is absent. (**INVALID** instead, if a mandatory input is self-inconsistent
   — see §3's `duplicate_disagreements`, or a FIXCAL raw whose `controls.noop_raw1_eq_fix1` /
   `controls.instrumented_eq_raw1` is false: FIXCAL's own no-op control failing means that model's
   curves are a harness artifact, so debug, do not interpret.)

1. **ρ is DESCRIPTIVE.** Spearman ρ (P1: `frac_neg` vs `steepness`; P2: `steepness` vs `reach_score`) is
   reported overall and within groups, with its rank vectors and tie counts. **No gate reads it**, and
   in particular there is **no ρ threshold anywhere in this rule** — with n=7 and a bimodal predictor a
   ρ near 1.0 is exactly what two clusters produce, so it carries no information the ordering gates do
   not already carry.

2. **Gate 1 — overall ordering (P1 and P2).** For each prediction count, over all model pairs, the
   **concordant** (C) and **discordant** (D) pairs against the predicted direction; pairs tied on either
   variable are excluded and reported as **T**. The ordering **HOLDS iff C > D**. **Both P1 and P2 must
   HOLD.** `strict_monotone` (D = 0) is reported as a stronger descriptive flag and is **never gated**
   (PROTOCOL §4: never gate on incidental shape).

3. **Gate 2 — P3 (held-out arm).** P3 HOLDS iff **P3b and P3c** both hold as defined in §4. P3a is
   reported as a premise check and is **not** a gate.

4. **Gate 3 — within-group ordering (mandatory before the mechanism may be called established).**
   Split the cohort by §3b. Within every group holding ≥3 models with non-degenerate variation,
   recompute the P1 and P2 orderings (C vs D, same tie handling) and report them **per group** alongside
   the overall ones. The mechanism may be called established only if the predicted orderings **HOLD
   within the largest such group** — i.e. only if the ordering survives with the between-family contrast
   removed. A group of <3 models, or one with no computable pair (all tied), is reported as
   **not computable**, which does **not** satisfy this gate.

5. **Gate 4 — family confound (the TRIAGE power caveat, as a gate).**
   `family_confounded` = **TRUE** iff the overall orderings hold (gate 1) while the within-group check
   of gate 3 **fails or is not computable**. When TRUE:
   - `effective_n` is set to the number of non-empty predictor groups (**2**),
   - the report states plainly, in the bottom line: **"this is 2 effective points, not 7"** — the result
     is "the GPT-2-family model with the extreme zero-point differs from the rest", restated,
   - and the disposition is **MECHANISM-NOT-ESTABLISHED**, *regardless* of ρ, of gates 1–2, and of how
     large the between-group separation is.
   When FALSE, `effective_n` = the number of models contributing to the surviving within-group ordering.
   Both fields are written to `summary.json`.

6. **Disposition.**
   - **MECHANISM-SUPPORTED** iff gate 1 (P1 **and** P2 overall) **and** gate 2 (P3) **and** gate 3
     (within-group) all hold **and** `family_confounded` is FALSE. Only then may downstream proposals and the paper
     assert the gauge-artifact reading — and then only as **descriptive cross-model evidence at n=7**,
     scoped: it is an ordering across 7 checkpoints, never a within-model causal claim, and never a
     licence to predict a specific θ′ from a zero-point.
   - **MECHANISM-NOT-ESTABLISHED** otherwise (any gate fails, or family-only separation).
   - **PENDING** if the cohort is incomplete; **INVALID** if an input is self-inconsistent (§5.0).

7. **Failure branch (frozen, from the request, in force verbatim).** On **MECHANISM-NOT-ESTABLISHED**
   the mechanism **stays open**: *Downstream proposals and the paper keep the measured maps* (FIXCAL's calibration
   table, MATCHED's matched-suppression head-to-head, the a1_zeropoint run's zero-point table) *and* **drop the
   gauge-artifact explanation entirely** — it is removed, not softened, not hedged, not relegated to a
   footnote. No re-cut, no re-gate, no alternative predictor substituted after seeing the table: any
   such reading is recorded as post-hoc and non-binding, and would need a fresh pre-registration on
   checkpoints outside this cohort. This is the third mechanism claim in this workstream (E2, then the
   post-hoc zero-point reading); the discipline that killed E2 applies unchanged here.

**D1 — the one anticipated post-hoc reading, disarmed in advance.** If P3 fails *and* P3a shows
gpt2-large is not extreme, someone will observe that this is *consistent* with the mechanism
(gpt2-large should not pattern with gpt2 if it does not share gpt2's zero-point). That observation is
**recorded as a diagnostic and is explicitly not a rescue**: P3 counts as FAILED for gating, the
disposition follows §5.6, and the failure branch fires. Reason: the premise was falsifiable from
the a1_zeropoint run's already-published table before this analysis ran (§3b, §4), so "the premise was wrong" cannot be
discovered from these results and cannot be used to re-cut them. If the mechanism-derived version of P3
("a model patterns with gpt2 iff it shares gpt2's extreme zero-point") is worth testing, it needs its
own pre-registration and its own checkpoints.

*Explicitly NOT gates anywhere in this rule: any ρ threshold or p-value; any absolute
frac-seen-logit-positive, median top-1 logit, steepness or θ′ value; monotonicity or smoothness of any
curve; the size of any shortfall; the median-top1-logit ordering.*

## 6. Outputs

`experiments/ZPREACH/analyze_zpreach.py` — pure assembly + adjudication, no model loads, no GPU, no new
decode:

```
python code/analyze_zpreach.py \
    --zeropoint-dirs results/a1_zeropoint \
    --fixcal-dir results/fix_calibration \
    --out-dir results/zeropoint_reachability
```

writes `REPORT.md` (presence table; the n=7 per-model table; P1/P2/P3 orderings with C/D/T and
descriptive ρ; the within-group orderings; the `effective_n` / `family_confounded` judgement; the
disposition and the branch it fires; §5 quoted verbatim) and `summary.json` (the same, structured,
including `effective_n`, `family_confounded`, `missing`, and every per-model quantity).

Cost: seconds, CPU, stdlib only (Spearman is hand-rolled — no scipy).
