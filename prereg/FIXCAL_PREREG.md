# FIXCAL — FIXCAL: the migration map θ_raw → θ′_fix at MATCHED repetition suppression

Frozen before running. Implements FIXCAL exactly as
triaged: **bisection, not a fixed θ′ grid**, a **first-class UNREACHABLE verdict**, the map under
**all three** suppression metrics, plus the author's amendments **A** (in-loop confidence) and **B**
(interpretation of UNREACHABLE), and the protocol refinement (**report the closed-form required θ′**).

Reuses the fix_loopcheck harness unmodified: `code/run_a1_loopcheck.py::greedy` (greedy decode with
the HF repetition penalty, optional `--fix` = `log_softmax` *before* penalizing) and `::degen`
(rep_rate / distinct2 / longest_run), on `code/run_a1.py::PROMPTS` (16 prompts).

## 1. The question

the fix_loopcheck run established that the fixed (normalize-before-penalize) operator still breaks loops, but is much
gentler per unit θ. Every downstream proposals (to SGLang, mistral.rs, llama.cpp, transformers)
needs a migration note: *"your `repetition_penalty=1.1` ≈ our θ′ = X"*. This experiment measures that
map — or establishes that it does not exist.

**CORE (what is measured):** for each raw anchor θ_raw, the θ′ at which the fixed operator delivers the
**same suppression** as the raw operator at θ_raw — per model, per suppression metric — with a
first-class **UNREACHABLE** outcome when no such θ′ ≤ θ′_max exists.

This is a **calibration measurement**, not a structure-existence test: the deliverable is a map plus
its verdict classes. There is no "confirmed/refuted" headline; §4 of PROTOCOL.md still binds in the
form that matters here — **gate and report on contrasts only, never on an absolute magnitude.**

## 2. Decomposition (PROTOCOL.md §1)

- **CORE** — the matched-suppression map θ_raw → θ′_fix per model, with its verdict class
  (MATCHED / UNREACHABLE / TRIVIAL) per anchor.
- **EMBELLISHMENT E1** — the map is *metric-independent* (the same map whether you match on rep_rate,
  longest_run, or distinct2). Adjudicated separately; never conjoined into the CORE.
- **EMBELLISHMENT E2 (amendment A)** — UNREACHABLE is *explained* by in-loop top-1 confidence, and
  quantitatively by the closed-form θ′_required. Adjudicated separately.
- *Saturability:* `distinct2` saturates at 1.0 (it did in A1) and `longest_run` floors at 1.0 — both are
  therefore **secondary**; `rep_rate` (unbounded away from its ends in this regime) is the **primary**
  matching metric. All three are reported.

## 3. Protocol

**Decode.** Greedy, no sampling, inference-only, KV-cached, long generations: `--max-new 256`
(fix_loopcheck's 128 is a floor, not the setting). Fixed 16-prompt A1 set. dtype float32 (bfloat16 for the 7Bs;
`greedy()` casts logits to fp32 before the argmax, so the decision is fp32 either way).

**Suppression metrics** (mean over the 16 prompts), all from `degen()`:
`rep_rate` (fraction of generated tokens already in prompt ∪ earlier generation — *primary*),
`longest_run` (longest run of one repeated token), `distinct2` (distinct bigram fraction).
Suppression increases as rep_rate ↓, longest_run ↓, distinct2 ↑.

**(a) Raw dense grid.** θ_raw ∈ {1.02, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3}, plus θ = 1.0 as the
unpenalized baseline. All three metrics recorded at every point.

**(b) Bisection (per anchor × per metric, independently).** Target = that anchor's raw metric value.
Search θ′ ∈ [1.0, θ′_max] with **θ′_max = 10** (pre-registered ceiling), standard bisection on the
(assumed monotone) fix metric-vs-θ′ curve. Stop when |metric(θ′) − target| ≤ tol or after 12
iterations, whichever is first. Per-metric tolerances, frozen: **rep_rate 0.01, distinct2 0.01,
longest_run 0.25** (longest_run is a mean of integers and lives on a different scale; it is expected to
exhaust iterations rather than converge — the reported bracket carries that).
Every θ′ evaluation is a full 16-prompt generation sweep and is **cached by θ′** (bisection midpoints
are dyadic and coincide across anchors/metrics, so the cache does most of the work).

**Verdict classes — exactly three, per (anchor, metric), decided at the ENDPOINTS first, then bisected;
reported, never substituted:**
- `TRIVIAL` — the *unpenalized* fix (θ′ = 1.0) already matches the anchor within tol. Almost always
  means the raw operator did not actually suppress at that θ_raw; carries `raw_suppresses` either way.
  Reported, but excluded from the headline migration map.
- `UNREACHABLE` — the fix at θ′_max falls short of the anchor's suppression **by more than tol**.
  Recorded as `reachable: false` **with the achieved metric at θ′_max and the shortfall**. This is a
  RESULT, not an error, and it is an *actual evaluation at θ′_max*, never an extrapolation.
  **No nearby (θ_raw, θ′) pair is ever substituted for an UNREACHABLE anchor.**
- `MATCHED` — otherwise the anchor is bracketed within tolerance by [1, θ′_max] and is bisected toward
  the exact target; reported θ′ = the first midpoint with |Δmetric| ≤ tol (so it is the *smallest*
  matching θ′ to bisection resolution, not the ceiling). If the iteration budget is exhausted first,
  the reported θ′ is the final upper bracket `hi` (the smallest evaluated θ′ reaching *at least* the
  anchor's suppression), with `converged: false` and the bracket [lo, hi] carried in the JSON.

Monotonicity of either curve is **reported descriptively, never gated** (PROTOCOL §4). Where the fix
curve is non-monotone, the endpoint verdicts (TRIVIAL / UNREACHABLE) are unaffected — they are direct
evaluations — while an interior MATCHED θ′ is *one* match, not necessarily the unique one; the report
says so.

**(c) Closed-form required θ′ (protocol refinement).** Under the fix every logit is a log-prob, hence
negative, so HF's sign branch (`v<0 → v*θ`, else `v/θ`) **always multiplies**: the penalized token's
score becomes `θ·log p = log(p^θ)`. Power-scaling is monotone, so it cannot reorder two tokens that are
*both* penalized; the greedy argmax leaves a penalized top token only when some competitor overtakes it:

>  **p_top^θ′ < p_comp  ⟺  θ′·ln p_top < ln p_comp  ⟺  θ′ > ln(p_comp) / ln(p_top)**  (both logs < 0)
>
>  **θ′_required = ln(p_runner) / ln(p_top)**

Worked: (0.70, 0.20) → 4.5; (0.90, 0.05) → 28; (0.99, 0.005) → 527. Loops made of confident tokens are
structurally out of reach of any deployable θ′.

During the **raw θ = 1.0 (unpenalized) greedy run** — a separate instrumented pass over the same 16
prompts — we log, at every loop position: `p_top`, `p_runner`, and `theta_required`.
- **`p_runner` (the spec's definition, headline):** the highest-probability token that is **not** the
  argmax. `theta_required = ln(p_runner)/ln(p_top)`.
- **`p_runner_unseen` (exact, also logged):** the highest-probability token **not in the penalized
  (seen) set**. Because power-scaling preserves the order *within* the penalized set, this is the only
  competitor that can actually take the argmax under the fix. Since `p_runner_unseen ≤ p_runner`,
  `theta_required` (headline) is a **lower bound** on `theta_required_unseen` (exact). Both are
  reported; the headline stays the spec's form.

**Loop-token definition (frozen, exact).** A generated token at position *i* is a **loop token** iff
`gen[i] ∈ prompt_ids ∪ {gen[0..i-1]}` — i.e. exactly the repeat events `rep_rate` counts in
`run_a1_loopcheck.degen()`, and exactly the condition "this token is in the set the penalty acts on".
(A **strict** variant, `gen[i] ∈ {gen[0..i-1]}` only, excluding prompt tokens, is logged and reported
alongside; the primary is the rep_rate-consistent one, because the penalty's `seen` set includes the
prompt.) θ′_required is a *static* per-position quantity read off the unpenalized trajectory; once a
flip occurs the trajectory diverges, so it is a per-step requirement, not a simulated rollout.

**Guards (frozen).** If `p_top ≥ 1 − 1e-6` (ln p_top ≈ 0) or `p_runner ≤ 1e-12`, θ′_required is treated
as **+∞**; such tokens are excluded from the finite distribution and **counted separately**
(`n_ptop_saturated`, `n_prunner_underflow`, `frac_infinite`). Quantiles that land in the infinite mass
are reported as infinite (JSON `null` + an explicit flag), never silently dropped.

**Reported per model:** mean/median/p10/p90 of in-loop `p_top` (amendment A's confidence metric), the
distribution (median, p10, p90) of θ′_required and θ′_required_unseen, the fraction infinite, and the
fraction of loop tokens whose θ′_required exceeds θ′_max.

## 4. Mandatory controls (PROTOCOL.md §3, §5)

- **No-op control (must return the null exactly).** raw θ=1.0 vs fix θ=1.0 must produce *identical*
  per-prompt (rep_rate, distinct2, longest_run). `log_softmax` is strictly monotone and the penalty is
  the identity at θ=1, so any difference means the harness is wrong → **INVALID, debug, do not
  interpret.**
- **Instrumentation control.** The instrumented θ=1.0 pass must reproduce the raw θ=1.0 sweep's
  per-prompt metrics exactly (same trajectory), or the loop statistics describe a different run.
- **Bisection bracket control.** Every bisection reports the bracket it terminated on plus the number
  of fresh sweeps; UNREACHABLE requires an actual evaluation at θ′_max, never an extrapolation.
- **Truncation control (§5).** `max_new` is fixed at 256 for every θ, raw and fix alike, so no verdict
  can be produced by one arm being cut off earlier than another; the raw grid at the same `max_new` is
  the reference.

## 5. FROZEN REPORTING RULE

**FIXCAL is a calibration measurement, not a hypothesis test: it reports a map, and every claim is a
CONTRAST — never an absolute magnitude.**

1. **Validity.** The no-op control (raw θ=1.0 ≡ fix θ=1.0, per prompt, exactly) and the instrumentation
   control must both pass, else **INVALID** — debug the harness, do not interpret. An anchor whose raw
   metric does not improve on the unpenalized baseline is reported as `TRIVIAL / raw_suppresses:false`
   and excluded from the headline map; that is a property of the anchor, never evidence about the fix.
2. **Per anchor × per metric the verdict is exactly one of `MATCHED`, `UNREACHABLE`, `TRIVIAL`**, always
   reported. `UNREACHABLE` carries the achieved metric at θ′_max and the shortfall. **No nearby pair is
   ever substituted for an UNREACHABLE anchor**, in this report or in the MATCHED experiment.
3. **Headline = the migration map itself**, reported under all three matching metrics side by side.
   `rep_rate` is primary; `longest_run` and `distinct2` are reported and never conjoined into a single
   verdict.
4. **Metric-consistency (E1) is judged as a contrast between spreads, not a cutoff.** The map is called
   *metric-independent* iff (a) all three metrics give the same verdict class at every anchor **and**
   (b) at anchors MATCHED under all three, the across-metric spread of θ′ (max/min ratio) is smaller
   than the across-anchor spread of θ′ under the primary metric. Otherwise *metric-dependent*, which is
   reported as weakening any single-number migration note.
5. **Branch (i) — deployed anchors MATCHED.** If the deployed band (θ_raw ∈ {1.02 ≈ ExLlama's 1.025,
   1.05, 1.1 = Ollama/GPT4All default, 1.15}) is MATCHED, report the map and state the ratio
   θ′_fix / θ_raw explicitly (expected θ′ ≫ θ_raw); the migration note is "your θ_raw ≈ our θ′", and
   MATCHED proceeds on those pairs.
6. **Branch (ii) — any deployed anchor UNREACHABLE.** Report the **closed-form required θ′** — the
   per-anchor quantile form (below) plus the median/p90 of the per-token θ′_required distribution — as
   a **structural impossibility** claim: *"you would need θ′ ≈ X"*, not *"we did not find one below
   10"*, per the protocol refinement. Per amendment B the reading is: the raw
   operator's suppression strength at θ ≈ 1.1 is substantially a **gauge artifact** (it is strong
   *because* it reads the arbitrary zero-point), and the normalized form is **not a drop-in with a
   rescaled dial**. Consequences, frozen now: downstream proposals (to SGLang, mistral.rs, llama.cpp, transformers) are **PAUSED**, and the recommendation narrows to **normalized-multiplicative for
   gauge-correctness + subtractive presence/frequency penalties for loop-breaking** (subtractive is
   shift-invariant *and* effective on confident tokens, since z − α hits all logits equally — the
   combination the hosted APIs converged on; cf. STACKS-SURVEY class (c)). MATCHED runs only on anchors
   that are MATCHED.
7. **Amendment-A prediction (E2), pre-registered, reported either way.** UNREACHABLE should track
   in-loop confidence. Adjudicated as a **cross-model CONTRAST**, never an absolute θ′ threshold
   (PROTOCOL §4): **E2 HOLDS iff every model whose deployed anchor is UNREACHABLE ranks strictly above
   every model whose deployed anchor is MATCHED on BOTH (a) mean in-loop top-1 probability and
   (b) median closed-form θ′_required** (∞ ranks highest). If every model falls on the same side of the
   deployed-anchor verdict, the contrast is **not computable** and only the per-model statistics are
   reported. Reported as diagnostics but explicitly NOT gated: sign(median θ′_required − θ′_max), the
   fraction of loop tokens with θ′_required > θ′_max, and the **per-anchor closed-form required θ′** =
   quantile over loop tokens of θ′_required at level q = 1 − target/loop_frac (to drop the loop rate to
   an anchor's target you must flip that fraction of loop tokens; flip the cheapest first). That
   per-anchor form is *static* — it is read off the unpenalized trajectory and so ignores the cascade
   (one flip can derail the rest of a loop), making it an **upper bound** on what the empirical
   bisection needs; the empirical bisection is the result, the closed form is the explanation. A
   failure of this prediction does not change the map; it is reported as the mechanism being
   unexplained.

*(Explicitly NOT gates: any fixed θ′ threshold, any fixed rep_rate level, monotonicity or smoothness of
either curve. Monotonicity is reported descriptively — a non-monotone or model-chaotic map is itself a
reportable finding that weakens the migration story, per FIXCAL's expected-signature note.)*

## 6. Models & outputs

`gpt2`, `EleutherAI/pythia-2.8b`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-Coder-7B` (all cached;
float32 / bfloat16 for the 7Bs). Inference-only, greedy, no sampling.

`runs/FIXCAL/raw_<model>.json` — raw grid curves (all 3 metrics), the evaluated θ′ cache, per-anchor
bisection results per matching metric, the controls, and the loop-confidence / θ′_required statistics.
`analyze_fixcal.py --raws '<glob>' --out-dir runs/FIXCAL` → `REPORT.md` + `summary.json`.

Cost: one θ′ evaluation = one 16-prompt × 256-token greedy sweep. Per model the run is
8 raw sweeps + 1 instrumented sweep + the fix sweeps (2 endpoints + bisection; bisection midpoints are
dyadic and cached, and the UNREACHABLE branch costs **zero** extra sweeps, so the count is bounded well
below 21 searches × 12 iterations). Exact commands in `run_fixcal.py`'s docstring.
