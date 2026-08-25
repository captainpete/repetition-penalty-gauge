# FIXCAL — migration map θ_raw → θ′_fix at matched repetition suppression

Models: 7 · `gpt2` (float32, max_new=256, n_prompts=16, θ′max=10), `gpt2-large` (float32, max_new=256, n_prompts=16, θ′max=10), `pythia-2.8b` (float32, max_new=256, n_prompts=16, θ′max=10), `Qwen2.5-7B` (bfloat16, max_new=256, n_prompts=16, θ′max=10), `Qwen2.5-7B-Instruct` (bfloat16, max_new=256, n_prompts=16, θ′max=10), `Qwen2.5-Coder-7B` (bfloat16, max_new=256, n_prompts=16, θ′max=10), `starcoder2-7b` (bfloat16, max_new=256, n_prompts=16, θ′max=10)

**Validity: controls PASS**

**Branch: (ii) UNREACHABLE at a deployed anchor** → downstream proposals: **PAUSED**

## Frozen reporting rule (PREREG.md §5, verbatim)

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

## 0. Controls (PROTOCOL §3/§5)

| model | no-op raw θ=1 ≡ fix θ′=1 | instrumented θ=1 ≡ raw θ=1 | rep_rate @θ=1 | fix sweeps |
|---|:--:|:--:|--:|--:|
| gpt2 | PASS | PASS | 0.9385 | 10 |
| gpt2-large | PASS | PASS | 0.8870 | 27 |
| pythia-2.8b | PASS | PASS | 0.8608 | 27 |
| Qwen2.5-7B | PASS | PASS | 0.6135 | 14 |
| Qwen2.5-7B-Instruct | PASS | PASS | 0.5854 | 12 |
| Qwen2.5-Coder-7B | PASS | PASS | 0.6885 | 18 |
| starcoder2-7b | PASS | PASS | 0.8638 | 28 |

## gpt2

### Raw dense grid (all three suppression metrics)

| θ_raw | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1 (baseline) | 0.9385 | 1.438 | 0.0865 |
| 1.02 | 0.3831 | 2.000 | 0.8703 |
| 1.05 | 0.1392 | 1.312 | 0.9946 |
| 1.08 | 0.0833 | 1.188 | 0.9973 |
| 1.1 | 0.0654 | 1.125 | 0.9971 |
| 1.15 | 0.0439 | 1.125 | 0.9995 |
| 1.2 | 0.0303 | 1.125 | 0.9993 |
| 1.3 | 0.0122 | 1.125 | 1.0000 |

### Fix curve — every evaluated θ′ (bisection cache)

| θ′ | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1.0000 | 0.9385 | 1.438 | 0.0865 |
| 2.1250 | 0.5723 | 2.000 | 0.6449 |
| 2.6875 | 0.4258 | 2.000 | 0.8218 |
| 2.8281 | 0.3950 | 2.000 | 0.8603 |
| 2.8633 | 0.3831 | 2.000 | 0.8718 |
| 2.8984 | 0.3450 | 2.000 | 0.9152 |
| 2.9688 | 0.3601 | 2.000 | 0.8951 |
| 3.2500 | 0.3523 | 2.000 | 0.8870 |
| 5.5000 | 0.2178 | 2.000 | 0.9679 |
| 10.0000 | 0.1521 | 2.000 | 0.9794 |

### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)

`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated θ′ reaching at least the anchor's suppression (bracket in summary.json).

| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | raw suppresses? | verdicts agree? |
|--:|---|---|---|:--:|:--:|
| 1.02 | 2.863 | TRIVIAL (θ′=1.0; raw did not suppress) | 2.863 | yes | NO |
| 1.05 | **UNREACHABLE** (θ′max: 0.1521 vs 0.1392) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.9794 vs 0.9946) | yes | NO |
| 1.08 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0833) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.9794 vs 0.9973) | yes | NO |
| 1.1 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0654) | **UNREACHABLE** (θ′max: 2.0000 vs 1.1250) | **UNREACHABLE** (θ′max: 0.9794 vs 0.9971) | yes | yes |
| 1.15 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0439) | **UNREACHABLE** (θ′max: 2.0000 vs 1.1250) | **UNREACHABLE** (θ′max: 0.9794 vs 0.9995) | yes | yes |
| 1.2 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0303) | **UNREACHABLE** (θ′max: 2.0000 vs 1.1250) | **UNREACHABLE** (θ′max: 0.9794 vs 0.9993) | yes | yes |
| 1.3 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0122) | **UNREACHABLE** (θ′max: 2.0000 vs 1.1250) | **UNREACHABLE** (θ′max: 0.9794 vs 1.0000) | yes | yes |

### Deployed band (called out)

| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | θ′/θ_raw | closed-form θ′ needed (static) |
|--:|---|--:|---|--:|--:|--:|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 0.3831 | MATCHED | 2.863 | 2.81× | 1724.1 |
| 1.05 | — | 0.1392 | **UNREACHABLE** | **UNREACHABLE** (θ′max: 0.1521 vs 0.1392) | — | 6459.7 |
| 1.1 | Ollama / GPT4All default | 0.0654 | **UNREACHABLE** | **UNREACHABLE** (θ′max: 0.1521 vs 0.0654) | — | 12316.6 |
| 1.15 | — | 0.0439 | **UNREACHABLE** | **UNREACHABLE** (θ′max: 0.1521 vs 0.0439) | — | 15623.2 |

*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac (flip the cheapest loop tokens first). STATIC — read off the unpenalized trajectory, so it ignores the cascade (one flip can derail the rest of a loop) and is an upper bound on what the empirical bisection needs. Diagnostic, not a gate; the empirical column is the result.

### Loop confidence + closed-form required θ′ (raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)

- loop tokens: 3844/4096 (loop_frac 0.9385; strict, prompt tokens excluded: 3816, 0.9316)
- **in-loop top-1 probability** (amendment A): mean 0.9183, median 0.9942, p10 0.7125, p90 0.9990  ·  all-token mean 0.8787
- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): median 1099.74, p10 7.36, p90 8921.97; frac ∞ 0.0000; frac > θ′max 0.8910
- **θ′_required (exact, best UNSEEN competitor)**: median 1131.93, p90 9344.64; frac > θ′max 0.9006
- guards: p_top ≥ 1−1e-6 on 0 loop tokens; p_runner ≤ 1e-12 on 0 (both → θ′_required = ∞)

### Monotonicity of the two curves (DESCRIPTIVE — never a gate)

| curve | metric | points | violations (suppression falls as θ rises) | max drop | at θ |
|---|---|--:|--:|--:|---|
| raw | rep_rate | 8 | 0 | 0.0000 | — |
| raw | longest_run | 8 | 1 | 0.5625 | 1.02 |
| raw | distinct2 | 8 | 2 | 0.0002 | 1.1, 1.2 |
| fix | rep_rate | 10 | 1 | 0.0151 | 2.96875 |
| fix | longest_run | 10 | 1 | 0.5625 | 2.125 |
| fix | distinct2 | 10 | 2 | 0.0201 | 2.96875, 3.25 |

Non-monotone: raw/longest_run, raw/distinct2, fix/rep_rate, fix/longest_run, fix/distinct2. The bisection assumes a monotone metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, not an extrapolation) but the interior θ′ is one match, not necessarily the unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a finding that weakens the migration story).

### Metric-consistency (E1, spread contrast)
- verdict classes agree at every anchor: **NO**
- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): n/a
- across-anchor θ′ spread under the primary metric: n/a
- **metric-independence not computable** (no anchor MATCHED under all three metrics, and/or <2 MATCHED anchors under the primary metric) — the verdict-class agreement above carries E1 on its own

## gpt2-large

### Raw dense grid (all three suppression metrics)

| θ_raw | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1 (baseline) | 0.8870 | 1.750 | 0.1730 |
| 1.02 | 0.8464 | 1.938 | 0.2365 |
| 1.05 | 0.7861 | 1.875 | 0.3316 |
| 1.08 | 0.7368 | 1.938 | 0.4017 |
| 1.1 | 0.6372 | 2.000 | 0.5559 |
| 1.15 | 0.4668 | 2.000 | 0.8051 |
| 1.2 | 0.3928 | 2.000 | 0.8588 |
| 1.3 | 0.2832 | 2.000 | 0.9414 |

### Fix curve — every evaluated θ′ (bisection cache)

| θ′ | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1.0000 | 0.8870 | 1.750 | 0.1730 |
| 1.0703 | 0.8638 | 1.812 | 0.2066 |
| 1.1055 | 0.8457 | 1.938 | 0.2324 |
| 1.1406 | 0.8276 | 1.938 | 0.2664 |
| 1.2812 | 0.8311 | 1.812 | 0.2618 |
| 1.3516 | 0.8047 | 1.812 | 0.2993 |
| 1.3691 | 0.7837 | 1.875 | 0.3360 |
| 1.3867 | 0.7656 | 1.875 | 0.3591 |
| 1.4219 | 0.7424 | 1.875 | 0.4037 |
| 1.5625 | 0.7561 | 1.875 | 0.3664 |
| 1.6328 | 0.7368 | 1.875 | 0.4037 |
| 1.7031 | 0.6711 | 1.875 | 0.5201 |
| 1.7207 | 0.6555 | 1.938 | 0.5458 |
| 1.7295 | 0.6470 | 1.938 | 0.5522 |
| 1.7383 | 0.6348 | 1.938 | 0.5723 |
| 1.7734 | 0.6184 | 1.938 | 0.5890 |
| 1.8438 | 0.6089 | 2.000 | 0.6078 |
| 2.1250 | 0.5220 | 2.000 | 0.7289 |
| 2.4062 | 0.4695 | 2.000 | 0.8103 |
| 2.6875 | 0.4434 | 2.000 | 0.8314 |
| 2.9688 | 0.4111 | 2.000 | 0.8674 |
| 3.2500 | 0.3899 | 2.000 | 0.8789 |
| 4.3750 | 0.3284 | 2.000 | 0.9225 |
| 4.9375 | 0.2939 | 2.000 | 0.9409 |
| 5.2188 | 0.2764 | 2.000 | 0.9527 |
| 5.5000 | 0.2629 | 2.000 | 0.9603 |
| 10.0000 | 0.1682 | 2.000 | 0.9819 |

### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)

`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated θ′ reaching at least the anchor's suppression (bracket in summary.json).

| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | raw suppresses? | verdicts agree? |
|--:|---|---|---|:--:|:--:|
| 1.02 | 1.105 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.105 | yes | NO |
| 1.05 | 1.369 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.369 | yes | NO |
| 1.08 | 1.633 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.633 | yes | NO |
| 1.1 | 1.738 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.729 | yes | NO |
| 1.15 | 2.406 | TRIVIAL (θ′=1.0; raw did not suppress) | 2.406 | yes | NO |
| 1.2 | 3.250 | TRIVIAL (θ′=1.0; raw did not suppress) | 2.969 | yes | NO |
| 1.3 | 5.219 | TRIVIAL (θ′=1.0; raw did not suppress) | 4.938 | yes | NO |

### Deployed band (called out)

| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | θ′/θ_raw | closed-form θ′ needed (static) |
|--:|---|--:|---|--:|--:|--:|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 0.8464 | MATCHED | 1.105 | 1.08× | 1.5 |
| 1.05 | — | 0.7861 | MATCHED | 1.369 | 1.30× | 3.8 |
| 1.1 | Ollama / GPT4All default | 0.6372 | MATCHED | 1.738 | 1.58× | 91.5 |
| 1.15 | — | 0.4668 | MATCHED | 2.406 | 2.09× | 1031.6 |

*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac (flip the cheapest loop tokens first). STATIC — read off the unpenalized trajectory, so it ignores the cascade (one flip can derail the rest of a loop) and is an upper bound on what the empirical bisection needs. Diagnostic, not a gate; the empirical column is the result.

### Loop confidence + closed-form required θ′ (raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)

- loop tokens: 3633/4096 (loop_frac 0.8870; strict, prompt tokens excluded: 3597, 0.8782)
- **in-loop top-1 probability** (amendment A): mean 0.8874, median 0.9949, p10 0.5073, p90 0.9992  ·  all-token mean 0.8295
- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): median 1295.10, p10 3.07, p90 10653.50; frac ∞ 0.0000; frac > θ′max 0.8310
- **θ′_required (exact, best UNSEEN competitor)**: median 1414.88, p90 11086.65; frac > θ′max 0.8412
- guards: p_top ≥ 1−1e-6 on 0 loop tokens; p_runner ≤ 1e-12 on 0 (both → θ′_required = ∞)

### Monotonicity of the two curves (DESCRIPTIVE — never a gate)

| curve | metric | points | violations (suppression falls as θ rises) | max drop | at θ |
|---|---|--:|--:|--:|---|
| raw | rep_rate | 8 | 0 | 0.0000 | — |
| raw | longest_run | 8 | 3 | 0.1875 | 1.02, 1.08, 1.1 |
| raw | distinct2 | 8 | 0 | 0.0000 | — |
| fix | rep_rate | 27 | 2 | 0.0137 | 1.28125, 1.5625 |
| fix | longest_run | 27 | 5 | 0.1250 | 1.07031, 1.10547, 1.36914, 1.7207, 1.84375 |
| fix | distinct2 | 27 | 2 | 0.0373 | 1.28125, 1.5625 |

Non-monotone: raw/longest_run, fix/rep_rate, fix/longest_run, fix/distinct2. The bisection assumes a monotone metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, not an extrapolation) but the interior θ′ is one match, not necessarily the unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a finding that weakens the migration story).

### Metric-consistency (E1, spread contrast)
- verdict classes agree at every anchor: **NO**
- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): n/a
- across-anchor θ′ spread under the primary metric: 4.721
- **metric-independence not computable** (no anchor MATCHED under all three metrics, and/or <2 MATCHED anchors under the primary metric) — the verdict-class agreement above carries E1 on its own

## pythia-2.8b

### Raw dense grid (all three suppression metrics)

| θ_raw | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1 (baseline) | 0.8608 | 1.750 | 0.2191 |
| 1.02 | 0.8313 | 1.750 | 0.2728 |
| 1.05 | 0.6848 | 2.000 | 0.4946 |
| 1.08 | 0.5745 | 2.000 | 0.6600 |
| 1.1 | 0.5095 | 2.000 | 0.7456 |
| 1.15 | 0.3767 | 2.000 | 0.8652 |
| 1.2 | 0.2747 | 2.000 | 0.9576 |
| 1.3 | 0.1641 | 1.938 | 0.9757 |

### Fix curve — every evaluated θ′ (bisection cache)

| θ′ | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1.0000 | 0.8608 | 1.750 | 0.2191 |
| 1.1406 | 0.8545 | 1.750 | 0.2267 |
| 1.2109 | 0.8372 | 1.750 | 0.2549 |
| 1.2461 | 0.8230 | 1.812 | 0.2787 |
| 1.2812 | 0.8142 | 1.750 | 0.2971 |
| 1.5625 | 0.7256 | 1.812 | 0.4395 |
| 1.6328 | 0.7136 | 1.875 | 0.4551 |
| 1.6680 | 0.6931 | 1.812 | 0.4919 |
| 1.7031 | 0.6807 | 1.812 | 0.5221 |
| 1.8438 | 0.6726 | 1.875 | 0.5137 |
| 2.1250 | 0.6072 | 1.938 | 0.6208 |
| 2.1602 | 0.5854 | 1.875 | 0.6544 |
| 2.1953 | 0.5671 | 1.938 | 0.6821 |
| 2.2656 | 0.5554 | 1.938 | 0.6831 |
| 2.4062 | 0.5195 | 1.938 | 0.7500 |
| 2.4766 | 0.5244 | 2.000 | 0.7510 |
| 2.5117 | 0.5088 | 2.000 | 0.7699 |
| 2.5469 | 0.4990 | 2.000 | 0.7777 |
| 2.6875 | 0.4917 | 2.000 | 0.7777 |
| 3.2500 | 0.4531 | 1.938 | 0.8201 |
| 3.5312 | 0.4082 | 1.875 | 0.8669 |
| 3.8125 | 0.3857 | 1.875 | 0.8870 |
| 4.3750 | 0.3484 | 1.875 | 0.9130 |
| 5.5000 | 0.3286 | 1.875 | 0.9181 |
| 6.6250 | 0.2720 | 2.000 | 0.9495 |
| 7.7500 | 0.2483 | 1.938 | 0.9608 |
| 10.0000 | 0.2109 | 2.062 | 0.9635 |

### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)

`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated θ′ reaching at least the anchor's suppression (bracket in summary.json).

| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | raw suppresses? | verdicts agree? |
|--:|---|---|---|:--:|:--:|
| 1.02 | 1.211 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.246 | yes | NO |
| 1.05 | 1.703 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.668 | yes | NO |
| 1.08 | 2.195 | TRIVIAL (θ′=1.0; raw did not suppress) | 2.160 | yes | NO |
| 1.1 | 2.512 | TRIVIAL (θ′=1.0; raw did not suppress) | 2.406 | yes | NO |
| 1.15 | 3.812 | TRIVIAL (θ′=1.0; raw did not suppress) | 3.531 | yes | NO |
| 1.2 | 6.625 | TRIVIAL (θ′=1.0; raw did not suppress) | 7.750 | yes | NO |
| 1.3 | **UNREACHABLE** (θ′max: 0.2109 vs 0.1641) | TRIVIAL (θ′=1.0; raw did not suppress) | **UNREACHABLE** (θ′max: 0.9635 vs 0.9757) | yes | NO |

### Deployed band (called out)

| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | θ′/θ_raw | closed-form θ′ needed (static) |
|--:|---|--:|---|--:|--:|--:|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 0.8313 | MATCHED | 1.211 | 1.19× | 1.3 |
| 1.05 | — | 0.6848 | MATCHED | 1.703 | 1.62× | 13.1 |
| 1.1 | Ollama / GPT4All default | 0.5095 | MATCHED | 2.512 | 2.28× | 194.8 |
| 1.15 | — | 0.3767 | MATCHED | 3.812 | 3.32× | 1016.2 |

*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac (flip the cheapest loop tokens first). STATIC — read off the unpenalized trajectory, so it ignores the cascade (one flip can derail the rest of a loop) and is an upper bound on what the empirical bisection needs. Diagnostic, not a gate; the empirical column is the result.

### Loop confidence + closed-form required θ′ (raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)

- loop tokens: 3526/4096 (loop_frac 0.8608; strict, prompt tokens excluded: 3499, 0.8542)
- **in-loop top-1 probability** (amendment A): mean 0.8763, median 0.9894, p10 0.4967, p90 0.9996  ·  all-token mean 0.8143
- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): median 552.05, p10 2.64, p90 23610.10; frac ∞ 0.0009; frac > θ′max 0.8128
- **θ′_required (exact, best UNSEEN competitor)**: median 606.32, p90 24249.15; frac > θ′max 0.8298
- guards: p_top ≥ 1−1e-6 on 3 loop tokens; p_runner ≤ 1e-12 on 0 (both → θ′_required = ∞)

### Monotonicity of the two curves (DESCRIPTIVE — never a gate)

| curve | metric | points | violations (suppression falls as θ rises) | max drop | at θ |
|---|---|--:|--:|--:|---|
| raw | rep_rate | 8 | 0 | 0.0000 | — |
| raw | longest_run | 8 | 1 | 0.2500 | 1.05 |
| raw | distinct2 | 8 | 0 | 0.0000 | — |
| fix | rep_rate | 27 | 1 | 0.0049 | 2.47656 |
| fix | longest_run | 27 | 9 | 0.1250 | 1.24609, 1.5625, 1.63281, 1.84375, 2.125, 2.19531, 2.47656, 6.625, 10 |
| fix | distinct2 | 27 | 1 | 0.0083 | 1.84375 |

Non-monotone: raw/longest_run, fix/rep_rate, fix/longest_run, fix/distinct2. The bisection assumes a monotone metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, not an extrapolation) but the interior θ′ is one match, not necessarily the unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a finding that weakens the migration story).

### Metric-consistency (E1, spread contrast)
- verdict classes agree at every anchor: **NO**
- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): n/a
- across-anchor θ′ spread under the primary metric: 5.471
- **metric-independence not computable** (no anchor MATCHED under all three metrics, and/or <2 MATCHED anchors under the primary metric) — the verdict-class agreement above carries E1 on its own

## Qwen2.5-7B

### Raw dense grid (all three suppression metrics)

| θ_raw | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1 (baseline) | 0.6135 | 1.250 | 0.6542 |
| 1.02 | 0.5615 | 1.375 | 0.7309 |
| 1.05 | 0.5308 | 1.188 | 0.7520 |
| 1.08 | 0.4575 | 1.312 | 0.8353 |
| 1.1 | 0.4021 | 1.312 | 0.8605 |
| 1.15 | 0.2922 | 1.062 | 0.9172 |
| 1.2 | 0.1792 | 1.062 | 0.9637 |
| 1.3 | 0.0679 | 1.062 | 0.9892 |

### Fix curve — every evaluated θ′ (bisection cache)

| θ′ | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1.0000 | 0.6135 | 1.250 | 0.6542 |
| 1.2812 | 0.5593 | 1.250 | 0.7375 |
| 1.5625 | 0.5479 | 1.312 | 0.7512 |
| 1.8438 | 0.5383 | 1.250 | 0.7635 |
| 2.1250 | 0.5066 | 1.312 | 0.8042 |
| 2.6875 | 0.4863 | 1.125 | 0.8123 |
| 2.9688 | 0.4656 | 1.312 | 0.8429 |
| 3.2500 | 0.4641 | 1.188 | 0.8468 |
| 4.3750 | 0.4392 | 1.125 | 0.8547 |
| 5.5000 | 0.4016 | 1.188 | 0.8733 |
| 7.7500 | 0.3430 | 1.312 | 0.8966 |
| 8.8750 | 0.3342 | 1.250 | 0.8995 |
| 9.4375 | 0.3210 | 1.250 | 0.9137 |
| 10.0000 | 0.3188 | 1.188 | 0.9103 |

### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)

`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated θ′ reaching at least the anchor's suppression (bracket in summary.json).

| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | raw suppresses? | verdicts agree? |
|--:|---|---|---|:--:|:--:|
| 1.02 | 1.281 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.281 | yes | NO |
| 1.05 | 1.844 | TRIVIAL (θ′=1.0) | 1.562 | yes | NO |
| 1.08 | 3.250 | TRIVIAL (θ′=1.0; raw did not suppress) | 2.969 | yes | NO |
| 1.1 | 5.500 | TRIVIAL (θ′=1.0; raw did not suppress) | 4.375 | yes | NO |
| 1.15 | **UNREACHABLE** (θ′max: 0.3188 vs 0.2922) | TRIVIAL (θ′=1.0) | 9.438 | yes | NO |
| 1.2 | **UNREACHABLE** (θ′max: 0.3188 vs 0.1792) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.9103 vs 0.9637) | yes | NO |
| 1.3 | **UNREACHABLE** (θ′max: 0.3188 vs 0.0679) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.9103 vs 0.9892) | yes | NO |

### Deployed band (called out)

| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | θ′/θ_raw | closed-form θ′ needed (static) |
|--:|---|--:|---|--:|--:|--:|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 0.5615 | MATCHED | 1.281 | 1.26× | 1.6 |
| 1.05 | — | 0.5308 | MATCHED | 1.844 | 1.76× | 2.2 |
| 1.1 | Ollama / GPT4All default | 0.4021 | MATCHED | 5.500 | 5.00× | 20.0 |
| 1.15 | — | 0.2922 | **UNREACHABLE** | **UNREACHABLE** (θ′max: 0.3188 vs 0.2922) | — | 278.4 |

*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac (flip the cheapest loop tokens first). STATIC — read off the unpenalized trajectory, so it ignores the cascade (one flip can derail the rest of a loop) and is an upper bound on what the empirical bisection needs. Diagnostic, not a gate; the empirical column is the result.

### Loop confidence + closed-form required θ′ (raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)

- loop tokens: 2513/4096 (loop_frac 0.6135; strict, prompt tokens excluded: 2470, 0.6030)
- **in-loop top-1 probability** (amendment A): mean 0.8443, median 0.9760, p10 0.4709, p90 0.9999  ·  all-token mean 0.7670
- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): median 187.64, p10 1.78, p90 205697.58; frac ∞ 0.0191; frac > θ′max 0.7139
- **θ′_required (exact, best UNSEEN competitor)**: median 217.49, p90 221967.77; frac > θ′max 0.7585
- guards: p_top ≥ 1−1e-6 on 48 loop tokens; p_runner ≤ 1e-12 on 0 (both → θ′_required = ∞)

### Monotonicity of the two curves (DESCRIPTIVE — never a gate)

| curve | metric | points | violations (suppression falls as θ rises) | max drop | at θ |
|---|---|--:|--:|--:|---|
| raw | rep_rate | 8 | 0 | 0.0000 | — |
| raw | longest_run | 8 | 2 | 0.1250 | 1.02, 1.08 |
| raw | distinct2 | 8 | 0 | 0.0000 | — |
| fix | rep_rate | 14 | 0 | 0.0000 | — |
| fix | longest_run | 14 | 5 | 0.1875 | 1.5625, 2.125, 2.96875, 5.5, 7.75 |
| fix | distinct2 | 14 | 1 | 0.0034 | 10 |

Non-monotone: raw/longest_run, fix/longest_run, fix/distinct2. The bisection assumes a monotone metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, not an extrapolation) but the interior θ′ is one match, not necessarily the unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a finding that weakens the migration story).

### Metric-consistency (E1, spread contrast)
- verdict classes agree at every anchor: **NO**
- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): n/a
- across-anchor θ′ spread under the primary metric: 4.293
- **metric-independence not computable** (no anchor MATCHED under all three metrics, and/or <2 MATCHED anchors under the primary metric) — the verdict-class agreement above carries E1 on its own

## Qwen2.5-7B-Instruct

### Raw dense grid (all three suppression metrics)

| θ_raw | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1 (baseline) | 0.5854 | 1.062 | 0.7174 |
| 1.02 | 0.5718 | 1.188 | 0.7385 |
| 1.05 | 0.5127 | 1.062 | 0.8059 |
| 1.08 | 0.4600 | 1.125 | 0.8475 |
| 1.1 | 0.4312 | 1.125 | 0.8586 |
| 1.15 | 0.3789 | 1.062 | 0.8841 |
| 1.2 | 0.2703 | 1.000 | 0.9385 |
| 1.3 | 0.1223 | 1.062 | 0.9792 |

### Fix curve — every evaluated θ′ (bisection cache)

| θ′ | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1.0000 | 0.5854 | 1.062 | 0.7174 |
| 1.2812 | 0.5942 | 1.062 | 0.6936 |
| 1.3516 | 0.5698 | 1.000 | 0.7424 |
| 1.4219 | 0.5623 | 1.000 | 0.7522 |
| 1.5625 | 0.5449 | 1.062 | 0.7784 |
| 2.1250 | 0.5105 | 1.188 | 0.8093 |
| 3.2500 | 0.4714 | 1.125 | 0.8485 |
| 3.8125 | 0.4553 | 1.188 | 0.8534 |
| 4.3750 | 0.4458 | 1.188 | 0.8691 |
| 4.9375 | 0.4246 | 1.188 | 0.8816 |
| 5.5000 | 0.4202 | 1.188 | 0.8934 |
| 10.0000 | 0.3928 | 1.125 | 0.8811 |

### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)

`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated θ′ reaching at least the anchor's suppression (bracket in summary.json).

| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | raw suppresses? | verdicts agree? |
|--:|---|---|---|:--:|:--:|
| 1.02 | 1.422 | TRIVIAL (θ′=1.0; raw did not suppress) | 1.352 | yes | NO |
| 1.05 | 2.125 | TRIVIAL (θ′=1.0; raw did not suppress) | 2.125 | yes | NO |
| 1.08 | 3.812 | TRIVIAL (θ′=1.0; raw did not suppress) | 3.250 | yes | NO |
| 1.1 | 4.938 | TRIVIAL (θ′=1.0; raw did not suppress) | 3.812 | yes | NO |
| 1.15 | **UNREACHABLE** (θ′max: 0.3928 vs 0.3789) | TRIVIAL (θ′=1.0; raw did not suppress) | 5.500 | yes | NO |
| 1.2 | **UNREACHABLE** (θ′max: 0.3928 vs 0.2703) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.8811 vs 0.9385) | yes | NO |
| 1.3 | **UNREACHABLE** (θ′max: 0.3928 vs 0.1223) | TRIVIAL (θ′=1.0; raw did not suppress) | **UNREACHABLE** (θ′max: 0.8811 vs 0.9792) | yes | NO |

### Deployed band (called out)

| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | θ′/θ_raw | closed-form θ′ needed (static) |
|--:|---|--:|---|--:|--:|--:|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 0.5718 | MATCHED | 1.422 | 1.39× | 1.1 |
| 1.05 | — | 0.5127 | MATCHED | 2.125 | 2.02× | 1.8 |
| 1.1 | Ollama / GPT4All default | 0.4312 | MATCHED | 4.938 | 4.49× | 4.7 |
| 1.15 | — | 0.3789 | **UNREACHABLE** | **UNREACHABLE** (θ′max: 0.3928 vs 0.3789) | — | 11.6 |

*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac (flip the cheapest loop tokens first). STATIC — read off the unpenalized trajectory, so it ignores the cascade (one flip can derail the rest of a loop) and is an upper bound on what the empirical bisection needs. Diagnostic, not a gate; the empirical column is the result.

### Loop confidence + closed-form required θ′ (raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)

- loop tokens: 2398/4096 (loop_frac 0.5854; strict, prompt tokens excluded: 2341, 0.5715)
- **in-loop top-1 probability** (amendment A): mean 0.8225, median 0.9514, p10 0.4457, p90 1.0000  ·  all-token mean 0.7557
- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): median 69.27, p10 1.57, p90 3856061.83; frac ∞ 0.0717; frac > θ′max 0.6593
- **θ′_required (exact, best UNSEEN competitor)**: median 98.23, p90 4158154.42; frac > θ′max 0.7139
- guards: p_top ≥ 1−1e-6 on 172 loop tokens; p_runner ≤ 1e-12 on 0 (both → θ′_required = ∞)

### Monotonicity of the two curves (DESCRIPTIVE — never a gate)

| curve | metric | points | violations (suppression falls as θ rises) | max drop | at θ |
|---|---|--:|--:|--:|---|
| raw | rep_rate | 8 | 0 | 0.0000 | — |
| raw | longest_run | 8 | 3 | 0.1250 | 1.02, 1.08, 1.3 |
| raw | distinct2 | 8 | 0 | 0.0000 | — |
| fix | rep_rate | 12 | 1 | 0.0088 | 1.28125 |
| fix | longest_run | 12 | 3 | 0.1250 | 1.5625, 2.125, 3.8125 |
| fix | distinct2 | 12 | 2 | 0.0238 | 1.28125, 10 |

Non-monotone: raw/longest_run, fix/rep_rate, fix/longest_run, fix/distinct2. The bisection assumes a monotone metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, not an extrapolation) but the interior θ′ is one match, not necessarily the unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a finding that weakens the migration story).

### Metric-consistency (E1, spread contrast)
- verdict classes agree at every anchor: **NO**
- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): n/a
- across-anchor θ′ spread under the primary metric: 3.473
- **metric-independence not computable** (no anchor MATCHED under all three metrics, and/or <2 MATCHED anchors under the primary metric) — the verdict-class agreement above carries E1 on its own

## Qwen2.5-Coder-7B

### Raw dense grid (all three suppression metrics)

| θ_raw | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1 (baseline) | 0.6885 | 1.188 | 0.5402 |
| 1.02 | 0.6282 | 1.062 | 0.6262 |
| 1.05 | 0.5168 | 1.125 | 0.7875 |
| 1.08 | 0.4648 | 1.125 | 0.8216 |
| 1.1 | 0.3745 | 1.062 | 0.8949 |
| 1.15 | 0.2539 | 1.062 | 0.9468 |
| 1.2 | 0.1404 | 1.062 | 0.9787 |
| 1.3 | 0.0535 | 1.000 | 0.9887 |

### Fix curve — every evaluated θ′ (bisection cache)

| θ′ | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1.0000 | 0.6885 | 1.188 | 0.5402 |
| 1.2812 | 0.6296 | 1.188 | 0.6277 |
| 1.5625 | 0.5869 | 1.250 | 0.6971 |
| 2.1250 | 0.5598 | 1.188 | 0.7159 |
| 3.2500 | 0.5100 | 1.375 | 0.7887 |
| 3.5312 | 0.4890 | 1.438 | 0.8034 |
| 3.5400 | 0.4880 | 1.438 | 0.8044 |
| 3.5444 | 0.4751 | 1.438 | 0.8289 |
| 3.5488 | 0.4714 | 1.438 | 0.8341 |
| 3.5664 | 0.4714 | 1.438 | 0.8360 |
| 3.6016 | 0.4670 | 1.375 | 0.8414 |
| 3.6719 | 0.4688 | 1.375 | 0.8407 |
| 3.8125 | 0.4736 | 1.375 | 0.8355 |
| 4.3750 | 0.4651 | 1.250 | 0.8478 |
| 5.5000 | 0.4219 | 1.125 | 0.8730 |
| 6.6250 | 0.3777 | 1.375 | 0.8914 |
| 7.7500 | 0.3628 | 1.312 | 0.9025 |
| 10.0000 | 0.3289 | 1.062 | 0.9000 |

### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)

`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated θ′ reaching at least the anchor's suppression (bracket in summary.json).

| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | raw suppresses? | verdicts agree? |
|--:|---|---|---|:--:|:--:|
| 1.02 | 1.281 | TRIVIAL (θ′=1.0) | 1.281 | yes | NO |
| 1.05 | 3.250 | TRIVIAL (θ′=1.0) | 3.250 | yes | NO |
| 1.08 | 4.375 | TRIVIAL (θ′=1.0) | 3.544 | yes | NO |
| 1.1 | 6.625 | TRIVIAL (θ′=1.0) | 7.750 | yes | NO |
| 1.15 | **UNREACHABLE** (θ′max: 0.3289 vs 0.2539) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.9000 vs 0.9468) | yes | NO |
| 1.2 | **UNREACHABLE** (θ′max: 0.3289 vs 0.1404) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.9000 vs 0.9787) | yes | NO |
| 1.3 | **UNREACHABLE** (θ′max: 0.3289 vs 0.0535) | TRIVIAL (θ′=1.0) | **UNREACHABLE** (θ′max: 0.9000 vs 0.9887) | yes | NO |

### Deployed band (called out)

| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | θ′/θ_raw | closed-form θ′ needed (static) |
|--:|---|--:|---|--:|--:|--:|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 0.6282 | MATCHED | 1.281 | 1.26× | 1.7 |
| 1.05 | — | 0.5168 | MATCHED | 3.250 | 3.10× | 9.0 |
| 1.1 | Ollama / GPT4All default | 0.3745 | MATCHED | 6.625 | 6.02× | 206.7 |
| 1.15 | — | 0.2539 | **UNREACHABLE** | **UNREACHABLE** (θ′max: 0.3289 vs 0.2539) | — | 3893.5 |

*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac (flip the cheapest loop tokens first). STATIC — read off the unpenalized trajectory, so it ignores the cascade (one flip can derail the rest of a loop) and is an upper bound on what the empirical bisection needs. Diagnostic, not a gate; the empirical column is the result.

### Loop confidence + closed-form required θ′ (raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)

- loop tokens: 2820/4096 (loop_frac 0.6885; strict, prompt tokens excluded: 2773, 0.6770)
- **in-loop top-1 probability** (amendment A): mean 0.8588, median 0.9883, p10 0.4866, p90 0.9999  ·  all-token mean 0.7830
- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): median 425.24, p10 1.94, p90 141794.95; frac ∞ 0.0234; frac > θ′max 0.7404
- **θ′_required (exact, best UNSEEN competitor)**: median 541.48, p90 150067.30; frac > θ′max 0.7716
- guards: p_top ≥ 1−1e-6 on 66 loop tokens; p_runner ≤ 1e-12 on 0 (both → θ′_required = ∞)

### Monotonicity of the two curves (DESCRIPTIVE — never a gate)

| curve | metric | points | violations (suppression falls as θ rises) | max drop | at θ |
|---|---|--:|--:|--:|---|
| raw | rep_rate | 8 | 0 | 0.0000 | — |
| raw | longest_run | 8 | 1 | 0.0625 | 1.05 |
| raw | distinct2 | 8 | 0 | 0.0000 | — |
| fix | rep_rate | 18 | 2 | 0.0049 | 3.67188, 3.8125 |
| fix | longest_run | 18 | 4 | 0.2500 | 1.5625, 3.25, 3.53125, 6.625 |
| fix | distinct2 | 18 | 3 | 0.0051 | 3.67188, 3.8125, 10 |

Non-monotone: raw/longest_run, fix/rep_rate, fix/longest_run, fix/distinct2. The bisection assumes a monotone metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, not an extrapolation) but the interior θ′ is one match, not necessarily the unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a finding that weakens the migration story).

### Metric-consistency (E1, spread contrast)
- verdict classes agree at every anchor: **NO**
- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): n/a
- across-anchor θ′ spread under the primary metric: 5.171
- **metric-independence not computable** (no anchor MATCHED under all three metrics, and/or <2 MATCHED anchors under the primary metric) — the verdict-class agreement above carries E1 on its own

## starcoder2-7b

### Raw dense grid (all three suppression metrics)

| θ_raw | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1 (baseline) | 0.8638 | 15.625 | 0.2074 |
| 1.02 | 0.8477 | 16.062 | 0.2250 |
| 1.05 | 0.7634 | 2.375 | 0.3586 |
| 1.08 | 0.5940 | 2.625 | 0.6037 |
| 1.1 | 0.5312 | 2.625 | 0.7083 |
| 1.15 | 0.4368 | 2.000 | 0.8100 |
| 1.2 | 0.3608 | 2.000 | 0.8686 |
| 1.3 | 0.2566 | 1.688 | 0.9255 |

### Fix curve — every evaluated θ′ (bisection cache)

| θ′ | rep_rate | longest_run | distinct2 |
|--:|--:|--:|--:|
| 1.0000 | 0.8638 | 15.625 | 0.2074 |
| 1.0703 | 0.8760 | 15.625 | 0.1814 |
| 1.1055 | 0.8589 | 15.625 | 0.2113 |
| 1.1230 | 0.8582 | 15.625 | 0.2098 |
| 1.1252 | 0.8369 | 1.625 | 0.2471 |
| 1.1274 | 0.8362 | 1.625 | 0.2468 |
| 1.1318 | 0.8362 | 1.625 | 0.2468 |
| 1.1406 | 0.8330 | 1.625 | 0.2502 |
| 1.2812 | 0.7832 | 1.750 | 0.3343 |
| 1.4219 | 0.8010 | 2.375 | 0.3074 |
| 1.4570 | 0.7712 | 2.188 | 0.3517 |
| 1.4922 | 0.7581 | 2.188 | 0.3716 |
| 1.5625 | 0.7468 | 2.188 | 0.3887 |
| 2.1250 | 0.5967 | 1.938 | 0.6012 |
| 2.4062 | 0.5461 | 2.125 | 0.6735 |
| 2.5469 | 0.5295 | 2.062 | 0.6917 |
| 2.5820 | 0.5173 | 2.062 | 0.7120 |
| 2.6172 | 0.5122 | 2.312 | 0.7248 |
| 2.6875 | 0.5088 | 2.125 | 0.7260 |
| 3.2500 | 0.4863 | 2.062 | 0.7549 |
| 4.3750 | 0.4377 | 2.125 | 0.8005 |
| 5.5000 | 0.3745 | 2.000 | 0.8441 |
| 5.7812 | 0.3806 | 1.875 | 0.8353 |
| 5.9219 | 0.3623 | 1.938 | 0.8586 |
| 6.0625 | 0.3401 | 1.812 | 0.8603 |
| 6.6250 | 0.3240 | 1.750 | 0.8828 |
| 7.7500 | 0.3215 | 1.688 | 0.8767 |
| 10.0000 | 0.2749 | 1.750 | 0.9110 |

### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)

`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated θ′ reaching at least the anchor's suppression (bracket in summary.json).

| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | raw suppresses? | verdicts agree? |
|--:|---|---|---|:--:|:--:|
| 1.02 | ~1.125 | TRIVIAL (θ′=1.0; raw did not suppress) | ~1.125 | yes | NO |
| 1.05 | 1.492 | 1.562 | 1.457 | yes | yes |
| 1.08 | 2.125 | ~1.125 | 2.125 | yes | yes |
| 1.1 | 2.547 | ~1.125 | 2.582 | yes | yes |
| 1.15 | 4.375 | 5.500 | 4.375 | yes | yes |
| 1.2 | 5.922 | 5.500 | 7.750 | yes | yes |
| 1.3 | **UNREACHABLE** (θ′max: 0.2749 vs 0.2566) | 7.750 | **UNREACHABLE** (θ′max: 0.9110 vs 0.9255) | yes | NO |

### Deployed band (called out)

| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | θ′/θ_raw | closed-form θ′ needed (static) |
|--:|---|--:|---|--:|--:|--:|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 0.8477 | MATCHED | ~1.125 | 1.10× | 1.2 |
| 1.05 | — | 0.7634 | MATCHED | 1.492 | 1.42× | 7.8 |
| 1.1 | Ollama / GPT4All default | 0.5312 | MATCHED | 2.547 | 2.32× | 405.3 |
| 1.15 | — | 0.4368 | MATCHED | 4.375 | 3.80× | 1244.6 |

*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac (flip the cheapest loop tokens first). STATIC — read off the unpenalized trajectory, so it ignores the cascade (one flip can derail the rest of a loop) and is an upper bound on what the empirical bisection needs. Diagnostic, not a gate; the empirical column is the result.

### Loop confidence + closed-form required θ′ (raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)

- loop tokens: 3538/4096 (loop_frac 0.8638; strict, prompt tokens excluded: 3504, 0.8555)
- **in-loop top-1 probability** (amendment A): mean 0.9112, median 0.9949, p10 0.6670, p90 0.9999  ·  all-token mean 0.8559
- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): median 1303.62, p10 5.61, p90 70844.97; frac ∞ 0.0006; frac > θ′max 0.8739
- **θ′_required (exact, best UNSEEN competitor)**: median 1482.84, p90 75952.75; frac > θ′max 0.8847
- guards: p_top ≥ 1−1e-6 on 2 loop tokens; p_runner ≤ 1e-12 on 0 (both → θ′_required = ∞)

### Monotonicity of the two curves (DESCRIPTIVE — never a gate)

| curve | metric | points | violations (suppression falls as θ rises) | max drop | at θ |
|---|---|--:|--:|--:|---|
| raw | rep_rate | 8 | 0 | 0.0000 | — |
| raw | longest_run | 8 | 2 | 0.4375 | 1.02, 1.08 |
| raw | distinct2 | 8 | 0 | 0.0000 | — |
| fix | rep_rate | 28 | 3 | 0.0178 | 1.07031, 1.42188, 5.78125 |
| fix | longest_run | 28 | 7 | 0.6250 | 1.28125, 1.42188, 2.40625, 2.61719, 4.375, 5.92188, 10 |
| fix | distinct2 | 28 | 6 | 0.0270 | 1.07031, 1.12305, 1.12744, 1.42188, 5.78125, 7.75 |

Non-monotone: raw/longest_run, fix/rep_rate, fix/longest_run, fix/distinct2. The bisection assumes a monotone metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, not an extrapolation) but the interior θ′ is one match, not necessarily the unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a finding that weakens the migration story).

### Metric-consistency (E1, spread contrast)
- verdict classes agree at every anchor: **NO**
- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): 2.295
- across-anchor θ′ spread under the primary metric: 5.263
- **map is metric-dependent**

## Merged cross-model migration table (primary metric = rep_rate)

| θ_raw | gpt2 | gpt2-large | pythia-2.8b | Qwen2.5-7B | Qwen2.5-7B-Instruct | Qwen2.5-Coder-7B | starcoder2-7b |
|--:|---|---|---|---|---|---|---|
| 1.02 | 2.863 | 1.105 | 1.211 | 1.281 | 1.422 | 1.281 | ~1.125 |
| 1.05 | **UNREACHABLE** (θ′max: 0.1521 vs 0.1392) | 1.369 | 1.703 | 1.844 | 2.125 | 3.250 | 1.492 |
| 1.08 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0833) | 1.633 | 2.195 | 3.250 | 3.812 | 4.375 | 2.125 |
| 1.1 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0654) | 1.738 | 2.512 | 5.500 | 4.938 | 6.625 | 2.547 |
| 1.15 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0439) | 2.406 | 3.812 | **UNREACHABLE** (θ′max: 0.3188 vs 0.2922) | **UNREACHABLE** (θ′max: 0.3928 vs 0.3789) | **UNREACHABLE** (θ′max: 0.3289 vs 0.2539) | 4.375 |
| 1.2 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0303) | 3.250 | 6.625 | **UNREACHABLE** (θ′max: 0.3188 vs 0.1792) | **UNREACHABLE** (θ′max: 0.3928 vs 0.2703) | **UNREACHABLE** (θ′max: 0.3289 vs 0.1404) | 5.922 |
| 1.3 | **UNREACHABLE** (θ′max: 0.1521 vs 0.0122) | 5.219 | **UNREACHABLE** (θ′max: 0.2109 vs 0.1641) | **UNREACHABLE** (θ′max: 0.3188 vs 0.0679) | **UNREACHABLE** (θ′max: 0.3928 vs 0.1223) | **UNREACHABLE** (θ′max: 0.3289 vs 0.0535) | **UNREACHABLE** (θ′max: 0.2749 vs 0.2566) |

### Deployed band, merged (rep_rate)

| θ_raw | engine | gpt2 | gpt2-large | pythia-2.8b | Qwen2.5-7B | Qwen2.5-7B-Instruct | Qwen2.5-Coder-7B | starcoder2-7b |
|--:|---|---|---|---|---|---|---|---|
| 1.02 | ExLlama ~1.025 (1.02 = nearest grid point) | 2.863 | 1.105 | 1.211 | 1.281 | 1.422 | 1.281 | ~1.125 |
| 1.05 | — | **UNREACHABLE** (θ′max: 0.1521 vs 0.1392) | 1.369 | 1.703 | 1.844 | 2.125 | 3.250 | 1.492 |
| 1.1 | Ollama / GPT4All default | **UNREACHABLE** (θ′max: 0.1521 vs 0.0654) | 1.738 | 2.512 | 5.500 | 4.938 | 6.625 | 2.547 |
| 1.15 | — | **UNREACHABLE** (θ′max: 0.1521 vs 0.0439) | 2.406 | 3.812 | **UNREACHABLE** (θ′max: 0.3188 vs 0.2922) | **UNREACHABLE** (θ′max: 0.3928 vs 0.3789) | **UNREACHABLE** (θ′max: 0.3289 vs 0.2539) | 4.375 |

## Loop confidence / required θ′, merged (amendment A / E2)

| model | in-loop p_top mean | in-loop p_top median | median θ′_required | p90 θ′_required | frac θ′_req > θ′max | deployed anchor UNREACHABLE? |
|---|--:|--:|--:|--:|--:|:--:|
| gpt2 | 0.9183 | 0.9942 | 1099.74 | 8921.97 | 0.8910 | YES |
| gpt2-large | 0.8874 | 0.9949 | 1295.10 | 10653.50 | 0.8310 | no |
| pythia-2.8b | 0.8763 | 0.9894 | 552.05 | 23610.10 | 0.8128 | no |
| Qwen2.5-7B | 0.8443 | 0.9760 | 187.64 | 205697.58 | 0.7139 | YES |
| Qwen2.5-7B-Instruct | 0.8225 | 0.9514 | 69.27 | 3856061.83 | 0.6593 | YES |
| Qwen2.5-Coder-7B | 0.8588 | 0.9883 | 425.24 | 141794.95 | 0.7404 | YES |
| starcoder2-7b | 0.9112 | 0.9949 | 1303.62 | 70844.97 | 0.8739 | no |

**Amendment-A prediction (E2), adjudicated as a CROSS-MODEL CONTRAST** (never an absolute θ′ threshold — PROTOCOL §4): UNREACHABLE models should rank above MATCHED models on both in-loop p_top and median θ′_required.
- UNREACHABLE at the deployed anchor: gpt2, Qwen2.5-7B, Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B  ·  MATCHED: gpt2-large, pythia-2.8b, starcoder2-7b
- separates on in-loop p_top: **no** (group means 0.8610 vs 0.8916; contrast -0.0307)
- separates on median θ′_required: **no**
- **E2 FAILS** (both required). A failure does not change the map; it means the mechanism is unexplained.

## Bottom line (frozen rule 5/6)

- `gpt2`: deployed anchors [1.05, 1.1, 1.15] are **UNREACHABLE** at θ′ ≤ 10. Closed form (static, upper bound) says you would need θ_raw=1.05 → θ′ ≈ 6460; θ_raw=1.1 → θ′ ≈ 12317; θ_raw=1.15 → θ′ ≈ 15623. In-loop top-1 p = 0.918 (mean), median per-token θ′_required = 1099.7.
- `Qwen2.5-7B`: deployed anchors [1.15] are **UNREACHABLE** at θ′ ≤ 10. Closed form (static, upper bound) says you would need θ_raw=1.15 → θ′ ≈ 278. In-loop top-1 p = 0.844 (mean), median per-token θ′_required = 187.6.
- `Qwen2.5-7B-Instruct`: deployed anchors [1.15] are **UNREACHABLE** at θ′ ≤ 10. Closed form (static, upper bound) says you would need θ_raw=1.15 → θ′ ≈ 12. In-loop top-1 p = 0.823 (mean), median per-token θ′_required = 69.3.
- `Qwen2.5-Coder-7B`: deployed anchors [1.15] are **UNREACHABLE** at θ′ ≤ 10. Closed form (static, upper bound) says you would need θ_raw=1.15 → θ′ ≈ 3894. In-loop top-1 p = 0.859 (mean), median per-token θ′_required = 425.2.

Per amendment B this is a **structural impossibility**, not a dial that was too small: the raw operator's suppression at the deployed θ is substantially a **gauge artifact** (it is strong *because* it reads the arbitrary zero-point), and the normalized form is **not a drop-in with a rescaled dial**.

**Downstream engine downstream_proposals (SGLang, mistral.rs, llama.cpp, transformers PR) are PAUSED.** The recommendation narrows to: *normalized-multiplicative for gauge-correctness + subtractive presence/frequency penalties for loop-breaking* (subtractive is shift-invariant AND effective on confident tokens, since z − α hits all logits equally). MATCHED runs only on anchors that are MATCHED — no nearby pair is substituted.

Reachable deployed anchors per model (the only pairs MATCHED may use): {"gpt2": [1.02], "gpt2-large": [1.02, 1.05, 1.1, 1.15], "pythia-2.8b": [1.02, 1.05, 1.1, 1.15], "Qwen2.5-7B": [1.02, 1.05, 1.1], "Qwen2.5-7B-Instruct": [1.02, 1.05, 1.1], "Qwen2.5-Coder-7B": [1.02, 1.05, 1.1], "starcoder2-7b": [1.02, 1.05, 1.1, 1.15]}
