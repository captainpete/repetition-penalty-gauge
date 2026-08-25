# MATCHED — matched-SUPPRESSION head-to-head (quality at equal anti-repetition strength)

Raws analysed: 14 · models: Qwen2.5-7B, Qwen2.5-Coder-7B, gpt2, pythia-2.8b

Pairs are FIXCAL `MATCHED` pairs on the primary metric `rep_rate`; the θ′ values were read from the FIXCAL summary at run time and asserted against the frozen FIXCAL table.


## Stage coverage (partial runs are reported, never silently completed)

| model | pairs | gauge | open | json | humaneval (gen) | humaneval (scored) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `Qwen2.5-7B` | ✓ | ✓ | ✓ | · | · | · |
| `Qwen2.5-Coder-7B` | ✓ | ✓ | · | ✓ | ✓ | ✓ |
| `gpt2` | ✓ | ✓ | ✓ | · | · | · |
| `pythia-2.8b` | ✓ | ✓ | ✓ | · | · | · |

## Frozen decision rule (PREREG.md §6, verbatim)

> **Premise/validity (per pair):** the pair must actually be matched — |rep_rate(raw) − rep_rate(fix)| within
> the FIXCAL tolerance (0.01). A pair failing this is reported `MISMATCHED` and EXCLUDED from the quality
> verdict (it is a calibration failure, never evidence about the fix). If no pair survives on a model →
> INVALID for that model.
> **Gauge gate (c):** flip-rate under fix must be 0 (exact) at every surviving pair, and flip-rate under raw
> > 0. Failure here is a **hard REFUTED** regardless of quality metrics — gauge-invariance is the fix's
> entire reason to exist.
> **CORE (quality at equal suppression), per model, gated on CONTRASTS with bootstrap CIs (unit =
> prompt/problem/schema):**
> **DOMINANT** iff at every surviving pair the fix is ≥ raw on the structured metrics (JSON validity, pass@1)
> with the paired CI excluding 0 on at least one, AND not significantly worse on any open-ended metric.
> **TIES** iff all paired CIs include 0 (no reliable difference either way) — the honest "gauge invariance at
> no measurable cost" outcome, which is still a viable recommendation.
> **LOSES** iff the fix is significantly worse (paired CI excluding 0 in raw's favour) on any metric at any
> surviving pair — downstream_proposals pause on that basis, and the losing metric/pair is named.
> Report per pair and per metric; never conjoin metrics into one number. The deployed anchor (θ_raw=1.10) is
> reported separately and called out, since it carries the downstream proposal decision.


*(Operationalisation, frozen in PREREG §6a: contrast = mean over units of (fix − raw) oriented so positive = fix better; paired bootstrap B=10 000, seed 0, 95% percentile CI; higher-is-better = json_valid, humaneval_pass1, quality, distinct1, distinct2; rep_rate is the MATCHING CHECK and longest_run is reported only — neither is scored as quality; evaluation order INVALID → REFUTED → LOSES → DOMINANT → TIES → FAVOURS_FIX.)*


## Verdicts

| model | verdict | why |
|---|---|---|
| `Qwen2.5-7B` | **TIES** | every paired CI includes 0 -- gauge invariance at no measurable cost (still a viable recommendation) |
| `Qwen2.5-Coder-7B` | **FAVOURS_FIX** | not LOSES and not TIES, but the DOMINANT conditions are unmet (no structured arm, or structured CIs include 0). Fix significantly better on: humaneval_pass1@theta_raw=1.1 |
| `gpt2` | **TIES** | every paired CI includes 0 -- gauge invariance at no measurable cost (still a viable recommendation) |
| `pythia-2.8b` | **TIES** | every paired CI includes 0 -- gauge invariance at no measurable cost (still a viable recommendation) |

## 0. Mandatory controls (PROTOCOL §3)

| model | check | value |
|---|---|---|
| `Qwen2.5-7B` | pairs.noop_raw1_eq_fix1 | True |
| `Qwen2.5-7B` | open.noop_raw1_eq_fix1 | True |
| `Qwen2.5-7B` | gauge.noop_flip_raw | 0.0 |
| `Qwen2.5-7B` | gauge.noop_flip_fix | 0.0 |
| `Qwen2.5-7B` | gauge.noop_gate_exact_zero | True |
| `Qwen2.5-7B` | gauge.instrumented_eq_loopcheck | True |
| `Qwen2.5-Coder-7B` | pairs.noop_raw1_eq_fix1 | True |
| `Qwen2.5-Coder-7B` | json.noop_raw1_eq_fix1 | True |
| `Qwen2.5-Coder-7B` | gauge.noop_flip_raw | 0.0 |
| `Qwen2.5-Coder-7B` | gauge.noop_flip_fix | 0.0 |
| `Qwen2.5-Coder-7B` | gauge.noop_gate_exact_zero | True |
| `Qwen2.5-Coder-7B` | gauge.instrumented_eq_loopcheck | True |
| `gpt2` | pairs.noop_raw1_eq_fix1 | True |
| `gpt2` | open.noop_raw1_eq_fix1 | True |
| `gpt2` | gauge.noop_flip_raw | 0.0 |
| `gpt2` | gauge.noop_flip_fix | 0.0 |
| `gpt2` | gauge.noop_gate_exact_zero | True |
| `gpt2` | gauge.instrumented_eq_loopcheck | True |
| `pythia-2.8b` | pairs.noop_raw1_eq_fix1 | True |
| `pythia-2.8b` | open.noop_raw1_eq_fix1 | True |
| `pythia-2.8b` | gauge.noop_flip_raw | 0.0 |
| `pythia-2.8b` | gauge.noop_flip_fix | 0.0 |
| `pythia-2.8b` | gauge.noop_gate_exact_zero | True |
| `pythia-2.8b` | gauge.instrumented_eq_loopcheck | True |
| `Qwen2.5-7B` | determinism (pairs vs open generations) | 0 mismatched |
| `gpt2` | determinism (pairs vs open generations) | 0 mismatched |
| `pythia-2.8b` | determinism (pairs vs open generations) | 0 mismatched |

## Stage 0a — premise: are the pairs actually matched? (|Δrep_rate| ≤ 0.01)

| model | θ_raw | θ′_fix | rep_rate raw | rep_rate fix | \|Δ\| | status | FIXCAL raw drift |
|---|--:|--:|--:|--:|--:|:--:|--:|
| `Qwen2.5-7B` | 1.02 | 1.2812 | 0.5615 | 0.5593 | 0.0022 | MATCHED | 0.0000 |
| `Qwen2.5-7B` | 1.05 | 1.8438 | 0.5308 | 0.5383 | 0.0076 | MATCHED | 0.0000 |
| `Qwen2.5-7B` | 1.1 **(deployed anchor)** | 5.5000 | 0.4021 | 0.4016 | 0.0005 | MATCHED | 0.0000 |
| `Qwen2.5-Coder-7B` | 1.02 | 1.2812 | 0.6282 | 0.6296 | 0.0015 | MATCHED | 0.0000 |
| `Qwen2.5-Coder-7B` | 1.05 | 3.2500 | 0.5168 | 0.5100 | 0.0068 | MATCHED | 0.0000 |
| `Qwen2.5-Coder-7B` | 1.1 **(deployed anchor)** | 6.6250 | 0.3745 | 0.3777 | 0.0032 | MATCHED | 0.0000 |
| `gpt2` | 1.02 | 2.8633 | 0.3831 | 0.3831 | 0.0000 | MATCHED | 0.0000 |
| `pythia-2.8b` | 1.02 | 1.2109 | 0.8313 | 0.8372 | 0.0059 | MATCHED | 0.0000 |
| `pythia-2.8b` | 1.05 | 1.7031 | 0.6848 | 0.6807 | 0.0042 | MATCHED | 0.0000 |
| `pythia-2.8b` | 1.1 **(deployed anchor)** | 2.5117 | 0.5095 | 0.5088 | 0.0007 | MATCHED | 0.0000 |

## Stage 0b — gauge no-regression gate (A1 c=±5 flip-rate at every pair)

flip(fix) must be **exactly 0** and flip(raw) **> 0**. Failure = hard REFUTED.

| model | θ_raw | θ′_fix | flip raw | flip fix | status | sub-diagnosis (reported, not a re-gate) |
|---|--:|--:|--:|--:|:--:|---|
| `Qwen2.5-7B` | 1.02 | 1.2812 | 0.643066 | 0.000000 | PASS | — |
| `Qwen2.5-7B` | 1.05 | 1.8438 | 0.713867 | 0.000000 | PASS | — |
| `Qwen2.5-7B` | 1.1 **(deployed anchor)** | 5.5000 | 0.873535 | 0.000000 | PASS | — |
| `Qwen2.5-Coder-7B` | 1.02 | 1.2812 | 0.677979 | 0.000000 | PASS | — |
| `Qwen2.5-Coder-7B` | 1.05 | 3.2500 | 0.833984 | 0.000000 | PASS | — |
| `Qwen2.5-Coder-7B` | 1.1 **(deployed anchor)** | 6.6250 | 0.905273 | 0.000000 | PASS | — |
| `gpt2` | 1.02 | 2.8633 | 0.707031 | 0.000000 | PASS | — |
| `pythia-2.8b` | 1.02 | 1.2109 | 0.644531 | 0.000000 | PASS | — |
| `pythia-2.8b` | 1.05 | 1.7031 | 0.878174 | 0.000000 | PASS | — |
| `pythia-2.8b` | 1.1 **(deployed anchor)** | 2.5117 | 0.933594 | 0.000000 | PASS | — |

## CORE — quality contrasts at equal suppression (fix − raw, positive = fix better)

**bold CI** = excludes 0. Rows on MISMATCHED pairs are excluded from the verdict and marked.


### `Qwen2.5-7B`  (Qwen/Qwen2.5-7B, bfloat16)

open-ended quality measure: **judge** ({'judge': 'Qwen/Qwen2.5-7B-Instruct', 'scale': [1, 5], 'blind': True, 'randomized_order': True, 'seed': 1234, 'n_unparseable': 0})

quality fallback chain: ["mauve: ModuleNotFoundError: No module named 'mauve'"]

θ=1.0 unpenalized reference: `{"open_theta1": {"rep_rate": 0.613525390625, "distinct1": 0.39697265625, "distinct2": 0.6541666666666667, "quality": 4.0}}`

| θ_raw | metric | class | raw | fix | diff (fix−raw) | 95% CI | n | note |
|--:|---|---|--:|--:|--:|---|--:|---|
| 1.02 | distinct1 | open_ended | 0.4495 | 0.4512 | 0.0017 | [-0.0261, +0.0276] | 16 |  |
| 1.02 | distinct2 | open_ended | 0.7309 | 0.7375 | 0.0066 | [-0.0409, +0.0485] | 16 |  |
| 1.02 | rep_rate | reported_only | 0.5615 | 0.5593 | -0.0022 | [-0.0278, +0.0254] | 16 | reported only, not a quality metric |
| 1.02 | longest_run | reported_only | 1.3750 | 1.2500 | -0.1250 | [-0.3125, +0.0000] | 16 | reported only, not a quality metric |
| 1.02 | quality | open_ended | 4.1250 | 3.8750 | -0.2500 | [-0.6250, +0.1250] | 16 |  |
| 1.05 | distinct1 | open_ended | 0.4792 | 0.4739 | -0.0054 | [-0.0508, +0.0454] | 16 |  |
| 1.05 | distinct2 | open_ended | 0.7520 | 0.7635 | 0.0115 | [-0.0654, +0.1017] | 16 |  |
| 1.05 | rep_rate | reported_only | 0.5308 | 0.5383 | 0.0076 | [-0.0425, +0.0522] | 16 | reported only, not a quality metric |
| 1.05 | longest_run | reported_only | 1.1875 | 1.2500 | 0.0625 | [+0.0000, +0.1875] | 16 | reported only, not a quality metric |
| 1.05 | quality | open_ended | 4.0625 | 4.0000 | -0.0625 | [-0.4375, +0.3125] | 16 |  |
| 1.1 | distinct1 | open_ended | 0.6067 | 0.6079 | 0.0012 | [-0.0430, +0.0405] | 16 | deployed anchor |
| 1.1 | distinct2 | open_ended | 0.8605 | 0.8733 | 0.0127 | [-0.0238, +0.0502] | 16 | deployed anchor |
| 1.1 | rep_rate | reported_only | 0.4021 | 0.4016 | -0.0005 | [-0.0396, +0.0430] | 16 | reported only, not a quality metric; deployed anchor |
| 1.1 | longest_run | reported_only | 1.3125 | 1.1875 | -0.1250 | [-0.3125, +0.0000] | 16 | reported only, not a quality metric; deployed anchor |
| 1.1 | quality | open_ended | 4.1875 | 4.1875 | 0.0000 | [-0.3125, +0.3750] | 16 | deployed anchor |

### `Qwen2.5-Coder-7B`  (Qwen/Qwen2.5-Coder-7B, bfloat16)

θ=1.0 unpenalized reference: `{"json_theta1_valid": 1.0, "humaneval_theta1_pass1": 0.5914634146341463}`

| θ_raw | metric | class | raw | fix | diff (fix−raw) | 95% CI | n | note |
|--:|---|---|--:|--:|--:|---|--:|---|
| 1.02 | json_valid | structured | 1.0000 | 1.0000 | 0.0000 | [+0.0000, +0.0000] | 48 | SATURATED |
| 1.05 | json_valid | structured | 1.0000 | 1.0000 | 0.0000 | [+0.0000, +0.0000] | 48 | SATURATED |
| 1.1 | json_valid | structured | 1.0000 | 1.0000 | 0.0000 | [+0.0000, +0.0000] | 48 | SATURATED; deployed anchor |
| 1.02 | humaneval_pass1 | structured | 0.6037 | 0.5976 | -0.0061 | [-0.0427, +0.0305] | 164 |  |
| 1.05 | humaneval_pass1 | structured | 0.5671 | 0.5427 | -0.0244 | [-0.0610, +0.0122] | 164 |  |
| 1.1 | humaneval_pass1 | structured | 0.4695 | 0.5305 | 0.0610 | **[+0.0122, +0.1159]** | 164 | deployed anchor |

### `gpt2`  (gpt2, float32)

open-ended quality measure: **judge** ({'judge': 'Qwen/Qwen2.5-7B-Instruct', 'scale': [1, 5], 'blind': True, 'randomized_order': True, 'seed': 1234, 'n_unparseable': 0})

quality fallback chain: ["mauve: ModuleNotFoundError: No module named 'mauve'"]

θ=1.0 unpenalized reference: `{"open_theta1": {"rep_rate": 0.9384765625, "distinct1": 0.068359375, "distinct2": 0.08651960784313724, "quality": 1.625}}`

| θ_raw | metric | class | raw | fix | diff (fix−raw) | 95% CI | n | note |
|--:|---|---|--:|--:|--:|---|--:|---|
| 1.02 | distinct1 | open_ended | 0.6238 | 0.6235 | -0.0002 | [-0.0181, +0.0183] | 16 |  |
| 1.02 | distinct2 | open_ended | 0.8703 | 0.8718 | 0.0015 | [-0.0243, +0.0265] | 16 |  |
| 1.02 | rep_rate | reported_only | 0.3831 | 0.3831 | 0.0000 | [-0.0186, +0.0178] | 16 | reported only, not a quality metric |
| 1.02 | longest_run | reported_only | 2.0000 | 2.0000 | 0.0000 | [+0.0000, +0.0000] | 16 | reported only, not a quality metric |
| 1.02 | quality | open_ended | 2.8125 | 2.9375 | 0.1250 | [-0.2500, +0.5000] | 16 |  |

### `pythia-2.8b`  (EleutherAI/pythia-2.8b, float32)

open-ended quality measure: **judge** ({'judge': 'Qwen/Qwen2.5-7B-Instruct', 'scale': [1, 5], 'blind': True, 'randomized_order': True, 'seed': 1234, 'n_unparseable': 0})

quality fallback chain: ["mauve: ModuleNotFoundError: No module named 'mauve'"]

θ=1.0 unpenalized reference: `{"open_theta1": {"rep_rate": 0.86083984375, "distinct1": 0.145751953125, "distinct2": 0.2191176470588235, "quality": 2.4375}}`

| θ_raw | metric | class | raw | fix | diff (fix−raw) | 95% CI | n | note |
|--:|---|---|--:|--:|--:|---|--:|---|
| 1.02 | distinct1 | open_ended | 0.1738 | 0.1685 | -0.0054 | [-0.0244, +0.0139] | 16 |  |
| 1.02 | distinct2 | open_ended | 0.2728 | 0.2549 | -0.0179 | [-0.0542, +0.0186] | 16 |  |
| 1.02 | rep_rate | reported_only | 0.8313 | 0.8372 | 0.0059 | [-0.0134, +0.0251] | 16 | reported only, not a quality metric |
| 1.02 | longest_run | reported_only | 1.7500 | 1.7500 | 0.0000 | [+0.0000, +0.0000] | 16 | reported only, not a quality metric |
| 1.02 | quality | open_ended | 2.7500 | 2.5625 | -0.1875 | [-0.5000, +0.1250] | 16 |  |
| 1.05 | distinct1 | open_ended | 0.3223 | 0.3259 | 0.0037 | [-0.0825, +0.0884] | 16 |  |
| 1.05 | distinct2 | open_ended | 0.4946 | 0.5221 | 0.0275 | [-0.1103, +0.1662] | 16 |  |
| 1.05 | rep_rate | reported_only | 0.6848 | 0.6807 | -0.0042 | [-0.0872, +0.0803] | 16 | reported only, not a quality metric |
| 1.05 | longest_run | reported_only | 2.0000 | 1.8125 | -0.1875 | [-0.3750, +0.0000] | 16 | reported only, not a quality metric |
| 1.05 | quality | open_ended | 3.3750 | 3.3750 | 0.0000 | [-0.5625, +0.5625] | 16 |  |
| 1.1 | distinct1 | open_ended | 0.4980 | 0.4988 | 0.0007 | [-0.0322, +0.0334] | 16 | deployed anchor |
| 1.1 | distinct2 | open_ended | 0.7456 | 0.7699 | 0.0243 | [-0.0211, +0.0728] | 16 | deployed anchor |
| 1.1 | rep_rate | reported_only | 0.5095 | 0.5088 | -0.0007 | [-0.0330, +0.0325] | 16 | reported only, not a quality metric; deployed anchor |
| 1.1 | longest_run | reported_only | 2.0000 | 2.0000 | 0.0000 | [+0.0000, +0.0000] | 16 | reported only, not a quality metric; deployed anchor |
| 1.1 | quality | open_ended | 3.3750 | 3.5625 | 0.1875 | [+0.0000, +0.3750] | 16 | deployed anchor |

## Deployed anchor (θ_raw = 1.10) — reported separately; it carries the downstream proposal decision

- `Qwen2.5-7B`: premise **MATCHED**, gauge **PASS**; distinct1 0.0012 [-0.0430, +0.0405], distinct2 0.0127 [-0.0238, +0.0502], quality 0.0000 [-0.3125, +0.3750]
- `Qwen2.5-Coder-7B`: premise **MATCHED**, gauge **PASS**; json_valid 0.0000 [+0.0000, +0.0000], humaneval_pass1 0.0610 **[+0.0122, +0.1159]**
- `gpt2`: **no 1.10 pair exists** (UNREACHABLE in FIXCAL) — this model cannot speak to the deployed anchor. Documented, never substituted.
- `pythia-2.8b`: premise **MATCHED**, gauge **PASS**; distinct1 0.0007 [-0.0322, +0.0334], distinct2 0.0243 [-0.0211, +0.0728], quality 0.1875 [+0.0000, +0.3750]

## Framing (FIXCAL AUTHOR RULING rider 1 — not a gate)

The 2.5×–6.6× per-model spread of θ′ at the same θ_raw *is A1 measured in suppression space*: raw θ=1.1 was never one intervention to begin with, so "calibrate per model" is not a defect of the fix relative to raw — it makes explicit what raw was already doing implicitly. Per rider 2, no gauge-artifact *mechanism* claim is made here (that is ZPREACH).

