# ZPREACH — zero-point → raw-steepness → fix-reachability — **MECHANISM-NOT-ESTABLISHED**

Frozen test of FIXCAL's surviving *post-hoc* reading: raw's deployed-band suppression strength is a **gauge artifact** of an extreme logit zero-point, and fix-reachability is limited by **how hard RAW drives the metric**, not by how weak the fix is. Assembly only — no new decode. Rule frozen in `PREREG.md` §5 (quoted verbatim at the end).

- deployed anchor: **θ_raw = 1.1** (Ollama / GPT4All default), primary metric `rep_rate`
- steepness: `raw_grid[1.00].rep_rate − raw_grid[1.05].rep_rate`
- reachability score: reach_score = theta' if MATCHED at theta_raw=1.1 (rep_rate) | 1.0 if TRIVIAL | +inf if UNREACHABLE. LOWER = MORE reachable. Ranked by ascending midranks (rank 1 = most reachable); all UNREACHABLE models tie at +inf in one midrank block and are never separated by their shortfalls; pairs tied on either variable are excluded from C/D and counted as T.
- group split: EXTREME = frac_seen_logit_positive < 0.5 (multiply-branch dominant) vs NON-EXTREME >= 0.5

## Cohort presence (frozen n=7)

| model | (i) zero-point | (ii) raw steepness | (iii) reach @1.1 | complete |
|---|:--:|:--:|:--:|:--:|
| `gpt2` | yes | yes | yes | **yes** |
| `gpt2-large` | yes | yes | yes | **yes** |
| `pythia-2.8b` | yes | yes | yes | **yes** |
| `Qwen2.5-7B` | yes | yes | yes | **yes** |
| `Qwen2.5-7B-Instruct` | yes | yes | yes | **yes** |
| `Qwen2.5-Coder-7B` | yes | yes | yes | **yes** |
| `starcoder2-7b` | yes | yes | yes | **yes** |

**7 / 7 complete.**

## Per-model table (sorted by zero-point extremity, most extreme first)

| model | family | group | frac-pos | **frac-neg** (P1 x) | median top-1 logit | rep_rate θ=1.0 | rep_rate θ=1.05 | **steepness** | verdict @1.1 | θ′ | θ′/θ_raw | achieved vs target (shortfall) | **reach_score** |
|---|---|---|--:|--:|--:|--:|--:|--:|:--:|--:|--:|---|--:|
| `gpt2` | gpt2 | EXTREME | 0.091 | **0.909** | -97.50 | 0.9385 | 0.1392 | **0.7993** | UNREACHABLE | -- | -- | 0.1521 vs 0.0654 (short 0.0867) | **∞** |
| `gpt2-large` | gpt2 | NON-EXTREME | 0.746 | **0.254** | 11.88 | 0.8870 | 0.7861 | **0.1008** | MATCHED | 1.738 | 1.580 | achieved 0.6348 ≈ target 0.6372 | **1.738** |
| `starcoder2-7b` | starcoder | NON-EXTREME | 0.828 | **0.172** | 15.50 | 0.8638 | 0.7634 | **0.1003** | MATCHED | 2.547 | 2.315 | achieved 0.5295 ≈ target 0.5312 | **2.547** |
| `pythia-2.8b` | pythia | NON-EXTREME | 0.938 | **0.062** | 17.62 | 0.8608 | 0.6848 | **0.1760** | MATCHED | 2.512 | 2.283 | achieved 0.5088 ≈ target 0.5095 | **2.512** |
| `Qwen2.5-7B-Instruct` | qwen | NON-EXTREME | 0.944 | **0.056** | 23.62 | 0.5854 | 0.5127 | **0.0728** | MATCHED | 4.938 | 4.489 | achieved 0.4246 ≈ target 0.4312 | **4.938** |
| `Qwen2.5-7B` | qwen | NON-EXTREME | 0.961 | **0.039** | 23.19 | 0.6135 | 0.5308 | **0.0828** | MATCHED | 5.500 | 5.000 | achieved 0.4016 ≈ target 0.4021 | **5.500** |
| `Qwen2.5-Coder-7B` | qwen | NON-EXTREME | 0.986 | **0.014** | 24.19 | 0.6885 | 0.5168 | **0.1716** | MATCHED | 6.625 | 6.023 | achieved 0.3777 ≈ target 0.3745 | **6.625** |

*(`reach_score`: lower = more reachable; ∞ = UNREACHABLE at θ′≤θ′_max, ranked worst and tied with every other UNREACHABLE model. `median top-1 logit` is a reported diagnostic only — PROTOCOL §4 forbids gating on an absolute logit.)*

## P1 — zero-point extremity → raw steepness

### P1 (overall, all present models) — raw steepness increases with frac-seen-logit-NEGATIVE

- models used (n=7): `gpt2`, `gpt2-large`, `pythia-2.8b`, `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`, `Qwen2.5-Coder-7B`, `starcoder2-7b`
- ordering by `frac_neg` ascending: `Qwen2.5-Coder-7B` < `Qwen2.5-7B` < `Qwen2.5-7B-Instruct` < `pythia-2.8b` < `starcoder2-7b` < `gpt2-large` < `gpt2`
- pairs: **C=14** concordant, **D=7** discordant, T=0 tied (excluded)
- **ordering HOLDS** (gate: C > D)
- Spearman ρ = **0.429** — *DESCRIPTIVE ONLY; no gate reads it* (frozen rule 1)

## P2 — raw steepness → un-reachability at θ_raw=1.1

### P2 (overall, all present models) — the matching cost at 1.1 increases with raw steepness (= reachability decreases)

- models used (n=7): `gpt2`, `gpt2-large`, `pythia-2.8b`, `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`, `Qwen2.5-Coder-7B`, `starcoder2-7b`
- ordering by `steepness` ascending: `Qwen2.5-7B-Instruct` < `Qwen2.5-7B` < `starcoder2-7b` < `gpt2-large` < `Qwen2.5-Coder-7B` < `pythia-2.8b` < `gpt2`
- pairs: **C=12** concordant, **D=9** discordant, T=0 tied (excluded)
- **ordering HOLDS** (gate: C > D)
- Spearman ρ = **0.214** — *DESCRIPTIVE ONLY; no gate reads it* (frozen rule 1)

## P3 — does gpt2-large pattern with gpt2?

Sides are computed on **midranks**, not values (PREREG §4): `reach_score` is censored at +∞, so a value-median could itself be +∞ and the clause would pass unconditionally.

- cohort lower-median rank: steepness 4.0, reach_score 4.0 (n=7)
- `gpt2`: steepness 0.7993 (rank 7.0) → **high**; @1.1 UNREACHABLE score ∞ (rank 7.0) → **high**
- `gpt2-large`: steepness 0.1008 (rank 4.0) → **low**; @1.1 MATCHED score 1.738 (rank 1.0) → **low**
- **P3b (same side on steepness): FAILS**
- **P3c (same side on reach_score): FAILS**
- **P3 FAILS** (both clauses required — frozen rule 3)

**P3a premise check (REPORTED, NOT A GATE):** is gpt2-large's zero-point actually extreme (frac-positive < 0.5)? measured frac-positive = 0.746 → **NO**. The request's parenthetical *"(both extreme zero-points per the a1_zeropoint run)"* is **contradicted by the a1_zeropoint run's own table**, which was pre-registered as an expected failure in PREREG §4. Per PREREG §5-D1 this does **not** rescue a P3 failure: the premise was falsifiable before this analysis ran, so it cannot be discovered from these results and cannot be used to re-cut them.

## Within-group ordering (frozen rule 4 — the TRIAGE power caveat, as a gate)

Split: EXTREME = frac_seen_logit_positive < 0.5 (multiply-branch dominant) vs NON-EXTREME >= 0.5

**EXTREME** (n=1): `gpt2`  → **<3 models: not computable, does NOT satisfy the gate**
  - P1 within EXTREME: **not computable** (n<2)
  - P2 within EXTREME: **not computable** (n<2)

**NON-EXTREME** (n=6): `gpt2-large`, `pythia-2.8b`, `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`, `Qwen2.5-Coder-7B`, `starcoder2-7b`
  - P1 within NON-EXTREME: C=8 / D=7 / T=0 → **HOLDS**  ·  ρ = 0.086 (descriptive)
  - P2 within NON-EXTREME: C=6 / D=9 / T=0 → **FAILS**  ·  ρ = -0.257 (descriptive)

Largest eligible group: **NON-EXTREME**  ·  within-group check **FAILS**.

### Family diagnostic (reported, never a gate)

| family | models | P1 within-family | P2 within-family |
|---|---|---|---|
| gpt2 | `gpt2`, `gpt2-large` | C=1/D=0/T=0 → HOLDS | C=1/D=0/T=0 → HOLDS |
| pythia | `pythia-2.8b` | not computable | not computable |
| qwen | `Qwen2.5-7B`, `Qwen2.5-7B-Instruct`, `Qwen2.5-Coder-7B` | C=0/D=3/T=0 → FAILS | C=3/D=0/T=0 → HOLDS |
| starcoder | `starcoder2-7b` | not computable | not computable |

## Power judgement — `effective_n` / `family_confounded`

- `family_confounded` = **True**
- `effective_n` = **2** (cohort n_present = 7)

> **This is 2 effective points, not 7.** The overall ordering holds only because the extreme-zero-point group separates from the rest; with that between-group contrast removed, the predicted ordering does not survive inside the larger group. The result restates *"the GPT-2-family model with the extreme zero-point differs from the others"* — it is **not** 7 independent draws, and the mechanism is **NOT** established by it, whatever ρ says.

## Bottom line

**MECHANISM-NOT-ESTABLISHED** (frozen rule 7): P3 fails (gpt2-large does not pattern with gpt2); separation is family-only (`family_confounded=true`, effective_n=2).

**Failure branch, in force verbatim:** the mechanism **stays open**. Filings and the paper **keep the measured maps** (FIXCAL's calibration table, MATCHED's matched-suppression head-to-head, the a1_zeropoint run's zero-point table) and **drop the gauge-artifact explanation entirely** — removed, not softened, not hedged, not relegated to a footnote. No re-cut, no re-gate, no alternative predictor substituted after seeing this table; any such reading is post-hoc and non-binding and needs a fresh pre-registration on checkpoints outside this cohort.

---

## Frozen decision rule (verbatim, `PREREG.md` §5)

(could not read PREREG.md Sec 5 next to the analyzer: [Errno 2] No such file or directory: '/home/peter/paper-repetition-penalty/public/code/PREREG.md')

