# MATCHED — MATCHED: matched-SUPPRESSION head-to-head (quality at equal anti-repetition strength)

Frozen before running. Depends on **FIXCAL** (`paper-repetition-penalty/results/fix_calibration/`),
which supplies the θ_raw → θ′_fix map. House style: `experiments/PROTOCOL.md` §3/§4/§4a/§4b.

## 1. The question, and why matched-θ was the wrong comparison

Every existing raw-vs-fix comparison (the a1b_fix, a2_downstream, and fix_loopcheck runs) is at **matched θ**, which is confounded: the normalized
operator does less work per unit θ (the fix_loopcheck run), so at the same dial setting it trivially looks gentler. The
claim an engine maintainer will actually demand is **dominance at equal suppression**: same loop-breaking,
less collateral damage. FIXCAL built the map that makes that comparison possible. MATCHED runs it.

Per the FIXCAL frozen rider **rider (1)** — used as framing everywhere in this experiment, not as a gate:
the 2.5×–6.6× per-model spread of θ′ at the same θ_raw *is A1 measured in suppression space*. Raw θ=1.1 was
never one intervention to begin with; "calibrate per model" is not a defect of the fix relative to raw, it
makes explicit what raw was already doing implicitly.

Per rider (2): **the gauge-artifact mechanism claim stays out of this experiment entirely.** MATCHED measures
quality at matched suppression. It asserts nothing about *why* reachability varies (that is ZPREACH).

## 2. Decomposition 

- **CORE** — at *equal measured repetition suppression*, is the normalized (fix) operator's output quality
  ≥ the raw operator's? Load-bearing. Adjudicated per model, per matched pair, per metric, always as a
  **paired contrast with a bootstrap CI**, never as an absolute level.
- **GAUGE GATE (not an embellishment — a hard gate)** — at every matched pair, the A1 c=±5 flip-rate must be
  **exactly 0** under the fix and **> 0** under raw. Gauge-invariance is the fix's entire reason to exist;
  losing it at any pair is a hard REFUTED regardless of quality.
- **PREMISE / VALIDITY** — the pair must actually *be* matched at run time (re-measured rep_rate agreeing
  within the FIXCAL tolerance). An unmatched pair is a calibration failure, never evidence about the fix.
- **EMBELLISHMENTS (reported, never conjoined into the verdict)** — distinct-1/distinct-2 diversity contrasts;
  the raw-vs-fix gap as a function of suppression level; the θ=1.0 unpenalized reference for every metric;
  the determinism cross-check between the `pairs` and `open` stages.

**Saturability audit .** `distinct2` saturates at ~1.0 on gpt2 (FIXCAL report: "uninformative
there") and `json_valid` has a hard 0/1 floor/ceiling. Both are reported with an explicit `saturated` flag
when a condition's mean is within 0.02 of 0 or 1, and a saturated metric cannot carry a DOMINANT verdict on
its own. `rep_rate` is the **matching check only** and is never scored as quality.

## 3. The frozen pairs (allowlist — no substitution, ever)

FIXCAL froze the MATCHED allowlist (primary metric `rep_rate`, verdict `MATCHED`):

| θ_raw | gpt2 | pythia-2.8b | Qwen2.5-7B | Qwen2.5-Coder-7B |
|--:|---|---|---|---|
| 1.02 | 2.86 | 1.21 | 1.28 | 1.28 |
| 1.05 | — (UNREACHABLE) | 1.70 | 1.84 | 3.25 |
| **1.10 (deployed anchor: Ollama / GPT4All)** | — (UNREACHABLE) | **2.51** | **5.50** | **6.63** |
| 1.15 | — (UNREACHABLE) | 3.81 | — (UNREACHABLE) | — (UNREACHABLE) |

θ′ values are **read at run time** from `fix_calibration_summary.json` (a byte-identical vendored copy of
`paper-repetition-penalty/results/fix_calibration/summary.json`, sha256
`930039cc6b90c4714d80fe85bd8805326eb26820a317e691c203482fb6fba500`; the paper repo is not mounted in the lab
container, so the copy travels with the experiment and its sha256 is recorded in every raw). The runner
**asserts** each read θ′ against the table above (tol 0.006 — the table is 2-dp rounded) and against the
`MATCHED`/`rep_rate` verdict, and **aborts** on any mismatch. No θ′ is hardcoded as an input.

**Levels used (2–3 per model, per the request; the deployed anchor θ_raw=1.10 included wherever available):**
- gpt2 → **{1.02} only.** This is the whole allowlist: gpt2 has **no** 1.10 pair (UNREACHABLE at θ′≤10;
  closed-form required θ′ ≈ 12 300). *Documented deviation from "anchor at 1.10": impossible on gpt2 by
  measurement, not by choice.* gpt2 therefore carries one level and cannot speak to the deployed anchor.
- pythia-2.8b → **{1.02, 1.05, 1.10}** (its 1.15 pair is in the allowlist but is excluded by the 2–3-level
  cap; it is reachable via `--levels 1.05,1.10,1.15` and is the only way to probe above the deployed band).
- Qwen2.5-7B → **{1.02, 1.05, 1.10}** (the full allowlist).
- Qwen2.5-Coder-7B → **{1.02, 1.05, 1.10}** (the full allowlist).

Per MATCHED's frozen pre-registration (1): **no nearby pair is ever substituted for an UNREACHABLE anchor.**
`--levels` is checked against the allowlist and aborts on anything outside it.

## 4. The three arms

**(a) Structured output — Qwen2.5-Coder-7B.**
1. **JSON-schema validity (LEADS the structured case — it is the clean metric).** a2_downstream harness reused
   unmodified: `experiments/A2/run_a2_downstream.py`'s `JSON_TASKS` (6 schemas), `RepPen` LogitsProcessor
   (raw/fix during generation), `gen_batch`, `first_json`, `json_valid`. Unit of replication = one schema
   instance (6 schemas × `--json-reps` prompt variants).
2. **Canonical HumanEval pass@1 (canonical-harness upgrade — load-bearing, not polish).** Official protocol: canonical
   prompt verbatim, the standard stop sequences `["\nclass", "\ndef", "\n#", "\nif", "\nprint"]`, standard
   completion extraction (truncate the continuation at the earliest stop sequence), greedy, pass@1 over the
   164 problems. Problem source, in order, recorded in the raw as `humaneval_source`:
   `human_eval` package → `datasets` `openai/openai_humaneval` (**cached in the lab; this is what will be
   used**) → a vendored JSONL via `--humaneval-jsonl`. Unit of replication = one problem.

**(b) Open-ended text — gpt2, pythia-2.8b, Qwen2.5-7B**, the 16 A1 prompts (`run_a1.PROMPTS`), greedy,
`max_new=256`, no EOS stopping — **identical to the FIXCAL protocol the pairs were calibrated under**, so the
matching is evaluated on exactly the distribution it was fitted on. Metrics per prompt: distinct-1,
distinct-2, longest-run, **rep_rate as the matching check**, plus a fluency/quality measure.

**Quality measure — frozen preference order, whichever is used is recorded in the raw as `quality_measure`:**
1. **MAUVE** vs the θ=1.0 (unpenalized) reference generations, if `mauve` (the `mauve-text` package) imports.
   MAUVE is a *set-level* scalar, so the prompt-level CI is obtained by featurizing once and bootstrapping the
   prompt indices through `compute_mauve(p_features=…, q_features=…)`.
2. else **judge-scored coherence** with the cached `Qwen/Qwen2.5-7B-Instruct` (`experiments/judge_lib.py`),
   fixed 1–5 rubric, **blind** (the judge never sees the condition) and in **randomized order** (one shuffled
   pool of all texts from both conditions, fixed seed).
3. else **mean NLL** of the generation under a fixed held-out reference model (`--nll-ref`, default
   `gpt2-large`); reported as **negative** NLL so that higher = better, like the other quality metrics.

**Availability, recorded now, before running:** `mauve` is **NOT installed** in the lab venv
(`Qwen/Qwen2.5-7B-Instruct` **is** cached in the run environment) → **the run will use the judge** unless the
operator installs `mauve-text` first. The MAUVE path is implemented but is **untested** (it cannot be
smoke-tested without the package); it is therefore wrapped so that *any* exception in it falls back to the
judge and records `quality_fallback_reason` in the raw, so an untested path cannot sink an unsupervised run.

**(c) Gauge no-regression gate — the one thing the fix cannot lose.** At **every** matched pair, the A1
flip-rate test with c=±5, under raw and under fix. Implemented by importing `code/run_a1.py` and
calling its **unmodified** `generate()` with `run_a1.FIX` toggled — the same vetted code path A1/A1b used.
Flip rate = position-wise argmax disagreement between the c=+5 and c=−5 greedy runs, per prompt then pooled
(exactly `A1b/analyze_a1b.py`'s endpoint). Expect flip(fix)=0 exactly and flip(raw)≫0.

## 5. Mandatory controls (PROTOCOL §3/§5)

- **No-op / identity control, must return the null exactly.** (i) `pairs`/`open`: raw θ=1.0 ≡ fix θ′=1.0
  per prompt, exactly (the FIXCAL control). (ii) `gauge`: flip-rate at θ=1.0 must be **0 exactly** under both
  raw and fix and both c — a provable gauge no-op. A failure means the *harness* is wrong; debug, do not
  interpret → **INVALID**.
- **Determinism cross-check.** `pairs` and `open` regenerate the same greedy text independently; the analyzer
  reports any per-prompt rep_rate disagreement between the two stages as a harness artifact.
- **θ=1.0 unpenalized reference** is generated/evaluated in every arm, so every contrast can be read against
  the do-nothing baseline.
- **FIXCAL agreement.** The re-measured raw rep_rate at each anchor is compared to the value FIXCAL recorded;
  a drift larger than the matching tolerance is reported (dtype/library drift), and it is the *run-time*
  measurement that gates.

## 6. FROZEN DECISION RULE (verbatim — contrasts/CIs only, PROTOCOL §4/§4a)

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
> surviving pair — downstream proposals pause on that basis, and the losing metric/pair is named.
> Report per pair and per metric; never conjoin metrics into one number. The deployed anchor (θ_raw=1.10) is
> reported separately and called out, since it carries the recommendation decision.

### 6a. Frozen operationalisation of the rule (fixed now, before any data)

- **Contrast** = mean over units of (fix − raw) **oriented so that positive = the fix is better**, with a
  **paired** bootstrap over the unit of replication (resample unit indices once per replicate and apply the
  same indices to both arms). B = 10 000, seed 0, percentile 95% CI.
- **Metric directions (frozen).** Higher-is-better: `json_valid`, `humaneval_pass1`, `quality`
  (MAUVE / judge coherence / **negative** NLL), `distinct1`, `distinct2`. (distinct-n: higher = more diverse
  at *equal* repetition suppression; this is the same orientation FIXCAL used, `SIGN[distinct2] = +1`.)
  Lower-is-better and **NOT** a quality metric: `rep_rate` (matching check), `longest_run` (reported only).
- **Metric classes.** *Structured* = {`json_valid`, `humaneval_pass1`} (Coder-7B only).
  *Open-ended* = {`quality`, `distinct1`, `distinct2`} (gpt2, pythia-2.8b, Qwen2.5-7B; also reported for
  Coder-7B if the open stage is run on it).
- **Consequence, stated up front so it is not read as a surprise:** for the three models with **no structured
  arm** (gpt2, pythia-2.8b, Qwen2.5-7B), DOMINANT is *unreachable by construction* — their attainable
  outcomes are TIES or LOSES. DOMINANT is defined only for Qwen2.5-Coder-7B. This follows the rule verbatim
  and is not a late re-gate.
- **Evaluation order (mutually exclusive):** INVALID (control failure / no surviving pair) → REFUTED (gauge
  gate) → LOSES → DOMINANT → TIES. A structured metric whose CI excludes 0 *in raw's favour* is LOSES, not
  a failed DOMINANT.
- **INCOMPLETE** is a fourth, non-scientific outcome: a stage is missing from the raws. A missing gauge stage
  never silently passes the gate; a missing `pairs` stage never silently grants the premise.
- **Gauge-gate sub-diagnosis (reported alongside the verdict; it does NOT change it).** The frozen gate
  conjoins two different things: *flip(fix) = 0* (the real property under test) and *flip(raw) > 0* (a
  **positive control** that the c=±5 test has power at that anchor). Every FAIL is therefore labelled
  `FIX_NOT_INVARIANT` (the real refutation) or `RAW_CONTROL_FLAT` (flip(raw)=0 — the test had no power there,
  which is *not* evidence against the fix). Recorded now because the lowest anchors are the ones at risk:
  A1b measured flip(raw)@θ=1.02 = 0.648 (gpt2) / 0.609 (pythia-2.8b) at 16 prompts × 200 tokens, so the
  control has ample power at the real settings and a `RAW_CONTROL_FLAT` reading would indicate a
  too-short/too-narrow run, not a result. The verdict still follows the frozen rule verbatim.
- **Residual category, frozen now (bookkeeping, not a re-gate).** The three-way rule has one gap: an outcome
  that is neither LOSES nor DOMINANT nor TIES, i.e. some CI excludes 0 **in the fix's favour** while the
  DOMINANT conditions are unmet — which happens by construction on the three models with no structured arm.
  That outcome is reported as **FAVOURS_FIX**, with the metric and pair named. Written down before any data
  exists so it cannot be mistaken for a post-hoc escape hatch.
- **Explicitly NOT gates:** any absolute pass@1 / JSON-validity / MAUVE / coherence level; any fixed θ′;
  monotonicity of the gap in suppression level; agreement between metrics. All reported descriptively.

## 7. STAGED PLAN (cheap kill first; each stage writes its own raw so a later failure never loses earlier work)

- **Stage 0 — `pairs` + `gauge`.** Pair verification (rep_rate match, no-op control) and the gauge gate at
  every pair, on all four models. Cheap. **If the gauge gate fails → REFUTED, stop** (no quality metric can
  rescue a fix that is not gauge-invariant).
- **Stage 1 — `open`** (arm b), cheap models first: gpt2 → pythia-2.8b → Qwen2.5-7B.
- **Stage 2 — `json`** (arm a, part 1) on Qwen2.5-Coder-7B — the clean structured metric, leads the case.
- **Stage 3 — `humaneval`** (arm a, part 2) on Qwen2.5-Coder-7B: **generation** first (writes completions to
  disk, no execution), then the separate opt-in **`--exec` scoring** step.

## 8. ⚠ CODE EXECUTION SAFETY (mandatory, non-negotiable)

HumanEval scoring executes model-generated code. Frozen implementation:

- **Opt-in and separated.** `--exec` defaults **OFF**. `--stage humaneval` without `--exec` only *generates*
  and writes completions to disk. Scoring is a distinct invocation, normally
  `--stage humaneval --exec --score-from <completions.json>`, which **does not load a model at all**.
- **Per-sample subprocess.** Every sample runs in its own child process, never in-process, launched with
  `[sys.executable, "-I", "-S", "prog.py"]` (`-I` isolates from `PYTHON*` env and user site; `-S` drops
  site-packages so only the stdlib is importable — the canonical HumanEval tests need nothing else).
- **Fresh throwaway cwd.** `cwd` = a fresh `tempfile.mkdtemp()` per sample, **never** anything under `/work`.
  The tempdir is `shutil.rmtree`'d in a `finally` block.
- **Hard timeout ~10 s** (`--exec-timeout`, default 10.0). Implemented as
  `Popen(..., start_new_session=True)` + `wait(timeout=)` — the exact mechanism `subprocess.run(timeout=)`
  uses internally — **extended** to `os.killpg(SIGKILL)` the child's whole process group on expiry, so a
  sample that forks cannot outlive its timeout. *(Documented strengthening of the "`subprocess.run(timeout=)`"
  wording: `run()` kills only the direct child.)*
- **`resource` limits** applied in `preexec_fn` (child only): `RLIMIT_AS` 2 GB, `RLIMIT_NPROC` 64,
  `RLIMIT_FSIZE` 8 MB, `RLIMIT_CORE` 0, plus `os.setsid()`.
- **Env scrubbed.** The child gets only `{PATH=/usr/bin:/bin, HOME=<tempdir>, TMPDIR=<tempdir>, LANG,
  PYTHONDONTWRITEBYTECODE, PYTHONHASHSEED}` — no proxy vars, no `HF_*`, no `PYTHONPATH`, no `LD_*`.
- **No stdin, capped output.** `stdin=DEVNULL`; stdout/stderr go to files *inside* the tempdir (so they are
  bounded by `RLIMIT_FSIZE`, unlike pipes) and only a truncated head is retained.
- **Residual risk (stated, not waved away).** The sandbox is a *hardened subprocess*, not a container or
  seccomp jail. A sample still runs as the `lab` user with the network reachable and read access to the
  filesystem, and could within its 10 s / 2 GB / 64-process budget make outbound connections or read
  world-readable files; `RLIMIT_NPROC` 64 bounds but does not forbid forking, and `RLIMIT_FSIZE` 8 MB bounds
  but does not forbid writes outside the tempdir to paths the `lab` user can write. Accepted because the
  inputs are greedy completions of the 164 canonical HumanEval prompts from a code model (not adversarial
  input), and because the alternative (in-process `exec`) is strictly worse. If the threat model ever
  changes, run the scoring step inside a disposable container with no network.

## 9. Real-run commands (in order; run as user `lab` in the container, `/work` = repo root)

Models are cached under `/hf/hub`; `--dtype auto` reproduces the dtype FIXCAL calibrated each pair under
(fp32 for gpt2/pythia, bf16 for the Qwens). Nothing below trains anything.

```bash
# from the repository root
# python = your transformers-capable interpreter
R=results/matched_strength

# ---- Stage 0 (cheap kill): pair verification + gauge gate, all four models -------------------------
for M in gpt2 EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B Qwen/Qwen2.5-Coder-7B; do
  python code/run_matched.py --model $M --stage pairs
  python code/run_matched.py --model $M --stage gauge
done
$P analyze_matched.py --raws "$R/raw_*.json" --out-dir $R     # STOP HERE if the gauge gate FAILs

# ---- Stage 1: arm (b) open-ended, cheap models first ------------------------------------------------
python code/run_matched.py --model gpt2                 --stage open
python code/run_matched.py --model EleutherAI/pythia-2.8b --stage open
python code/run_matched.py --model Qwen/Qwen2.5-7B      --stage open

# ---- Stage 2: arm (a) part 1 -- JSON-schema validity (leads the structured case) --------------------
python code/run_matched.py --model Qwen/Qwen2.5-Coder-7B --stage json

# ---- Stage 3: arm (a) part 2 -- canonical HumanEval: GENERATE, then (opt-in) SCORE ------------------
python code/run_matched.py --model Qwen/Qwen2.5-Coder-7B --stage humaneval
$P run_matched.py --stage humaneval --exec --score-from $R/raw_Qwen2.5-Coder-7B_humaneval.json   # no GPU, no model

# ---- analyze (handles partial stages) ---------------------------------------------------------------
$P analyze_matched.py --raws "$R/raw_*.json" --out-dir $R
```

Notes for the operator:
- The quality measure is resolved automatically (MAUVE → judge → NLL). To get MAUVE instead of the judge,
  `pip install mauve-text` into the run environment **before** Stage 1; otherwise the judge is used and this
  is recorded in the raw. Do not mix measures across models — pick one before Stage 1 starts.
- `--stage open` offloads the generator to CPU before loading the 7B judge (a 7B judge and a 7B generator do
  not co-reside on 24 GB). `--keep-gen-model-on-gpu` disables that.
- pythia-2.8b's fourth allowlisted pair is available with
  `--levels 1.05,1.10,1.15` (it is the only probe above the deployed band; outside the frozen 2–3-level
  default, so run it as a clearly-labelled extra, not as a replacement).
- **Never** pass `--exec` on a machine you care about without reading §8 first.

## 10. Outputs

`runs/MATCHED/raw_<model>_<stage>.json` per model × stage (`pairs`, `gauge`, `open`, `json`, `humaneval`,
plus `humaneval_scored` from the `--exec` step) → `analyze_matched.py --raws '<glob>' --out-dir` →
`REPORT.md` + `summary.json`, which report which stages are present and apply §6 verbatim.

Inference-only throughout (no training, no optimizer, `torch.no_grad`).
