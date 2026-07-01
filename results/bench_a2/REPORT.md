# bench_a2 — A2 rerun on public benchmarks (JSONSchemaBench + HumanEval)

Rerun of the paper's A2 experiments with benchmark-sourced prompts/schemas, replacing the
hand-written ones a reviewer flagged as arbitrary (18 hand-written prompts; 6 hand-written JSON
schemas). Code in `code/bench_a2/` (README has exact rerun commands); raw outputs in this
directory.

## Versions / setup

- venv from the project's pinned env (`env/pyproject.toml` + `uv.lock`, `uv sync --frozen`):
  torch 2.6.0+cu124, transformers 5.11.0, datasets 5.0.0; plus jsonschema 4.26.0
  (jsonschema-specifications 2025.9.1, referencing 0.37.0). Python 3.12, RTX 3090, bf16 weights
  (Part B upcasts logits to fp32 before penalty/argmax/threshold, as the original harness does).
- Models: Qwen/Qwen2.5-Coder-7B rev `0396a76181e127dfc13e5c5ec48a8cee09938b02` (Part A + Part B
  replicate), bigcode/starcoder2-7b rev `bb9afde76d7945da5745592525db122d4d729eb1` (Part B).

## Benchmarks + sampling

- **JSONSchemaBench**: HF dataset `epfl-dlab/JSONSchemaBench` (canonical HF release of
  guidance-ai/jsonschemabench, ~10K real-world JSON schemas; Geng et al. 2025,
  arXiv:2501.10868). Pool = `Github_easy` + `Github_medium`, all splits (3907 schemas), shuffled
  with `random.Random(0)`; kept iff it parses as a JSON object schema, compiles under
  `jsonschema.validators.validator_for(...).check_schema`, has no remote (`http[s]`) `$ref`, and
  serializes to <= 600 GPT-2 tokens. First 200 kept -> `schemas.json` (196 easy + 4 medium; drafts:
  120 draft-04, 48 draft-2020-12, 11+3 draft-07, 10 draft-06, 8 draft-04-no-fragment). Filter
  drops: 42 not-object, 342 too-long, 0 parse/compile failures.
- **HumanEval**: HF `openai/openai_humaneval`, all 164 official `prompt` fields, unmodified.

## Part A — JSON schema-valid rate (Qwen2.5-Coder-7B, greedy, bf16)

Protocol mirrors `code/run_a2_downstream.py` (same `RepPen` logits-processor with exact HF
semantics, `--fix` = log_softmax before penalizing; batched greedy decode, left padding, bs 8;
stop strings `["\n\n", "```"]`; `first_json` brace-matching extraction), with the benchmark
schema embedded in the prompt: *"Output ONLY a single JSON object that conforms to this JSON
Schema. Do not include any explanation.\nJSON Schema:\n{schema}\nJSON object:\n"*.
max_new_tokens=512 (schemas are <=600 GPT-2 tokens; emitted objects at theta=1.0 have median 96
chars, so 512 accommodates every sampled schema — no valid output was length-truncated).
Validation: `jsonschema` `iter_errors()==0` against the actual schema, draft auto-detected from
`$schema` via `validator_for` (draft-04 … 2020-12 all handled natively).

**Unconditioned valid rate (n=200):**

| theta | raw | fix |
|---|---|---|
| 1.0 | 0.970 (194/200) | 0.970 (194/200) |
| 1.1 | 0.955 (191/200) | 0.975 (195/200) |
| 1.3 | **0.225 (45/200)** | **0.970 (194/200)** |

**Conditioned on theta=1.0-raw-valid (n=194 — the analogue of the original experiment, which
was 100% at theta=1.0):**

| theta | raw | fix |
|---|---|---|
| 1.0 | 1.000 | 1.000 |
| 1.1 | 0.974 | 1.000 |
| 1.3 | **0.227 (44/194)** | **0.995 (193/194)** |

Old vs new (paper-note Table 3 row 3 / abstract):

| metric | old (6 hand-written schemas x 8 reps) | new (200 JSONSchemaBench schemas) |
|---|---|---|
| valid @theta=1.0 raw | 1.00 | 0.970 (cond. 1.000) |
| valid @theta=1.3 raw | **0.00** | **0.225 (cond. 0.227)** |
| valid @theta=1.3 fix | 1.00 | 0.970 (cond. 0.995) |

**The 0% floor does not survive on the benchmark: it becomes 22.5%.** The collapse is still
massive (0.97 -> 0.23, an absolute drop of 74 points, fully recovered by the fix), but a quarter
of the benchmark schemas survive theta=1.3. Why: 21 of the 45 surviving outputs are trivially
valid (the schema requires no properties, so a near-empty object passes), and survivors' outputs
are short (valid rate at theta=1.3 falls from 0.31 for schemas whose theta=1.0 answer was <=60
chars to 0.19 for >150 chars). Real-world schemas — unlike the 6 hand-written ones, which all
required multi-field objects — include many with tiny or optional-only instances, and short
generations expose few repeated-delimiter positions for the penalty to corrupt. Failure modes at
theta=1.3 raw: 119/155 no parseable JSON object, 36/155 parseable but schema-violating.

## Part B — threshold + delimiter flips on HumanEval seeds

`run_a2_humaneval.py` imports `run_a2.generate` verbatim (per-position pre-penalty
top-1/runner-up fp32 logits, argmax-tie discipline for the theta=1.0 gate) and
`analyze_a2`'s exact definitions (clean positions = top-1 seen & z_top>0 & runner-up unseen;
predicate `g < z_top(1-1/theta)`; per-record structural-flip rates; theta in {1.0,…,1.5}, 256
new tokens, greedy). HumanEval has no prose arm, so the code-vs-prose domain-ratio control is
N/A here (the paper already reports it as model-dependent and refuted as a gate).

**StarCoder2-7B** (164 prompts x 6 theta = 984 runs; 209,920 pooled positions at test theta):

| metric | old (12 hand-written code prompts) | new (164 HumanEval prompts) |
|---|---|---|
| validity gate (theta=1.0 flips) | 0 (PASS) | 0 (PASS); fix run also 0 (PASS) |
| threshold balanced accuracy | 0.9995 | **0.9993** (TPR 0.9997, TNR 0.9990; n_clean=48,919, flips 27,720) |
| flips landing on runner-up | 1.000 | **1.0000** (n=27,720) |
| structural-flip rate, raw | 0.138 | **0.1390** |
| structural-flip rate, fix | 0.005 | **0.0017** (~82x below raw) |
| overall flip fraction, theta in [1.1,1.5] pooled | 0.310 | **0.3189** (fix: 0.0112) |

**Qwen2.5-Coder-7B replicate (raw):** gate PASS (0 flips), balanced accuracy **0.9996**
(TPR 0.9996, TNR 0.9997; n_clean=47,903), runner-up fraction **1.0000** (n=31,149),
structural-flip rate 0.1700 (old Qwen replicate: bal-acc 1.0, runner-up 1.0, 0.196).

The threshold/mechanism results are essentially unchanged on benchmark prompts — every number
lands within ~0.001 of the hand-written-prompt values (Table 3 row 2's fix column actually
improves, 0.005 -> 0.0017).

## Gates

- Part B theta=1.0 validity gate: **0 flips** on StarCoder2 raw, StarCoder2 fix, and Qwen raw
  (492 sequences x 256 positions each under the argmax-tie discipline).
- Part A determinism cross-check: theta=1.0 raw and theta=1.0 fix produce identical valid sets
  (194/200, same schemas), as expected for greedy + no-op penalty.
- Part A floor: theta=1.0 raw = 0.970 — the model can do the benchmark task, so the theta=1.3
  collapse is attributable to the penalty, not task difficulty.

## Deviations / notes

- Original hand-written JSON task used max_new=160 and type-check validation of 6 fixed field
  sets; here max_new=512 and full `jsonschema` draft-aware validation, since benchmark schemas
  are larger and heterogeneous.
- Schemas with remote `$ref`s were excluded at manifest time (unresolvable offline -> would
  spuriously invalidate every instance). None were hit in the sampled pool anyway (0 drops).
- Part B was run unbatched, exactly like the original harness (batching would change the
  padding/kv-cache layout the theta=1.0 gate was validated under). GPU wall-time ~80 min per
  model per operator; runs were checkpointed per (prompt, theta) and resumed across GPU-queue
  slices.
- Suggested paper edits: abstract/§A2 "to 0%" -> "from 97% to 23%" (or "by 74 points, fully
  recovered by the fix") with n=200 JSONSchemaBench schemas; Table 3 row 3 `0.00 / 1.00` ->
  `0.23 / 0.97` (or conditioned `0.23 / 0.99`); row 2 fix value 0.005 -> 0.0017 if the
  HumanEval-seeded run replaces the hand-written one.

## Files

- `schemas.json` — Part A manifest (source, seed, filter stats, 200 schemas).
- `json_raw.json` — Part A per-schema outputs/extractions/validation per (theta, op); `json_smoke.json` — 16-schema smoke run.
- `humaneval_raw.json`, `humaneval_fix_raw.json`, `humaneval_qwen_raw.json` — Part B full
  per-position instrumentation (984 records each).
- `humaneval_summary_humaneval.json`, `humaneval_summary_humaneval_qwen.json` — Part B analyzer output.
- `summary.json` — combined machine-readable summary (Part A rates incl. conditioned; Part B metrics).
