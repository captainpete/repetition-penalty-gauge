# llama.cpp replication of A1 + A2 on the benchmark manifests (2026-07-02)

Re-measurement of the prior llama.cpp smoke replication (`results/smoke_llamacpp/`,
16 hand-written prompts / 6 hand-written schemas) on the **exact benchmark manifests
used by the main experiments**: A1 on the 200 WikiText-103 prefixes
(`results/bench_a1/prefixes.json`), A2 on the 200 JSONSchemaBench schemas
(`results/bench_a2/schemas.json`). Same llama.cpp build, same drivers (extended with
file-input / chunked-resume flags), same operator (`llama_sampler_init_penalties`).

- **llama.cpp commit:** `4fc4ec5541b243957ae5099edb67372f8f3b550e` (unchanged from the
  prior round; same CUDA build, same GGUFs: gpt2-large f16, Qwen2.5-Coder-7B bf16).
- **Drivers:** `code/smoke_llamacpp/src/a1_driver.cpp`, `src/a2_driver.cpp` — extended
  with `--prompts-file` / `--prompts-dir`, `--start/--end` (chunked, resumable),
  `--thetas`, `--last-ns`, `--threads`. Defaults preserve the original smoke behavior.
- **Inputs:** `code/smoke_llamacpp/prep_bench_inputs.py` materializes the two manifests;
  the A2 prompt is built with `code/bench_a2/run_json_bench.py`'s `build_prompt` copied
  VERBATIM (schema embedded via `json.dumps(schema, indent=2)`).
- **Scoring:** A1 `analyze_a1_flip.py` (extended to merge chunk files); A2
  `analyze_a2_json_bench.py` with `first_json` + `schema_valid` (jsonschema, draft
  auto-detected) copied VERBATIM from `code/bench_a2/run_json_bench.py`.

## Headline results

| Measurement | llama.cpp (this run) | HF-side benchmark reference | prior 16-prompt/6-schema round |
|---|---|---|---|
| A1 flip θ=1.0 (validity gate) | **0.0000 (0/40,000) — PASS** | 0/40,000 | 0/3,200 |
| A1 flip θ=1.3 | **0.9636** | 0.9639 | 0.9413 |
| A2 JSON valid θ=1.0 (whole ctx) | **0.975** | 0.970 | 1.000 |
| A2 JSON valid θ=1.1 (whole ctx) | **0.945** | 0.955 | 1.000 |
| A2 JSON valid θ=1.3 (whole ctx) | **0.290** | 0.225 | 0.167 |
| A2 JSON valid θ=1.3 (default 64-window) | **0.120** | — | 0.167 |

Both effects replicate inside llama.cpp on the benchmark inputs. The A1 flip rate at
θ=1.3 matches the HF-side benchmark number to 3 decimal places (0.9636 vs 0.9639)
across a different engine, f16 GGUF weights, and a different penalty-window mechanism.
The HF-side fix (log-softmax before penalize) reference for A2 is 0.970/0.975/0.970 —
the collapse measured here is attributable to the raw operator, not the model or task.

## A1 — gauge probe (gpt2-large f16 GGUF, CPU backend)

- 200 manifest prefixes × 200 new tokens × θ∈{1.0,1.3} × c∈{+5,−5} = 40,000 aligned
  greedy positions per θ. Chain: `logit_bias(all 50,257 vocab tokens, c)` →
  `penalties(1024, θ)` → `greedy`; prompt tokens accepted into the penalty window;
  generation continues past EOG for fixed-length alignment. CPU backend
  (`CUDA_VISIBLE_DEVICES=""`) for bit-exact determinism; 20 resumable chunks
  (`run_a1_bench.sh`).
- **Validity gate:** flip(θ=1.0) = 0/40,000 exactly. A uniform logit shift is a
  provable softmax/argmax no-op while the penalty is off.
- **θ=1.3:** flip rate **0.9636** (per-prompt mean 0.9635). HF-side benchmark: 0.9639
  (38,555/40,000). Prior hand-written-prompt round: 0.9413.
- Raw: `a1_chunks/a1_raw_*.json` (20 files, 800 generations), `a1_summary.json`,
  log `a1_run.log`.

## A2 — JSON validity (Qwen2.5-Coder-7B bf16 GGUF, CUDA)

- 200 manifest schemas × θ∈{1.0,1.1,1.3} × penalty_last_n∈{2048, 64} = 1,200 greedy
  generations, `max_new=512`, stops `"\n\n"` / triple-backtick / EOG — the bench_a2
  protocol. Chain: `penalties(last_n, θ)` → `greedy`; prompt tokens accepted into the
  window. 8 resumable chunks (`run_a2_bench.sh`), each resumable.
- `penalty_last_n=2048` ≥ prompt (~500–700 tokens) + 512 generated → whole-context ≈
  HF all-seen semantics (the prior round's 1024 would NOT cover these longer prompts).
  `64` is llama.cpp's default (`common/common.h: penalty_last_n = 64`).

| penalty_last_n | θ=1.0 | θ=1.1 | θ=1.3 |
|---|---|---|---|
| 2048 (whole context ≈ HF) | 0.975 (195/200) | 0.945 (189/200) | **0.290 (58/200)** |
| 64 (llama.cpp default) | 0.975 (195/200) | 0.930 (186/200) | **0.120 (24/200)** |
| HF-side benchmark (raw) | 0.970 | 0.955 | 0.225 |
| HF-side benchmark (fix) | 0.970 | 0.975 | 0.970 |

- Dominant θ=1.3 failure mode is **structural corruption**: `no_json` (no parseable
  brace-matched object) accounts for 97/142 whole-context failures and 155/176
  default-64 failures; the rest are schema violations. Example (schema 0):
  θ=1.0 → `{"transparency": 30}` (valid); θ=1.3 → `{"ground":{"transparency":5}}`
  (invalid: `additionalProperties: false`).
- **Unlike the prior round, the two window conditions now diverge — and the default is
  WORSE.** The prior 6 schemas had ~30-token prompts and byte-identical windows
  (0.167 = 0.167). With benchmark prompts (~600 tokens) and longer outputs, the
  64-token window at θ=1.3 gives 0.120 vs 0.290 for whole-context: the sliding window
  keeps penalizing exactly the most recent structural tokens (`"`, `:`, `}`, digits)
  while they are still needed, and never dilutes across the prompt. The default
  setting real users hit is not protective — it is the worst cell in the table.

## Gates / caveats / deviations

1. **A1 gate:** exact PASS (0/40,000).
2. **A2 GPU offload:** all 1,200 generations ran with `-ngl 26` (26/29 layers + output
   on GPU, 12.8 GiB) instead of full offload, to coexist with another resident GPU
   process on this shared machine (an ollama server intermittently holding
   7.6–10.4 GiB held by an unrelated process). The 3 CPU-resident layers compute in f32 —
   this may flip occasional borderline greedy tokens vs full offload but does not
   change the operator; the configuration was identical across all cells.
3. **Whole-context window is 2048 here** (prior round used 1024) because bench
   prompts + 512 new tokens exceed 1024. Semantics identical (all-seen).
4. **θ=1.0 is not 1.000 on the benchmark** (0.975 here, 0.970 HF-side): real
   JSONSchemaBench schemas include regex/format/required constraints the model fails
   even unpenalized — unlike the 6 toy schemas of the smoke round. The relevant
   quantity is the drop to 0.290/0.120 at θ=1.3 while the HF-side fix stays ~0.97.
5. A2 reps=1 per cell (200 distinct schemas; greedy decoding, determinism verified in
   the prior round at 8 reps).

## Files

- `a1_chunks/*.json`, `a1_summary.json`, `a1_run.log` — A1 raw token ids + summary.
- `a2_chunks/*.json` (+ per-chunk `.log`), `a2_summary.json`, `a2_run.log` — A2 raw
  generations (1,200) + summary with failure-reason breakdown.
- `summary.json` — machine-readable cross-stack summary
  (`code/smoke_llamacpp/summarize_bench.py`).
