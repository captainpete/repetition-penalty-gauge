# vLLM replication of A1 + A2 on the benchmark manifests (2026-07-02)

Re-measurement of the prior vLLM smoke replication (`results/smoke_vllm/`, 16
hand-written prompts / 6 hand-written schemas) on the **exact benchmark manifests used
by the main experiments**: A1 on the 200 WikiText-103 prefixes
(`results/bench_a1/prefixes.json`), A2 on the 200 JSONSchemaBench schemas
(`results/bench_a2/schemas.json`).

- **Environment (unchanged from prior round):** vllm **0.8.5.post1**, torch
  2.6.0+cu124, transformers 4.51.3, python 3.11, RTX 3090. Same operator
  (`apply_penalties` sign branch on raw logits, seen-set = prompt ∪ output).
- **A1 route:** V0 engine (`VLLM_USE_V1=0`) + per-request `logits_processors` ±c shift,
  verified to run before `apply_penalties`. `run_a1_vllm.py` gained
  `--prompts-manifest`, `--per-run-batches`, `--shift-fp32` (old 16-prompt behavior
  unchanged without flags). Runner: `code/smoke_vllm/run_bench_a1.sh`.
- **A2 route:** stock `SamplingParams.repetition_penalty` on the default V1 engine —
  no custom processors. NEW `run_a2_vllm_bench.py` mirrors
  `code/bench_a2/run_json_bench.py` VERBATIM (`build_prompt`, `first_json`,
  `schema_valid` with jsonschema; greedy, `max_new=512`, stop `["\n\n","```"]`),
  checkpointed per θ. Runner: `run_bench_a2.sh`.

## Headline results

| Measurement | vLLM (this run) | HF-side benchmark reference | prior 16-prompt/6-schema round |
|---|---|---|---|
| A1 flip θ=1.0 (validity gate) | **0.0000 (0/40,000) — PASS** (fp32) | 0/40,000 (fp32) | 0/3,200 (fp16) |
| A1 flip θ=1.3 | **0.963875** | 0.963875 | 0.9387 |
| A2 JSON valid θ=1.0 | **0.970 (194/200)** | 0.970 | 1.000 |
| A2 JSON valid θ=1.1 | **0.950 (190/200)** | 0.955 | 1.000 |
| A2 JSON valid θ=1.3 | **0.240 (48/200)** | 0.225 | 0.167 |

## A1 — gauge probe (gpt2-large, V0, fp32)

200 prefixes × 200 tokens (`ignore_eos`) × θ∈{1.0,1.3} × c∈{+5,−5}; flip rate compares
c=+5 vs c=−5 token-by-token (40,000 aligned positions per θ).

- **Gate:** θ=1.0 flip **exactly 0/40,000**. θ=1.3 flip **0.963875 (38,555/40,000)**.
- **Token-level cross-check:** the vLLM fp32 generations are **token-identical to the
  HF-side fp32 benchmark run at every one of the 80,000 compared positions per θ**
  (400/400 (prompt, θ, c) runs identical). vLLM's sampler reproduces the HF
  measurement exactly, flips and all.
- **dtype matters — two diagnosed gate failures on the way (raw files preserved):**
  1. `a1_raw_flatbatch_gatefail.json` — fp16 model, all 800 requests in one
     `llm.generate()`: θ=1.0 flip 0.0606. At this KV budget the +c/−c copies of a
     prompt land in different scheduling waves / RECOMPUTE-preemption states, so
     batch-shape-dependent fp16 kernels break pairwise identity. Fixed with
     `--per-run-batches` (one generate() per (θ,c) run → mirrored scheduling).
  2. `a1_raw_fp16shift_gatefail.json` / `a1_raw_fp16model_fp32shift_gatefail.json` —
     fp16 model, per-run batches: θ=1.0 flip 0.0238 (7/200 prompts, each one early
     tie-flip then cascade; θ=1.3 already 0.9639). Root cause: with an fp16 model the
     V0 processor loop writes the shifted row back into the fp16 logits tensor
     (`logits[i] = row`, `model_executor/layers/logits_processor.py:169`), so
     round(x+5) vs round(x−5) inverts rare near-ties — upcasting inside the processor
     (`--shift-fp32`) cannot help because the write-back re-rounds (byte-identical
     flip counts confirmed). This extends the prior round's bf16 finding: at n=200
     prompts even fp16 is not gauge-stable at θ=1.0 on vLLM's pre-penalty logit path.
  - Final run uses `--dtype float32` (matching the HF-side benchmark's fp32), where
    the shift is exact → gate passes. The θ=1.3 flip rate is essentially unaffected
    by dtype (0.9639 in all three runs) — the gauge-dependence result is robust; only
    the θ=1.0 no-op gate needs fp32.

## A2 — JSON validity (Qwen2.5-Coder-7B bf16, V1, stock knob)

200 schemas × θ∈{1.0,1.1,1.3}, raw operator only (the stock knob maintainers ship;
the fix column comes from the HF-side run: 0.970/0.975/0.970).

| θ | vLLM valid rate | HF-side (raw) |
|---|---|---|
| 1.0 | 0.970 (194/200) | 0.970 |
| 1.1 | 0.950 (190/200) | 0.955 |
| 1.3 | **0.240 (48/200)** | 0.225 |

vLLM's own `repetition_penalty` collapses schema-valid JSON from 97% to 24% at θ=1.3
on the benchmark — within 1.5 pp of the HF measurement at every θ, and the same
failure signature (unparseable/truncated objects). θ=1.0 matches HF exactly (194/200).

## Gates / caveats / deviations

1. **A1 gate:** PASS exactly (fp32). The fp16 gate failures are themselves a
   documented finding (see above), with raw files kept for provenance.
2. **A1 model dtype fp32** deviates from the prior smoke round (fp16) but matches the
   HF-side benchmark run (fp32) — the comparison target of this report.
3. **A2 ran unchanged** on the first attempt (one earlier launch crashed at engine
   init because an unrelated ollama process held 7.8–10.4 GiB VRAM on this shared GPU;
   relaunched with a free-VRAM pre-wait; no measurement impact).
4. A2 dtype bf16, `max_model_len=4096`, `gpu_memory_utilization=0.82`, greedy.

## Files

- `a1_raw.json` — final fp32 A1 run (800 generations, token ids, per-θ summaries).
- `a1_raw_flatbatch_gatefail.json`, `a1_raw_fp16shift_gatefail.json`,
  `a1_raw_fp16model_fp32shift_gatefail.json` — preserved gate-failure diagnostics.
- `a2_raw.json` (+ `a2_raw.json.theta*.part.json` checkpoints) — 600 scored
  generations with extraction + per-schema validity and failure reasons.
- `summary.json` — machine-readable summary (`code/smoke_llamacpp/summarize_bench.py`).
- Logs: `a1_rerun*.log`, `a2_run*.log`.
