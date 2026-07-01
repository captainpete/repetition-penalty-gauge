# Smoke replication of A1 & A2 inside vLLM's own stack — findings

**Bottom line:** Both measurements reproduce inside vLLM using its own `repetition_penalty`, and the operator in vLLM `main` is byte-identical to 0.8.5. A1 passed its validity gate cleanly on gpt2-large and matched HF; the bf16 7B A1 hit a real numerical gate failure (diagnosed). A2 reproduced the structured-output collapse.

## Environment (exact)
- vllm **0.8.5.post1**, torch 2.6.0+cu124, transformers 4.51.3, python 3.11.14. RTX 3090 (24GB), driver 575.57.08.
- Code: `code/smoke_vllm/` (`run_a1_vllm.py`, `run_a2_vllm.py`, `README.md`, `.venv` gitignored). Results: `results/smoke_vllm/` (`summary.json`, `a1_raw.json`, `a1_qwen_raw.json`, `a2_raw.json`).
- Models resolve through `HF_HUB_CACHE`; point it at a roomy disk before rerunning.

## Deliverable 1 — source verification

**(a) Operator is the sign-branched multiplicative form on RAW logits.** Installed `.venv/.../vllm/model_executor/layers/utils.py`, `apply_penalties`, lines 56–61:

```python
penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
# If logits are positive, divide by penalty, otherwise multiply by penalty.
scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
logits *= scaling
```

Sign branch on the raw logit, no normalization. **Seen-set = `prompt_mask | output_mask`** → penalizes tokens in prompt OR output (same set as HF's `unique(input_ids)`).

**(b) Ordering (verified by reading, not assumed):** In **V0**, `LogitsProcessor.forward` (`model_executor/layers/logits_processor.py:83`) applies per-request `logits_processors` BEFORE the `Sampler` calls `apply_penalties` (`sampler.py:262`) → **logits_processors run before penalties.** In **V1** (the 0.8.5 default), per-request `logits_processors` are outright rejected: `v1/engine/processor.py` raises `ValueError("vLLM V1 does not support per request user provided logits processors.")`. Both engines route through the *same* `utils.apply_penalties` (V1 via `v1/sample/ops/penalties.apply_all_penalties`).

**(c) Upstream `main` unchanged.** main HEAD `e196268` (fetched 2026-07-01). Operator refactored into `vllm/_custom_ops.py:apply_repetition_penalties_torch` but math is identical: `scaling = torch.where(logits > 0, 1.0/penalties, penalties); logits *= scaling`, same `prompt_mask | output_mask`. **The operator in vLLM main is unchanged since 0.8.5.**

## Deliverable 2 — A1 gauge probe

Route used: **V0 engine (`VLLM_USE_V1=0`) + per-request `logits_processors=[lambda past, logits: logits + c]`**, which per 1(b) runs before the penalty. Greedy, max_tokens=200, ignore_eos, 16 prompts × 200 = 3200 aligned positions, c=+5 vs c=−5.

**gpt2-large (dtype float16) — CLEAN, gate passed:**

| θ | vLLM flip rate | HF ref |
|---|---|---|
| 1.0 | **0.0000** (0/3200) ✓ gate | 0.0 |
| 1.15 | 0.9266 | — |
| 1.3 | **0.9387** | 0.941 |

Gate exactly 0; θ=1.3 matches HF's 0.941 within 0.2 pp. **A1 replicates in vLLM's own sampler.**

**Qwen2.5-7B (dtype bfloat16) — GATE FAILED, numbers withheld:** θ=1.0 flip rate = **0.3269** (should be 0), so θ>1 (0.935 / 0.881) are contaminated and not reportable. **Root cause (diagnosed):** vLLM's V0 `logits_processor` receives logits in the model compute dtype = bf16 (7 mantissa bits); adding +5 vs −5 is not order-preserving under bf16 rounding, so rare per-step argmax flips occur even at θ=1.0 and cascade autoregressively over 200 tokens. Evidence: (1) gpt2-large in fp16 gives exactly 0; (2) direct check — argmax-flip fraction under a ±5 shift is 0.005 in bf16 vs 0.000 in fp16/fp32; (3) HF's `run_a1.py` casts logits to fp32 before the shift, avoiding it. A clean bf16 A1 would need forcing fp32 logits (a `compute_logits` monkeypatch), out of scope for the stock-processor route.

## Deliverable 3 — A2 JSON validity (stock knob, V1 default engine)

Native `SamplingParams(repetition_penalty=θ)`, Qwen2.5-Coder-7B bf16, greedy, max_tokens=160, stop `["\n\n","```"]`; `JSON_TASKS`/prompts/`first_json`/`json_valid` copied verbatim (6 schemas × 8 = 48 prompts).

| θ | vLLM valid rate | HF ref |
|---|---|---|
| 1.0 | **1.000** (48/48) | 1.0 |
| 1.1 | **1.000** (48/48) | 1.0 |
| 1.3 | **0.167** (8/48) | 0.0 |

**Reproduced:** stock `repetition_penalty` collapses schema-valid JSON from 100% to 16.7% at θ=1.3. **Deviation:** vLLM leaves 8/48 valid vs HF's exact 0 — severe but not total on this stack (plausibly minor differences in seen-set/greedy tie-break/penalty ordering vs HF's per-step loop). Direction and threshold behavior match.

## Surprises / deviations
- vLLM V1 (the default) flatly rejects custom `logits_processors` — forced the A1 route to V0; A2 stayed on V1 (stock knob).
- The bf16 gate failure on the 7B A1 is a genuine finding: on a bf16 model, vLLM's pre-penalty logit path is not even numerically gauge-stable at θ=1 (before any penalty), because the shift isn't applied in fp32.
- A2's residual 8/48 at θ=1.3 (not exact 0) is the only quantitative divergence from the HF reference.

Machine-readable numbers are in `summary.json`.
