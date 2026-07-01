# smoke_vllm — replicate A1 & A2 inside vLLM's own stack

Re-measures the paper's A1 (gauge non-invariance) and A2 (structured-output collapse)
using vLLM's own `repetition_penalty`, on a single 24GB GPU. Results (numbers, raw
generations, findings) live in `../../results/smoke_vllm/` (`summary.json`, `a1_raw.json`,
`a1_qwen_raw.json`, `a2_raw.json`, `REPORT.md`).

## Environment

Python 3.11 venv, cu124 wheels (vLLM's cu13 wheels require a CUDA-13-capable driver;
on a driver-570 host the cu124 build below is the one that works):

```bash
uv venv .venv --python 3.11
VIRTUAL_ENV="$PWD/.venv" uv pip install --torch-backend=cu124 \
  "torch==2.6.0" "vllm==0.8.5.post1" "transformers==4.51.3" setuptools datasets
```

Pinned: vllm 0.8.5.post1, torch 2.6.0+cu124, transformers 4.51.3. The penalty operator
in vLLM `main` (checked at `e196268`) is mathematically identical to 0.8.5's, so the
version pin does not narrow the finding; see `../../results/smoke_vllm/REPORT.md`.

## Rerun

```bash
export HF_HUB_CACHE=/path/to/roomy/disk   # models download on first use

# A1 gauge probe — gpt2-large (primary, clean, gate passes)
./.venv/bin/python run_a1_vllm.py --model openai-community/gpt2-large

# A1 gauge probe — Qwen2.5-7B (bf16; validity gate FAILS, see REPORT — bf16 shift numerics)
./.venv/bin/python run_a1_vllm.py \
  --model Qwen/Qwen2.5-7B --dtype bfloat16 --max-model-len 2048 --gpu-mem-frac 0.90 \
  --out ../../results/smoke_vllm/a1_qwen_raw.json

# A2 JSON validity — Qwen2.5-Coder-7B, stock repetition_penalty knob
./.venv/bin/python run_a2_vllm.py
```

## Key implementation notes

- **A1 shift route:** vLLM V1 rejects per-request `logits_processors`; V0 applies them in
  `LogitsProcessor.forward` BEFORE the sampler's `apply_penalties`. So `run_a1_vllm.py`
  forces `VLLM_USE_V1=0` and injects the ±c gauge shift as a `logits_processor`, which
  therefore precedes the penalty. Validity gate: flip rate at θ=1.0 must be exactly 0.
- **A2** uses the native `SamplingParams.repetition_penalty` on the default (V1) engine —
  the stock knob, no custom processors.
- Both engines call the same `vllm/model_executor/layers/utils.py:apply_penalties`
  sign-branch.
