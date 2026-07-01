#!/usr/bin/env bash
# A1 gauge probe on the 200 WikiText-103 benchmark prefixes, inside vLLM (V0 engine +
# per-request logits_processor shift, fp16 gpt2-large). Idempotent: skips if the output
# is already complete. GPU jobs run sequentially (shared RTX 3090 on the original host).
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-results/smoke_vllm_bench/a1_raw.json}"
MANIFEST="results/bench_a1/prefixes.json"
mkdir -p "$(dirname "$OUT")"

if [ -f "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
s=d['summary']
sys.exit(0 if d['n_prompts']==200 and s['1.0']['positions']==40000 and '1.3' in s else 1)
" 2>/dev/null; then
  echo "A1 vLLM bench output already complete, skipping"
  exit 0
fi

# --per-run-batches: one generate() per (theta, c) run — a flat 800-request call puts a
# prompt's +c/-c copies in different scheduling waves (KV preemption at this mem frac),
# which flips fp16 argmaxes even at theta=1.0 and fails the gate (measured: 0.0606).
# --dtype float32 (matches the HF-side bench run, which is fp32): with an fp16 model the
# V0 processor loop writes the shifted row back into the fp16 logits tensor
# (logits[i] = row, model_executor/layers/logits_processor.py:169), so x+5 vs x-5 round
# differently at rare argmax ties and cascade (theta=1.0 flip 0.0238 on 7/200 prompts,
# gate fail) — --shift-fp32 alone cannot fix that. fp32 end-to-end makes the shift exact.
exec ./.venv/bin/python run_a1_vllm.py \
  --model openai-community/gpt2-large --dtype float32 --thetas 1.0,1.3 \
  --prompts-manifest "$MANIFEST" --gpu-mem-frac 0.35 --per-run-batches --shift-fp32 --out "$OUT"
