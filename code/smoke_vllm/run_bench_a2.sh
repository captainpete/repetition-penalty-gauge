#!/usr/bin/env bash
# A2 JSON validity on the 200 JSONSchemaBench schemas, inside vLLM with the STOCK
# repetition_penalty knob (V1 engine, Qwen2.5-Coder-7B bf16). run_a2_vllm_bench.py
# checkpoints per theta, so re-running this script resumes. GPU jobs run sequentially.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-results/smoke_vllm_bench/a2_raw.json}"
mkdir -p "$(dirname "$OUT")"

if [ -f "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
sys.exit(0 if len(d['summary'])==3 and d['n_schemas']==200 else 1)
" 2>/dev/null; then
  echo "A2 vLLM bench output already complete, skipping"
  exit 0
fi

exec ./.venv/bin/python run_a2_vllm_bench.py --out "$OUT"
