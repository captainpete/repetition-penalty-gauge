#!/usr/bin/env bash
# A2 JSON validity on the 200 JSONSchemaBench schemas (bench_inputs/a2_prompts, built by
# prep_bench_inputs.py with code/bench_a2's exact prompt), inside llama.cpp's own
# repeat_penalty sampler. Qwen2.5-Coder-7B bf16 GGUF, full GPU offload.
# theta in {1.0,1.1,1.3} x penalty_last_n in {2048 (whole context), 64 (default)}.
# Chunked + resumable: 8 chunks x 25 schemas; a chunk is skipped if its output parses
# and holds all 25*3*2 = 150 records. Analyze with
#   python3 analyze_a2_json_bench.py "<outdir>/a2_chunks/*.json" <outdir>/a2_summary.json
set -euo pipefail
cd "$(dirname "$0")"

OUTDIR="${1:-results/smoke_llamacpp_bench}"
NGL="${2:-99}"          # reduce (e.g. 26) to fit beside other residents on the shared GPU
MINFREE="${3:-18000}"   # min free VRAM gate, MiB (informational)
mkdir -p "$OUTDIR/a2_chunks"

for i in $(seq 0 7); do
  s=$((i * 25)); e=$((s + 25))
  out="$OUTDIR/a2_chunks/a2_raw_$(printf %03d $s).json"
  if [ -f "$out" ] && python3 -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if len(d['records'])==150 else 1)" 2>/dev/null; then
    echo "chunk $s-$e already complete, skipping"
    continue
  fi
  echo "=== chunk $s-$e -> $out ==="
  ./a2_driver -m models/qwen2.5-coder-7b-bf16.gguf \
    --max-new 512 --reps 1 -ngl "$NGL" \
    --prompts-dir bench_inputs/a2_prompts --start "$s" --end "$e" \
    --thetas 1.0,1.1,1.3 --last-ns 2048,64 --out "$out" 2> "$out.log"
done
echo "A2 bench (llama.cpp) all chunks done"
