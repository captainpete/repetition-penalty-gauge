#!/usr/bin/env bash
# A1 gauge probe on the 200 WikiText-103 benchmark prefixes (results/bench_a1/prefixes.json),
# inside llama.cpp's own sampler. CPU-only (deterministic).
# Chunked + resumable: 20 chunks x 10 prompts, each chunk its own JSON; a chunk is
# skipped if its output already parses as complete JSON. Merge/analyze afterwards with
#   python3 analyze_a1_flip.py "<outdir>/a1_chunks/*.json" <outdir>/a1_summary.json
set -euo pipefail
cd "$(dirname "$0")"

OUTDIR="${1:-results/smoke_llamacpp_bench}"
THREADS="${2:-24}"
mkdir -p "$OUTDIR/a1_chunks"

for i in $(seq 0 19); do
  s=$((i * 10)); e=$((s + 10))
  out="$OUTDIR/a1_chunks/a1_raw_$(printf %03d $s).json"
  if [ -f "$out" ] && python3 -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if len(d['records'])==40 else 1)" 2>/dev/null; then
    echo "chunk $s-$e already complete, skipping"
    continue
  fi
  echo "=== chunk $s-$e -> $out ==="
  CUDA_VISIBLE_DEVICES="" ./a1_driver -m models/gpt2-large-f16.gguf \
    --max-new 200 --penalty-last-n 1024 --thetas 1.0,1.3 \
    --prompts-file bench_inputs/a1_prefixes.txt \
    --start "$s" --end "$e" --threads "$THREADS" --out "$out" 2>/dev/null
done
echo "A1 bench (llama.cpp, CPU) all chunks done"
