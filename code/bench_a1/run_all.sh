#!/usr/bin/env bash
# Sequential driver for the full A1 benchmark grid. Jobs run sequentially (serialize externally if your GPU is shared
# on the shared 3090), so this also queues politely behind other agents' GPU jobs.
# Idempotent: jobs whose output file already exists are skipped, so restarts are cheap.
set -uo pipefail
cd code/bench_a1
export HF_HUB_CACHE=$HF_HUB_CACHE
export HF_DATASETS_CACHE=$(pwd)/.cache/datasets
PY=.venv/bin/python
R=results/bench_a1
LOG=$R/run_all.log
echo "=== run_all start $(date -Is) ===" | tee -a "$LOG"

run() {  # name minfree outfile -- cmd...
  local name="$1" mf="$2" out="$3"; shift 4
  if [ -f "$out" ]; then echo "--- [$name] SKIP (exists) ---" | tee -a "$LOG"; return 0; fi
  echo "--- [$name] $(date -Is): $* ---" | tee -a "$LOG"
  $G -m "$mf" -n "$name" -- "$@" >>"$LOG" 2>&1
  echo "--- [$name] exit $? $(date -Is) ---" | tee -a "$LOG"
}

# Table 1 — flip rates (small fp32 first, then 7B bf16)
run flip_gpt2  6000  $R/raw_flip_gpt2.json        -- $PY run_flip.py --model gpt2                        --dtype float32  --batch-size 100
run flip_g2l   6000  $R/raw_flip_gpt2-large.json  -- $PY run_flip.py --model openai-community/gpt2-large --dtype float32  --batch-size 50
run flip_pyth  6000  $R/raw_flip_pythia-2.8b.json -- $PY run_flip.py --model EleutherAI/pythia-2.8b      --dtype float32  --batch-size 50
# Table 3 — fix leg
run fix_gpt2   6000  $R/raw_flip_gpt2_fix.json        -- $PY run_flip.py --model gpt2                        --dtype float32  --fix --batch-size 100
run fix_g2l    6000  $R/raw_flip_gpt2-large_fix.json  -- $PY run_flip.py --model openai-community/gpt2-large --dtype float32  --fix --batch-size 50
run fix_pyth   6000  $R/raw_flip_pythia-2.8b_fix.json -- $PY run_flip.py --model EleutherAI/pythia-2.8b      --dtype float32  --fix --batch-size 50
# Table 2 — zero-point (small)
run zp_gpt2    6000  $R/raw_zp_gpt2.json        -- $PY run_zeropoint.py --model gpt2                        --dtype float32 --batch-size 100
run zp_g2l     6000  $R/raw_zp_gpt2-large.json  -- $PY run_zeropoint.py --model openai-community/gpt2-large --dtype float32 --batch-size 50
run zp_pyth    6000  $R/raw_zp_pythia-2.8b.json -- $PY run_zeropoint.py --model EleutherAI/pythia-2.8b      --dtype float32 --batch-size 50
# 7B (bf16)
run flip_qwen  20000 $R/raw_flip_Qwen2.5-7B.json          -- $PY run_flip.py --model Qwen/Qwen2.5-7B          --dtype bfloat16 --batch-size 16
run flip_qwenI 20000 $R/raw_flip_Qwen2.5-7B-Instruct.json -- $PY run_flip.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --batch-size 16
run zp_sc      20000 $R/raw_zp_starcoder2-7b.json    -- $PY run_zeropoint.py --model bigcode/starcoder2-7b  --dtype bfloat16 --batch-size 16
run zp_qc      20000 $R/raw_zp_Qwen2.5-Coder-7B.json -- $PY run_zeropoint.py --model Qwen/Qwen2.5-Coder-7B  --dtype bfloat16 --batch-size 16

echo "=== run_all done $(date -Is) ===" | tee -a "$LOG"
