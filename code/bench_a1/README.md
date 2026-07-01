# A1 re-run on benchmark-sourced prefixes (WikiText-103)

Re-runs the paper's A1 numbers (paper-note Tables 1–3, the flip-rate / zero-point / fix-leg
measurements) on the field-standard open-ended-generation protocol instead of the 16 hand-written
prompts: **200 32-token prefixes sampled from the WikiText-103 test set, seed 0** (the
SimCTG / contrastive-search protocol). Penalty semantics are identical to `code/run_a1.py`,
`code/run_a1_zeropoint.py`, and `code/analyze_a1b.py` — HF sign-branch over all previously-seen
ids, gauge shift added to the logits BEFORE the penalty, fp32 for the shift/penalty/argmax even for
bf16 models.

## Environment

`.venv/` is a uv venv pinned to the project stack: torch 2.6.0+cu124, transformers 5.11.0,
datasets 5.0.0, accelerate 1.13.0, tokenizers 0.22.2 (Python 3.11). Recreate:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 transformers==5.11.0 accelerate==1.13.0 datasets==5.0.0 \
  numpy==2.4.6 tokenizers==0.22.2 safetensors==0.8.0 sentencepiece==0.2.1 tqdm==4.68.2
```

Always: `export HF_HUB_CACHE=$HF_HUB_CACHE`. All GPU work via
a flock-based GPU mutex on the original host; run jobs sequentially on a shared GPU.

## Files

- `make_prefixes.py` — builds `results/bench_a1/prefixes.json` (WRITE ONCE; other agents reuse it).
- `bench_lib.py` — batched greedy decode + exact penalty semantics (shared by both runners).
- `run_flip.py` — Table 1 (flip rate c=+5 vs c=-5) and Table 3 fix leg (`--fix`).
- `run_zeropoint.py` — Table 2 (frac seen-logit >0, median top-1, natural-gauge flip@1.3).
- `run_flip.py --subtractive ALPHA` — subtractive presence-style control leg (gauge-invariant;
  flip rate expected 0). Writes `raw_flip_<label>_subtractive.json`.
- `analyze.py` — reads all `raw_*.json`, writes `summary.json` + `REPORT.md` (old vs new).
- `analyze_controls.py` — subtractive flip rates + cascade-honesty stats (first-divergence
  position, before/after flip) from the CTRL runs. Writes `summary_controls.json` (CPU, no GPU).
- `smoke_check.py` — verifies θ=1.0 gate = 0, batch-size invariance, and match vs the original
  per-step `run_a1.generate`.

## Rerun commands (exact)

Prefix manifest (CPU, once):
```bash
export HF_HUB_CACHE=$HF_HUB_CACHE
export HF_DATASETS_CACHE=$(pwd)/.cache/datasets
.venv/bin/python make_prefixes.py
```

Table 1 — flip rates (five models):
```bash
G=
$G -m 6000  -n flip_gpt2   -- .venv/bin/python run_flip.py --model gpt2                       --dtype float32 --batch-size 100
$G -m 6000  -n flip_g2l    -- .venv/bin/python run_flip.py --model openai-community/gpt2-large --dtype float32 --batch-size 50
$G -m 6000  -n flip_pyth   -- .venv/bin/python run_flip.py --model EleutherAI/pythia-2.8b      --dtype float32 --batch-size 50
$G -m 20000 -n flip_qwen   -- .venv/bin/python run_flip.py --model Qwen/Qwen2.5-7B             --dtype bfloat16 --batch-size 16
$G -m 20000 -n flip_qwenI  -- .venv/bin/python run_flip.py --model Qwen/Qwen2.5-7B-Instruct    --dtype bfloat16 --batch-size 16
```

Table 3 — fix leg (three models, `--fix`):
```bash
$G -m 6000 -n fix_gpt2 -- .venv/bin/python run_flip.py --model gpt2                       --dtype float32 --fix --batch-size 100
$G -m 6000 -n fix_g2l  -- .venv/bin/python run_flip.py --model openai-community/gpt2-large --dtype float32 --fix --batch-size 50
$G -m 6000 -n fix_pyth -- .venv/bin/python run_flip.py --model EleutherAI/pythia-2.8b      --dtype float32 --fix --batch-size 50
```

Subtractive (presence-style) control leg — `--subtractive 1.0`, single α, no θ grid (five models):
```bash
$G -m 6000  -n sub_gpt2  -- .venv/bin/python run_flip.py --model gpt2                       --dtype float32  --subtractive 1.0 --batch-size 100
$G -m 6000  -n sub_g2l   -- .venv/bin/python run_flip.py --model openai-community/gpt2-large --dtype float32  --subtractive 1.0 --batch-size 50
$G -m 6000  -n sub_pyth  -- .venv/bin/python run_flip.py --model EleutherAI/pythia-2.8b      --dtype float32  --subtractive 1.0 --batch-size 50
$G -m 20000 -n sub_qwen  -- .venv/bin/python run_flip.py --model Qwen/Qwen2.5-7B             --dtype bfloat16 --subtractive 1.0 --batch-size 16
$G -m 20000 -n sub_qwenI -- .venv/bin/python run_flip.py --model Qwen/Qwen2.5-7B-Instruct    --dtype bfloat16 --subtractive 1.0 --batch-size 16
```

Table 2 — zero-point (five checkpoints):
```bash
$G -m 6000  -n zp_gpt2 -- .venv/bin/python run_zeropoint.py --model gpt2                       --dtype float32 --batch-size 100
$G -m 6000  -n zp_g2l  -- .venv/bin/python run_zeropoint.py --model openai-community/gpt2-large --dtype float32 --batch-size 50
$G -m 6000  -n zp_pyth -- .venv/bin/python run_zeropoint.py --model EleutherAI/pythia-2.8b      --dtype float32 --batch-size 50
$G -m 20000 -n zp_sc   -- .venv/bin/python run_zeropoint.py --model bigcode/starcoder2-7b       --dtype bfloat16 --batch-size 16
$G -m 20000 -n zp_qc   -- .venv/bin/python run_zeropoint.py --model Qwen/Qwen2.5-Coder-7B        --dtype bfloat16 --batch-size 16
```

Analyze:
```bash
.venv/bin/python analyze.py            # -> results/bench_a1/{summary.json, REPORT.md}
.venv/bin/python analyze_controls.py   # -> results/bench_a1/summary_controls.json (CPU; subtractive
                                       #    flip rates + cascade-honesty stats on the CTRL runs)
```

## Notes / deviations from the original harness

- Prefixes are TEXT (shared across models); the original A1 used 16 hand-written prompts.
- Batched greedy decode (left-padding + explicit attention_mask/position_ids). Verified against the
  original per-step `run_a1.generate` and shown batch-size invariant; the θ=1.0 gate is exact by
  construction (adding a constant never changes an argmax).
- Zero-point uses max_new=200 (the original used 48); everything else matches
  `run_a1_zeropoint.py` (frac seen-logit>0 at c=0/θ=1, median of per-step top-1 logits,
  c_natural = −median, natural flip = greedy mismatch vs the raw c=0 decode).
