# Replication

All experiments are inference-only and fit a single 24–48 GB GPU.

## Environment (pinned)

`env/pyproject.toml` + `env/uv.lock` pin the exact stack (torch 2.6.0+cu124, transformers 5.11.0).
To recreate:

```bash
uv venv --python 3.11 && uv sync --frozen
```

The cu124 torch wheels need a recent NVIDIA driver (≥525). Models are public HuggingFace
checkpoints, downloaded on first use; set `HF_HUB_CACHE` to a roomy disk.

## How each experiment runs

Each `run_*.py` generates a `raw*.json`; each `analyze_*.py` applies the frozen decision rule and
writes `REPORT.md` + `summary.json`. The committed `results/` were produced by these scripts;
re-running with the same arguments reproduces them (greedy/deterministic where applicable; sampled
runs fix a seed). Each `run_*.py`/`analyze_*.py` takes `--out`/`--dir`/`--raw` and
experiment-specific flags (see its `argparse`). Model revisions are pinned in each `summary.json`.

7B models require `--dtype bfloat16` to fit the card; the gauge shift, the penalty, and the argmax
decision are computed in fp32, so the θ=1.0 no-op gate is exact regardless of model dtype.

## Per-experiment commands

**A1 — gauge non-invariance** (gpt2 / gpt2-large / pythia-2.8b, fp32; Qwen2.5-7B(-Instruct), bf16).
Greedy + seeded-sampling, `c ∈ {−5..+5}` added to the logits, `θ` swept.

```bash
python code/run_a1.py --model gpt2            # -> runs/A1/raw.json  (--out to change)
python code/analyze_a1.py  --raw runs/A1/raw.json
python code/analyze_a1b.py --dir runs/A1b     # flip-rate endpoint, θ sweep across models
python code/diag_divergence.py runs/A1/raw.json
```

**A1 zero-point realism** — where each real checkpoint sits on the sign boundary, and the flip
rate at each model's own median-centred gauge:

```bash
python code/run_a1_zeropoint.py --model gpt2        # -> runs/A1_zeropoint/
python code/analyze_a1_zeropoint.py --out-dir runs/A1_zeropoint
```

**A2 — corruption threshold** (starcoder2-7b, bf16 with fp32 decision). JSON/Python/prose prompts
with per-position logit instrumentation:

```bash
python code/run_a2.py                          # -> runs/A2/raw.json
python code/analyze_a2.py --raw runs/A2/raw.json
python code/explore_delim.py                   # brackets-only re-slice
```

**The fix** (normalize before penalizing) — a `--fix` flag on the run scripts inserts
`log_softmax(logits)` immediately before the penalty:

```bash
python code/run_a1.py --fix                    # A1 divergence -> 0 at every θ
python code/run_a2.py --fix                    # A2 corruption removed
```

**Downstream quality** (does the corruption matter end to end?) and **loop-suppression**
(does the fixed operator still break loops?):

```bash
python code/run_a2_downstream.py               # JSON validity + HumanEval, raw vs fix
python code/analyze_a2_downstream.py --raw runs/A2_downstream/raw.json
python code/run_a1_loopcheck.py --model gpt2   # repetition rate vs θ under the fix
```

## Benchmark replications (primary numbers in the note)

The note's headline numbers come from standard benchmarks; the original pre-registered runs
(16 fixed prompts, 6 hand-written schemas) remain below and in `results/`, with consistent
verdicts.

- **A1 on WikiText-103**: `code/bench_a1/` — 200 32-token prefixes from the WikiText-103 test
  set (seed 0; frozen manifest `results/bench_a1/prefixes.json`), flip rates + zero-point stats
  for all five models, fix leg. See `code/bench_a1/README.md` for exact commands.
- **A2 on HumanEval + JSONSchemaBench**: `code/bench_a2/` — threshold/delimiter metrics on the
  164 HumanEval prompts (StarCoder2-7B, Qwen2.5-Coder-7B) and JSON validity on 200
  JSONSchemaBench schemas (frozen manifest `results/bench_a2/schemas.json`), raw vs fix. The
  three large per-position raw files ship gzipped (`results/bench_a2/*.json.gz`); gunzip before
  running the analyzers.

## Cross-stack smoke replications (vLLM, llama.cpp)

A1 and A2 are also measured *inside* vLLM and llama.cpp, through each stack's own sampler
and its own `repetition_penalty` / `repeat_penalty` knob. These have their own pinned
environments and build steps:

- `code/smoke_vllm/README.md` — vLLM 0.8.5.post1 (cu124), offline `LLM` API; the gauge
  shift rides a V0 per-request logits processor (verified to precede the penalty).
- `code/smoke_llamacpp/README.md` — llama.cpp master @ `4fc4ec5`, C++ drivers against
  `libllama`; the gauge shift is a full-vocab `logit_bias` sampler placed before
  `penalties` in the chain.

Findings, raw generations, and source-verification excerpts: `results/smoke_vllm/REPORT.md`
and `results/smoke_llamacpp/REPORT.md`. The benchmark-input refresh (same manifests as the
primary experiments) is in `results/smoke_vllm_bench/` and `results/smoke_llamacpp_bench/`,
driven by `code/smoke_*/run_bench_*.sh` / `run_a*_bench.sh`.

## Building the paper

```bash
cd paper && latexmk -pdf paper-note.tex     # the short note
cd paper && latexmk -pdf paper.tex          # the extended version
```

## Calibration map + matched-suppression comparison (Section 5 of the note)

```bash
# calibration map (GPU decode; one invocation per model, see run_fixcal.py docstring)
python code/run_fixcal.py --model gpt2 --out results/fix_calibration/raw_gpt2.json
python code/analyze_fixcal.py --raws 'results/fix_calibration/raw_*.json' --out-dir results/fix_calibration

# matched-suppression head-to-head (GPU decode; stages per model, see run_matched.py docstring)
python code/run_matched.py --model gpt2 --stage pairs --out results/matched_strength/raw_gpt2_pairs.json
python code/analyze_matched.py --raws 'results/matched_strength/raw_*.json' --out-dir results/matched_strength

# zero-point / reachability adjudication (no decode; assembles existing raws)
python code/analyze_zpreach.py --zeropoint-dirs results/a1_zeropoint \
  --fixcal-dir results/fix_calibration --out-dir results/zeropoint_reachability
```

# in-domain dose verification for the HumanEval pairs (CPU; tokenizer only)
python code/check_matched_indomain.py   # -> reproduces results/matched_strength/INDOMAIN.txt

The analyzers regenerate each directory's `summary.json` and `REPORT.md` from the committed
raws (CPU-only). The judge arm of `run_matched.py` additionally needs a `judge_lib.py`
(Qwen2.5-7B-Instruct pairwise judge, lazy-imported inside that stage only); the judge outputs
it produced ship in `results/matched_strength/raw_*_open.json`. Frozen decision rules:
`prereg/FIXCAL_PREREG.md`, `prereg/MATCHED_PREREG.md`, `prereg/ZPREACH_PREREG.md`.
