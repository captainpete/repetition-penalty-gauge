# Gauge dependence and structured-output corruption in sign-branched repetition penalties

[![arXiv](https://img.shields.io/badge/arXiv-2607.09791-b31b1b.svg)](https://arxiv.org/abs/2607.09791)

Companion code, data, and paper for the finding that the multiplicative (CTRL-style) repetition
penalty, as shipped by HuggingFace Transformers, vLLM, llama.cpp, and most of the inference
ecosystem, branches on the **sign of the raw logit** and is therefore **gauge-dependent**: since
the softmax is unchanged by adding a constant to every logit, a model's logit zero-point is
arbitrary, and the penalty reads that arbitrary point.

Two measurable consequences, with shift-invariant penalties as measured controls:

- **The penalty is not well-defined.** Re-centring a model's logits by a constant is a provable
  no-op at `repetition_penalty = 1.0`, yet at a routine `1.3` it changes 58–96% of greedy tokens
  (200 WikiText-103 test prefixes, 40,000 positions per model, five models up to 7B, base and
  RLHF). Real checkpoints already sit at different zero-points (divide-fraction spans 0.17–0.95),
  so a fixed `repetition_penalty` is a different operation on every model.
- **It corrupts structured output.** On 200 real-world JSONSchemaBench schemas, `1.3` drops
  schema-valid output from 97% to 23% (Qwen2.5-Coder), via a closed-form threshold
  `g < z_top · (1 − 1/θ)` that flips a confident, correct delimiter to the runner-up (balanced
  accuracy 0.999 over 48,919 positions on the 164 HumanEval prompts; every flip lands on the
  runner-up).
- **Normalizing before the penalty.** In our measurements, applying the penalty to normalized
  log-probabilities (`log_softmax(logits)`) instead of raw logits removes both effects; JSON
  validity returns to its θ=1 level (97%). HuggingFace already ships this operator
  (`LogitNormalization`); it is off by default and applied *after* the penalty. Subtractive
  presence/frequency penalties are shift-invariant by construction and measure zero gauge
  sensitivity; not every stack exposes them.
- **Cross-stack replication.** Both effects reproduce *inside* vLLM (0.8.5.post1; operator
  verified identical in `main` @ `e196268`) and llama.cpp (master @ `4fc4ec5`), through each stack's
  own sampler and knob, on the same benchmark inputs: A1 flip rate at `1.3` is 0.964 in both
  (vLLM token-identical to the HF run), θ=1.0 gates exactly 0/40,000; JSON validity falls
  0.97 → 0.24 (vLLM) / 0.29 (llama.cpp), and llama.cpp's default 64-token window makes it worse
  (0.12). See `code/smoke_*` + `results/smoke_*_bench` (each has its own README/REPORT).
- **Ecosystem survey.** `STACKS-SURVEY.md` documents the identical sign-branch independently
  reimplemented in TGI, SGLang, TensorRT-LLM, ExLlamaV2, mlx-lm, LMDeploy, KoboldCpp, mistral.rs,
  candle, Ollama, and others (verbatim excerpts at pinned commits); none normalizes first.

## Paper

Published as [arXiv:2607.09791](https://arxiv.org/abs/2607.09791) (cs.LG).

- **`paper/paper-note.pdf`** — the short note (8 pages): mechanism and cost figures, the
  measurements, the cross-stack replication, and the normalized variant. Start here.
- **`paper/paper.pdf`** — the extended version (full derivations, per-model tables, pre-registered
  audit of what did and did not replicate, related work).

Sources: `paper/paper-note.tex`, `paper/paper.tex`, `paper/refs.bib`. Build with `latexmk -pdf`.

## Layout

```
paper/       the note + extended paper (.tex, .bib, .pdf)
code/        run_*.py (generate raw.json) and analyze_*.py (produce REPORT.md + summary.json);
             smoke_vllm/ + smoke_llamacpp/ replicate A1+A2 inside those stacks' own samplers
results/     the raw outputs, per-experiment REPORT.md, and summary.json that back every number
prereg/      pre-registered hypotheses and frozen decision rules (written before the runs)
findings/    honest record of what was verified, partial, and refuted
env/         pinned environment (pyproject.toml + uv.lock)
REPLICATION.md   how to reproduce every experiment
```

## Reproduce

All experiments are inference-only and run on a single 24–48 GB GPU. The environment is pinned:

```bash
uv venv --python 3.11 && uv sync --frozen      # uses env/pyproject.toml + env/uv.lock
```

Then, for example (full commands and models in `REPLICATION.md`):

```bash
python code/run_a1.py                    # A1 gauge sweep  -> runs/A1/raw.json
python code/analyze_a1b.py --dir runs/A1b
python code/run_a2.py                    # A2 threshold    -> runs/A2/raw.json
python code/run_a1.py --fix              # the normalize-before-penalize fix
```

Models are public HuggingFace checkpoints (`gpt2`, `openai-community/gpt2-large`,
`EleutherAI/pythia-2.8b`, `Qwen/Qwen2.5-7B`(-Instruct), `bigcode/starcoder2-7b`,
`Qwen/Qwen2.5-Coder-7B`). 7B models use `--dtype bfloat16` to fit the card; the gauge shift and
argmax decision stay in fp32, so the no-op gate is exact regardless of model dtype. The committed
`results/` were produced by exactly these scripts.

## Citation

```bibtex
@misc{hollows2026repetition,
  title         = {Gauge dependence and structured-output corruption in sign-branched
                   repetition penalties: measurements across models, inference stacks,
                   and alternative repetition controls},
  author        = {Hollows, Peter},
  year          = {2026},
  eprint        = {2607.09791},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2607.09791},
  url           = {https://arxiv.org/abs/2607.09791}
}
```

## License

- **Code** (`code/`, scripts): Apache License 2.0 — see [`LICENSE`](LICENSE).
- **Paper and text** (`paper/`, `prereg/`, `findings/`, this README): Creative Commons Attribution
  4.0 International (CC BY 4.0) — see [`paper/LICENSE`](paper/LICENSE).

© 2026 Peter Hollows.
