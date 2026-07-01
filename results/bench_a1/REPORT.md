# A1 on benchmark-sourced prefixes — RESULT

**Protocol.** 200 32-token prefixes sampled from the WikiText-103 test set, seed 0 (SimCTG / contrastive-search open-ended-generation protocol). Prefixes defined as TEXT (first 32 GPT-2 BPE tokens of each sampled prose segment, decoded back) so every model shares them. All runs greedy, 200 new tokens, HF sign-branch penalty, gauge shift added before the penalty, fp32 for the shift/penalty/argmax. See `REPORT.md` header table for old (16 hand-written prompts) vs new numbers.

## Table 1 — gauge flip rate (c=+5 vs c=-5)

| Model | gate flip(θ=1.0) | flip(θ=1.3) NEW | 95% CI (θ=1.3) | flip(θ=1.3) OLD (16-prompt) |
|---|---|---|---|---|
| gpt2-large | 0.000 PASS | **0.964** | [0.960, 0.968] | 0.941 |
| pythia-2.8b | 0.000 PASS | **0.938** | [0.931, 0.946] | 0.905 |
| gpt2 | 0.000 PASS | **0.582** | [0.539, 0.624] | 0.497 |
| Qwen2.5-7B | 0.000 PASS | **0.922** | [0.911, 0.932] | 0.871 |
| Qwen2.5-7B-Instruct | 0.000 PASS | **0.922** | [0.912, 0.931] | 0.897 |

Validity gate (θ=1.0 flip == 0 for every model): **PASS**

## Table 2 — zero-point (five checkpoints)

| Model | frac seen logit>0 NEW | (OLD) | median top-1 NEW | (OLD) | nat flip@1.3 NEW | (OLD) |
|---|---|---|---|---|---|---|
| gpt2 | 0.168 | (0.091) | -161.39 | (-97.5) | **0.705** | (0.569) |
| gpt2-large | 0.523 | (0.746) | 13.34 | (11.9) | **0.957** | (0.781) |
| starcoder2-7b | 0.776 | (0.828) | 18.75 | (15.5) | **0.962** | (0.858) |
| pythia-2.8b | 0.883 | (0.938) | 19.71 | (17.6) | **0.958** | (0.815) |
| Qwen2.5-Coder-7B | 0.946 | (0.986) | 24.75 | (24.2) | **0.970** | (0.796) |

Pooled natural flip@θ=1.3 (mean of 5 models): **0.910** (OLD 0.764)

## Table 3 (row 1) — fix leg (log_softmax before penalty)

| Model | flip(θ=1.0) | flip(θ=1.3) | expected |
|---|---|---|---|
| gpt2-large | 0.000 | 0.000 | 0.000 |
| pythia-2.8b | 0.000 | 0.000 | 0.000 |
| gpt2 | 0.000 | 0.000 | 0.000 |

## Reproducibility

- Environment: Python 3.11.14; torch 2.6.0+cu124; transformers 5.11.0; datasets 5.0.0;
  accelerate 1.13.0; tokenizers 0.22.2 (venv at `code/bench_a1/.venv`, pins mirror `env/uv.lock`).
- Hardware: single RTX 3090 (24 GB), GPU jobs serialized via a flock-based mutex.
- Dtypes: gpt2 / gpt2-large / pythia-2.8b fp32; Qwen2.5-7B, Qwen2.5-7B-Instruct, starcoder2-7b,
  Qwen2.5-Coder-7B bf16. In every case the gauge shift, penalty, and argmax are computed in fp32
  (logits cast with `.float()`), so the θ=1.0 no-op gate is exact regardless of model dtype.
- Prefix manifest: `results/bench_a1/prefixes.json` — 200 32-token prefixes sampled from the
  WikiText-103 test set with seed 0 (`Salesforce/wikitext`, `wikitext-103-raw-v1`, test split;
  prose-only filter: no headings, >= 40 GPT-2 tokens before truncation; prefixes stored as text so
  all models share them across tokenizers). FROZEN — do not regenerate.
- n = 200 prefixes x 200 greedy tokens = 40,000 aligned positions per model per comparison.
- CIs: percentile bootstrap over prefixes, 10^4 resamples, at θ=1.3.
- Exact rerun commands: `code/bench_a1/README.md`. Model revisions are pinned inside each
  `raw_*.json` (`revision` field).

## Validity gates

- θ=1.0 flip rate (c=+5 vs c=-5) = 0.000 exactly (0/40,000) on all five Table-1 models and all
  three fix-leg models: PASS.
- Harness equivalence: the batched decoder was verified token-identical to the original per-step
  `code/run_a1.py::generate` (θ=1.3, c=±5, gpt2) and batch-size invariant before the grid ran
  (`code/bench_a1/smoke_check.py` / CPU check).

## Deviations from the original 16-prompt harness

- Prompts: 200 WikiText-103 test prefixes (benchmark-sourced) instead of 16 hand-written prompts.
- Generation is batched across prefixes (left-padding, explicit position_ids); semantics verified
  identical to the per-step original.
- Zero-point runs use max_new=200 (original: 48) on the same 200 prefixes; metric definitions
  (frac seen-logit > 0 at c=0/θ=1; median of per-step top-1 logits; c_natural = -median;
  natural flip vs the raw c=0 decode at θ=1.3) match `code/run_a1_zeropoint.py` exactly.

## Reading of the results

Every effect the paper reports on the 16 hand-written prompts replicates on the field-standard
protocol, and is mostly stronger:
- Table 1 flips at θ=1.3 rise on all five models (e.g. gpt2 0.50 -> 0.58, gpt2-large 0.94 -> 0.96,
  Qwen2.5-7B 0.87 -> 0.92); the "50-94%" headline range becomes 58-96%.
- The zero-point spread across real checkpoints remains large (frac-positive 0.17 -> 0.95;
  median top-1 logit -161 -> +25), and the natural-gauge flip at θ=1.3 is 0.71-0.97 per model,
  pooled 0.910 (old pooled 0.764). gpt2's zero-point stats shift the most (frac_pos 0.09 -> 0.17,
  median -97.5 -> -161.4): with 200-token continuations on WikiText prose gpt2 spends more of the
  decode in high-uncertainty regions with strongly negative logits; the qualitative picture
  (gpt2 an outlier far below the sign boundary, the others above it) is unchanged. gpt2-large's
  frac_pos moves from 0.75 to 0.52 — still on the divide side on most tokens, but the point that a
  fixed repetition_penalty denotes different interventions across checkpoints is unaffected.
- The fix leg is 0.000 everywhere, as predicted by shift-invariance of log_softmax.
