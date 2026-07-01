# bench_a2 — A2 rerun on public benchmarks

Reruns the paper's A2 experiments with benchmark-sourced prompts/schemas, replacing the
18 hand-written prompts and 6 hand-written JSON schemas a reviewer flagged as arbitrary.

- **Part A** (JSON validity, paper-note Table 3 row 3): 200 real-world schemas sampled from
  **JSONSchemaBench** (HF `epfl-dlab/JSONSchemaBench`; guidance-ai,
  github.com/guidance-ai/jsonschemabench), Qwen2.5-Coder-7B, θ ∈ {1.0, 1.1, 1.3}, raw vs fix.
- **Part B** (threshold + delimiter-flip, paper-note §A2 + Table 3 row 2): the 164 official
  **HumanEval** prompts (HF `openai/openai_humaneval`, `prompt` field) as code seeds,
  StarCoder2-7B, θ ∈ {1.0 … 1.5}, raw + fix; instrumentation imported verbatim from
  `code/run_a2.py` / metric definitions from `code/analyze_a2.py`.

## Environment

```
cd code/bench_a2
uv sync --frozen            # pyproject.toml + uv.lock copied from env/ (torch 2.6.0+cu124, transformers 5.11.0)
uv pip install jsonschema   # jsonschema 4.26.0 (+ datasets 5.0.0 already pinned)
export HF_HUB_CACHE=$HF_HUB_CACHE HF_HOME=$HF_HUB_CACHE
```

## Rerun commands (in order)

```
# 1. Build the schema manifest (deterministic, seed 0) -> results/bench_a2/schemas.json
.venv/bin/python build_schemas.py

# 2. Part A: JSON validity grid (Qwen2.5-Coder-7B, bf16, greedy, 512 new tokens, bs 8)
.venv/bin/python run_json_bench.py --thetas 1.0,1.1,1.3 --ops raw,fix

# 3. Part B: StarCoder2-7B raw + fix (bf16, fp32 logit upcast, greedy, 256 new tokens)
.venv/bin/python run_a2_humaneval.py
.venv/bin/python run_a2_humaneval.py --fix

# 4. (optional replicate) Qwen2.5-Coder-7B raw
.venv/bin/python run_a2_humaneval.py \
    --model Qwen/Qwen2.5-Coder-7B \
    --out results/bench_a2/humaneval_qwen_raw.json

# 5. Analyze Part B + combine everything into results/bench_a2/summary.json
.venv/bin/python analyze_a2_humaneval.py
.venv/bin/python analyze_a2_humaneval.py \
    --raw results/bench_a2/humaneval_qwen_raw.json \
    --raw-fix /nonexistent
.venv/bin/python summarize.py
```

## Sampling procedure (Part A manifest)

Pool = `Github_easy` then `Github_medium`, splits train+val+test (3907 schemas), shuffled with
`random.Random(0)`. A schema is kept iff it (a) parses as a JSON object, (b) declares an object
instance, (c) compiles under `jsonschema.validators.validator_for(schema).check_schema`,
(d) contains no remote (`http[s]` `$ref`) reference, and (e) is ≤ 600 GPT-2 tokens serialized.
First 200 kept → `results/bench_a2/schemas.json` (196 Github_easy + 4 Github_medium; drafts
04/06/07/2020-12, validated with the draft each schema declares via `validator_for`).

## Protocol notes

- Part A mirrors `code/run_a2_downstream.py`: same `RepPen` LogitsProcessor (exact HF
  semantics; `--fix` = `log_softmax` before penalizing), greedy batched decode with left
  padding, stop strings `["\n\n", "```"]`, `first_json` brace-matching extraction. Only the
  prompt changes: the benchmark schema is embedded ("Output ONLY a single JSON object that
  conforms to this JSON Schema … JSON object:"). max_new_tokens = 512 (largest manifest schema
  is 600 GPT-2 tokens; instances are much shorter than their schemas — no length truncation of
  valid objects was observed at θ=1.0).
- Validation = `jsonschema` iter_errors == 0 against the actual benchmark schema, draft
  auto-detected from `$schema` via `validator_for`.
- Part B imports `run_a2.generate` (per-position pre-penalty top-1/runner-up logits, fp32
  upcast before penalty/argmax/threshold, argmax-tie discipline so θ=1.0 gives exactly 0
  flips) and reuses `analyze_a2`'s clean-position selection, threshold predicate, balanced
  accuracy/TPR/TNR, runner-up fraction, structural tagging, and per-record structural-flip
  rate. HumanEval has no prose arm, so the code-vs-prose domain-ratio control is N/A here.
