# Smoke replication of A1 (gauge) and A2 (JSON validity) inside llama.cpp

Measures the two published repetition-penalty effects using **llama.cpp's own
`repeat_penalty` sampler** (`llama_sampler_init_penalties`), so a maintainer
issue can cite numbers from their stack rather than inferred from HuggingFace.

Everything here except this README and the `src/`/`*.py` scripts is gitignored
(the llama.cpp clone, the venv, `models/*.gguf`, build output).

## Layout
- `src/a1_driver.cpp` — A1 gauge probe. Chain: `logit_bias(all vocab, +c)` -> `penalties(last_n, theta)` -> `greedy`. Builds `cur_p` over the full vocab and calls `llama_sampler_apply` + `llama_sampler_accept` directly (never `llama_sampler_sample`), so the bias covers every token and no backend pre-sampling path is hit.
- `src/a2_driver.cpp` — A2 JSON-validity probe. Chain: `penalties(last_n, theta)` -> `greedy`. Emits generated text; runs both `penalty_last_n` conditions in one process.
- `analyze_a1_flip.py` — flip rate per theta + validity gate.
- `analyze_a2_json.py` — schema-valid rate per (theta, penalty_last_n); `first_json`/`json_valid`/schemas copied VERBATIM from `code/run_a2_downstream.py`.

## Rebuild / rerun

### 1. Build llama.cpp (CUDA)
```
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp   # commit 4fc4ec5541b243957ae5099edb67372f8f3b550e
cd llama.cpp
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j 32
```

### 2. Conversion venv (CPU torch) + GGUFs
```
uv venv --python 3.11 venv && . venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
# NOTE: conversion/gpt2.py in this commit mishandles the .attn.bias mask buffers
# (yields them through the base mapper -> "Can not map tensor 'h.0.attn.bias'").
# Patched locally to `return` (skip) — these are constant causal-mask buffers, not weights.
python llama.cpp/convert_hf_to_gguf.py <gpt2-large snapshot dir> --outtype f16 --outfile models/gpt2-large-f16.gguf
python llama.cpp/convert_hf_to_gguf.py <qwen2.5-coder-7b snapshot dir> --outtype bf16 --outfile models/qwen2.5-coder-7b-bf16.gguf
```

### 3. Compile drivers
```
g++ -O2 -std=c++17 src/a1_driver.cpp -o a1_driver -Illama.cpp/include -Illama.cpp/ggml/include -Lllama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/llama.cpp/build/bin
g++ -O2 -std=c++17 src/a2_driver.cpp -o a2_driver -Illama.cpp/include -Illama.cpp/ggml/include -Lllama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/llama.cpp/build/bin
```

### 4. Run
```
# A1: gpt2-large, CPU (deterministic, no GPU)
CUDA_VISIBLE_DEVICES="" ./a1_driver -m models/gpt2-large-f16.gguf --max-new 200 --penalty-last-n 1024 --out ../../results/smoke_llamacpp/a1_raw.json
python analyze_a1_flip.py

# A2: Qwen2.5-Coder-7B, GPU (RTX 3090)
./a2_driver -m models/qwen2.5-coder-7b-bf16.gguf --max-new 160 --reps 8 -ngl 99 --out ../../results/smoke_llamacpp/a2_raw.json
python analyze_a2_json.py
```

## Key implementation notes
- **`penalty_last_n = -1` does NOT mean "context size" for the repeat penalty.**
  `llama_sampler_init_penalties` clamps `std::max(penalty_last_n, 0)`, so `-1` -> `0` = penalty disabled (the header comment and `--repeat-last-n` help text say "-1 = context size", but `common/sampling.cpp` passes the value through unresolved). To cover the whole context you must pass an explicit large positive N. The drivers pass `1024`.
- The penalty is a **sliding window** (ring buffer of the last `penalty_last_n` accepted tokens), not an all-seen set like HF. A2 measures both the whole-context match and the default-64 window.
- Prompt tokens are fed to `llama_sampler_accept` before generation so they sit in the penalty window (HF's seen-set includes the prompt).
