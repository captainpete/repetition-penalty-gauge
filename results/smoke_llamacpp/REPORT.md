# Smoke replication of A1 + A2 inside llama.cpp itself

Both published measurements (HF-transformers stack) were re-measured **inside
llama.cpp's own sampler stack**, using `llama_sampler_init_penalties`
(`repeat_penalty`), so a maintainer issue can cite numbers produced by their code.

- **llama.cpp commit:** `4fc4ec5541b243957ae5099edb67372f8f3b550e` (master, 2026-07-01 10:29:22 -0700)
- **Build:** `cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`, nvcc 12.9, gcc 10.5. CUDA build succeeded on the first attempt.
- **GGUFs** (converted with the repo's own `convert_hf_to_gguf.py`, CPU-torch venv, stored in `code/smoke_llamacpp/models/`, gitignored):
  - `gpt2-large-f16.gguf` — openai-community/gpt2-large, **f16**
  - `qwen2.5-coder-7b-bf16.gguf` — Qwen/Qwen2.5-Coder-7B, **bf16** (matches the precision of the HF A2 run)
- **Drivers:** small C++ programs linked against the built `libllama` (route chosen over llama-cpp-python to avoid any version/operator mismatch). Source: `code/smoke_llamacpp/src/a1_driver.cpp`, `src/a2_driver.cpp`. Analysis: `analyze_a1_flip.py`, `analyze_a2_json.py`.

## Headline results

| Measurement | llama.cpp (this run) | HF reference (paper) |
|---|---|---|
| A1 flip rate, θ=1.0 (validity gate) | **0.0000** (0/3200) — PASS | 0 by construction |
| A1 flip rate, θ=1.15 | **0.9228** | — |
| A1 flip rate, θ=1.3 | **0.9413** | **0.941** |
| A2 JSON valid, θ=1.0 | **1.000** (48/48) | 1.0 |
| A2 JSON valid, θ=1.1 | **1.000** (48/48) | 1.0 |
| A2 JSON valid, θ=1.3 | **0.167** (8/48) | 0.0 |

Both effects replicate inside llama.cpp. The A1 flip rate at θ=1.3 matches the
HF number to three decimal places (0.9413 vs 0.941) despite a different
inference engine, f16 GGUF weights, and a different penalty-window mechanism.

---

## Deliverable 1 — source verification

All excerpts from `src/llama-sampler.cpp` at commit `4fc4ec55…`.

### (a) Sign-branched multiplicative penalty on raw logits

`llama_sampler_penalties_apply`, **src/llama-sampler.cpp:2688-2694**:

```cpp
// The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
// This is common fix for this problem, which is to multiply by the penalty instead of dividing.
if (cur_p->data[i].logit <= 0) {
    cur_p->data[i].logit *= ctx->penalty_repeat;
} else {
    cur_p->data[i].logit /= ctx->penalty_repeat;
}
```

This is the CTRL form: divide positive raw logits by θ, multiply non-positive
raw logits by θ. (Boundary detail: llama.cpp branches on `logit <= 0`, HF's
`RepetitionPenaltyLogitsProcessor` on `logit < 0`; a logit exactly 0.0 is
unchanged by either branch, so behavior is identical.)

### (b) Seen-set semantics: sliding window, not all-seen

The penalty state is a **ring buffer of the last `penalty_last_n` accepted
tokens** plus a count map — `struct llama_sampler_penalties`,
**src/llama-sampler.cpp:2622-2632**:

```cpp
struct llama_sampler_penalties {
    const int32_t penalty_last_n;
    ...
    ring_buffer<llama_token> prev;
    // a frequency map to count token occurrences
    std::unordered_map<llama_token, int> token_count;
};
```

Eviction on accept — `llama_sampler_penalties_accept`, **src/llama-sampler.cpp:2644-2656**:

```cpp
ctx->token_count[token]++;
// if the ring buffer is full, remove the oldest token
if (ctx->prev.size() >= (size_t) ctx->penalty_last_n) {
    const auto old = ctx->prev.front();
    ctx->token_count[old]--;
    if (ctx->token_count[old] == 0) {
        ctx->token_count.erase(old);
    }
}
ctx->prev.push_back(token);
```

So llama.cpp penalizes **only the last `penalty_last_n` tokens** (default 64:
`common/common.h:234`, `int32_t penalty_last_n = 64;`), unlike HF, which
penalizes every token ever seen in the sequence.

**Bonus finding — `-1` does NOT mean "context size" for repeat_penalty.**
The header (`include/llama.h:1407`) and the `--repeat-last-n` help text
(`common/arg.cpp:1830`) both document `-1 = context size`, but
`llama_sampler_init_penalties` clamps it — **src/llama-sampler.cpp:2748-2754**:

```cpp
penalty_last_n = std::max(penalty_last_n, 0);
const bool is_empty = (penalty_last_n == 0 || ...);
if (is_empty) {
    return llama_sampler_init_empty("?penalties");
}
```

and `common/sampling.cpp:358` passes the user's value through unresolved:

```cpp
samplers.push_back(llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
```

So `--repeat-last-n -1` silently **disables** the repeat penalty instead of
covering the whole context (the DRY sampler, by contrast, does resolve `-1` at
src/llama-sampler.cpp:2935). For this reason the drivers pass an explicit
`penalty_last_n = 1024` (> prompt + 200/160 new tokens) to emulate HF's
all-seen semantics.

### (c) Chain application order

`llama_sampler_chain_apply`, **src/llama-sampler.cpp:642-662** — samplers are
applied strictly in insertion order:

```cpp
static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;
    ...
    for (auto & smpl : chain->samplers) {
        ...
        llama_sampler_apply(smpl.ptr, cur_p);
    }
}
```

`chain->samplers` is a vector appended to by `llama_sampler_chain_add`
(src/llama-sampler.cpp:876-882), so a `logit_bias` stage added before the
`penalties` stage provably executes first. (One caveat encoded in the loop: if
the chain has been backend-initialized, a leading run of backend-capable
samplers can be fused into the graph; our drivers never call
`llama_sampler_chain_backend_init` and never use `llama_sampler_sample`, so
the plain in-order CPU path at lines 649-661 is the one exercised.)

---

## Deliverable 2 — A1 gauge probe (gpt2-large)

**Method.** `src/a1_driver.cpp`, replicating `code/run_a1.py`'s design and its
16 canonical prompts. Chain, in order:

1. `llama_sampler_init_logit_bias(n_vocab, n_vocab, bias)` with **all 50257
   vocab tokens** biased by the same constant c (a softmax no-op),
2. `llama_sampler_init_penalties(1024, θ, 0, 0)` — repeat penalty only,
3. `llama_sampler_init_greedy()`.

The driver builds `cur_p` over the full vocabulary from
`llama_get_logits_ith(ctx, -1)` and calls `llama_sampler_apply` +
`llama_sampler_accept` manually (the exact pattern documented in
include/llama.h:1480-1489), never `llama_sampler_sample` — this guarantees the
bias covers every token and no backend pre-sampled-logits path is entered.
Prompt tokens are `accept`ed into the penalty window before generation
(HF's seen-set includes the prompt). Generation is a **fixed 200 tokens per
prompt, continuing past EOS**, so positions align: 16 × 200 = 3200 aligned
positions per θ. Fresh `llama_context` and sampler chain per (prompt, θ, c)
run; identical batch sizes across runs.

- `penalty_last_n = 1024` (explicit, since `-1` disables — see above) ≥ prompt
  + 200 generated, i.e. whole-context ≡ HF all-seen semantics for these lengths.
- `penalize_nl`: the current API has **no such flag** — older versions of
  `llama_sampler_init_penalties` took `penalize_nl`/`ignore_eos`; the current
  4-argument signature penalizes every token in the window, newline included,
  matching HF.
- Backend: **CPU** (`CUDA_VISIBLE_DEVICES=""`) for bit-exact determinism.
  f16 GGUF weights (compute in f32 on CPU).

**Results** (`a1_raw.json`, `a1_summary.json`):

| θ | flip rate (c=+5 vs c=−5) | positions |
|---|---|---|
| 1.0 | **0.0000** (0/3200) — VALIDITY GATE PASS | 3200 |
| 1.15 | 0.9228 | 3200 |
| 1.3 | **0.9413** | 3200 |

HF reference at θ=1.3: **0.941**. A uniform logit shift — provably invisible
at θ=1 (and empirically: zero flips) — flips **94.1%** of llama.cpp's greedy
tokens once its own `repeat_penalty` is set to 1.3.

## Deliverable 3 — A2 JSON validity (Qwen2.5-Coder-7B)

**Method.** `src/a2_driver.cpp` + `analyze_a2_json.py`. Prompt construction
(`"Output ONLY a single JSON object describing {desc}. JSON:\n"`), the 6
`JSON_TASKS` schemas, `first_json`, and the `json_valid` type-checker are
copied **verbatim** from `code/run_a2_downstream.py`. Greedy; `max_new = 160`
(the original's JSON budget); stop strings `"\n\n"` and triple-backtick (as
original); also stops on EOG. 8 reps per schema per cell (as original; greedy
reps are identical — confirmed 0/36 conditions with any variation, i.e. fully
deterministic on CUDA at batch size 1). Chain: `penalties(last_n, θ)` →
`greedy`. Prompt tokens accepted into the window. GGUF **bf16** (same precision
as the HF run); full GPU offload on an RTX 3090.

**Results** (`a2_raw.json`, `a2_summary.json`) — schema-valid rate, n=48/cell:

| penalty_last_n | θ=1.0 | θ=1.1 | θ=1.3 |
|---|---|---|---|
| 1024 (whole context ≈ HF semantics) | 1.000 | 1.000 | **0.167** |
| 64 (llama.cpp default — what real users hit) | 1.000 | 1.000 | **0.167** |
| HF reference (all-seen) | 1.0 | 1.0 | 0.0 |

The two window conditions produced **byte-identical text in 144/144 paired
generations**: greedy JSON completions finish in well under 64 generated
tokens, so the ring buffer never evicts. The default-64 window is therefore
*not* protective for short structured outputs — real users at
`--repeat-penalty 1.3` get the full collapse.

Failure mode at θ=1.3 is structural corruption, not drift — e.g. schema 0:

- θ=1.0: `{"name": "John Doe", "age": 30, "email": "johndoe@example.com"}` (valid)
- θ=1.3: `{"name": "", "age": , "email": ""}` — unparseable; after `"` and the
  digit tokens are penalized on reuse, values come out empty and `"age": ,`
  is not JSON.

## Caveats / deviations

1. **Quantization/precision:** gpt2-large f16, Qwen bf16 (vs HF f32/bf16). The
   A1 gauge no-op gate is unaffected (uniform shift + argmax), and the θ=1.3
   flip rate still lands on the HF value.
2. **Window semantics:** llama.cpp is a sliding window; HF is all-seen. We
   forced equivalence with `penalty_last_n=1024` (valid because prompt+gen <
   1024) and additionally measured the default 64 in A2.
3. **`-1` clamp:** header/CLI docs say `-1 = context size`, implementation
   disables the penalty. Anyone "replicating" with `--repeat-last-n -1` would
   silently measure θ=1.0. Maintainer-relevant by itself.
4. **A2 valid rate 0.167 vs HF 0.0:** 8/48 cells survive θ=1.3 in llama.cpp
   (HF: 0/48). Same qualitative collapse (100% → ~17%); residual difference
   plausibly from bf16-GGUF vs HF numerics and tokenizer pipeline differences
   flipping a few borderline greedy tokens. Direction and magnitude of the
   effect are unambiguous.
5. **EOS handling (A1):** generation deliberately continues past EOG for fixed
   200-token alignment, matching the original A1 protocol.
6. **Converter bug worked around (upstream, incidental):** at this commit
   `conversion/gpt2.py` routes the constant `h.N.attn.bias` causal-mask buffers
   through the tensor mapper, aborting with `Can not map tensor
   'h.0.attn.bias'`. Patched locally (in the gitignored clone) to skip them —
   they are not learned weights. Worth an upstream report on its own.
7. **A1 ran on the CPU backend** for determinism (gpt2-large is small); A2 ran
   fully offloaded on CUDA and was verified deterministic (8/8 identical reps
   per condition).

## Files

- `code/smoke_llamacpp/` — drivers, analysis scripts, README with exact rerun commands (clone/build/venv/models gitignored).
- `results/smoke_llamacpp/a1_raw.json` — 96 generations (16 prompts × 3 θ × 2 c), token ids.
- `results/smoke_llamacpp/a2_raw.json` — 288 generations (6 schemas × 3 θ × 2 windows × 8 reps), text.
- `results/smoke_llamacpp/a1_summary.json`, `a2_summary.json`, `summary.json` — machine-readable numbers.
