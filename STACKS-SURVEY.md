# Survey: which inference stacks carry the CTRL sign-branch repetition penalty

Date: 2026-07-01. Method: source reading at pinned commits (shallow clones / raw GitHub fetches),
plus issue/PR/doc verification. All work delegated to six parallel research agents; every core-form
classification below is backed by a verbatim excerpt read directly from source at the stated commit,
except where explicitly marked UNVERIFIED.

**The operator in question** (CTRL, Keskar et al. 2019, as fixed post-HF-issue-2302): for each
previously-seen token with raw logit `z`, branch on the sign: `z >= 0 → z/θ` (divide),
`z < 0 → z·θ` (multiply). The paper (paper/paper-note.tex) shows this is gauge-dependent (A1:
softmax shift-invariance makes the zero-point arbitrary) and corrupts structured output (A2).
Subtractive presence/frequency penalties (`z − α`) are shift-invariant and immune to A1.

**Classification key**
- (a) CTRL sign-branch on raw logits — subject to A1 and A2
- (b) multiplicative WITHOUT sign branch (the pre-2019 naive divide)
- (c) subtractive — immune to A1
- (d) applied to normalized log-probs — exempt
- (e) parameter accepted but not implemented / silently ignored
- (f) wrapper delegating to another engine (named)
- (g) no repetition penalty

Every implementation surveyed is class (a). Twelve implementations of the multiplicative penalty were read (HF, vLLM, llama.cpp previously; TGI,
SGLang, TensorRT-LLM, ExLlamaV2, mlx-lm, LMDeploy ×2 engines, aphrodite, KoboldCpp, mistral.rs,
candle, text-generation-webui's hijack, Ollama's MLX sampler here) — all sign-branch on the raw
logit. Not one stack normalizes before penalizing. No stack surveyed uses the naive form (b) or
the normalized form (d). The hosted APIs (OpenAI, Anthropic, Google) expose no multiplicative
knob at all — subtractive-only or nothing.

---

## 1. Summary table

| # | Stack | Knob | Form | Seen-set / window | A1? | A2? | Evidence |
|---|-------|------|------|-------------------|-----|-----|----------|
| 0 | HF transformers | `repetition_penalty` | (a) sign-branch, raw logits | full context | yes | yes | verified previously (PRIOR-ART.md; paper) |
| 0 | vLLM | `repetition_penalty` | (a) sign-branch, raw logits | prompt ∪ output | yes | yes | verified previously (paper, `e196268`) |
| 0 | llama.cpp | `repeat_penalty` | (a) sign-branch, raw logits | last-64 window (default) | yes | yes | verified previously (paper, `4fc4ec55`) |
| 1a | Ollama — GGML/GGUF path (vast majority of models) | `repeat_penalty` (default 1.1!) | (f) → llama.cpp `llama-server` subprocess ⇒ (a) | `repeat_last_n` default 64 | yes | yes | [llm/llama_server.go:1482](https://github.com/ollama/ollama/blob/cecd265/llm/llama_server.go) @ `cecd265` |
| 1b | Ollama — MLX path (Go sampler, mac/nvfp4 models) | `repeat_penalty` | (a) sign-branch, raw logits (own Go/MLX code) | ring buffer, `repeat_last_n` 64; `-1` → ctx; `0` → off | yes | yes | [x/mlxrunner/sample/sample.go:888-915](https://github.com/ollama/ollama/blob/cecd265/x/mlxrunner/sample/sample.go) @ `cecd265` |
| 1c | Ollama — old Go GGML engine (ollamarunner) | `repeat_penalty` | (e) accepted-but-ignored — **historical**; engine deleted by PR #16031 (merged 2026-05-29) | n/a | n/a | n/a | [issue #15783](https://github.com/ollama/ollama/issues/15783), [PR #15784](https://github.com/ollama/ollama/pull/15784) (open, orphaned), [PR #16031](https://github.com/ollama/ollama/pull/16031) |
| 2 | HF TGI (server) | `repetition_penalty` | (a) sign-branch, raw logits (Python processor; Rust only forwards) | full seq (prompt+generated), membership | yes | yes | [logits_process.py:108-117](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f/server/text_generation_server/utils/logits_process.py#L108-L117) @ `b4adbf2` |
| 2b | TGI llama.cpp backend | `repetition_penalty` | (f) → llama.cpp | llama.cpp's | yes | yes | `backends/llamacpp/src/backend.rs:183` @ `b4adbf2` |
| 3 | SGLang | `repetition_penalty` | (a) sign-branch, raw logits | **generated output only**, membership (scatter-overwrite) | yes | yes | [repetition_penalty.py:10-16](https://github.com/sgl-project/sglang/blob/926140d7/python/sglang/srt/sampling/penaltylib/repetition_penalty.py#L10-L16) @ `926140d` |
| 4 | TensorRT-LLM | `repetition_penalty` | (a) sign-branch (on temperature-scaled logit; sign preserved ⇒ same branch) | full seq incl. prompt, bitmap membership | yes | yes | [penaltyKernels.cu:199-206](https://github.com/NVIDIA/TensorRT-LLM/blob/46054021/cpp/tensorrt_llm/kernels/penaltyKernels.cu#L199-L206) @ `4605402` |
| 5 | ExLlamaV2 | `token_repetition_penalty` | (a) sign-branch, raw logits (+ additive pres/freq) | `range` (−1 = whole seq) + linear `decay` tail; once/token | yes | yes | [sampling.cpp:85-86](https://github.com/turboderp-org/exllamav2/blob/7dc12af3/exllamav2/exllamav2_ext/cpp/sampling.cpp#L85-L86) @ `7dc12af` |
| 6 | mlx-lm | `repetition_penalty` | (a) sign-branch, raw logits | last `context_size` tokens, **default 20** | yes | yes | [sample_utils.py:300-310](https://github.com/ml-explore/mlx-lm/blob/2ed22318/mlx_lm/sample_utils.py#L300-L310) @ `2ed2231` |
| 7 | LMDeploy TurboMind | `repetition_penalty` | (a) sign-branch, raw logits (CUDA) | full seq bitmask, once/token | yes | yes | [sampling_penalty_kernels.cu:167-171](https://github.com/InternLM/lmdeploy/blob/5868a57f/src/turbomind/kernels/sampling_penalty_kernels.cu#L169-L170) @ `5868a57` |
| 7b | LMDeploy PyTorch engine | `repetition_penalty` | (a) sign-branch, raw logits (gather/where/scatter, HF-identical) | full context ids | yes | yes | [logits_process.py:59-65](https://github.com/InternLM/lmdeploy/blob/5868a57f/lmdeploy/pytorch/engine/logits_process.py#L59-L65) @ `5868a57` |
| 8 | aphrodite-engine | `repetition_penalty` | (a) sign-branch, raw logits (vLLM-inherited, torch + CUDA) | prompt ∪ output, full context | yes | yes | [_custom_ops.py:707-718](https://github.com/aphrodite-engine/aphrodite-engine/blob/14e8de14/aphrodite/_custom_ops.py#L707-L718), [csrc/sampler.cu:36-42](https://github.com/aphrodite-engine/aphrodite-engine/blob/14e8de14/csrc/sampler.cu#L36-L42) @ `14e8de1` |
| 9 | KoboldCpp | `rep_pen` (+ `rep_pen_range`, `rep_pen_slope`) | (a) sign-branch, raw logits — **own code, not llama.cpp's sampler**, in both paths | last `rep_pen_range` tokens; slope = 2-tier bucket (older half gets `1+(θ−1)·slope`) | yes | yes | [gpttype_adapter.cpp:1730-1738](https://github.com/LostRuins/koboldcpp/blob/0c163a9b/gpttype_adapter.cpp#L1662-L1747) @ `0c163a9`, branch `concedo` |
| 10a | mistral.rs | `repetition_penalty` | (a) sign-branch — on logit **after** subtractive freq/pres applied (raw when those are 0) | caller context, `count>0` | yes | yes | [sampler.rs:1230-1243](https://github.com/EricLBuehler/mistral.rs/blob/15986c03/mistralrs-core/src/sampler.rs#L1210-L1246) @ `15986c0` |
| 10b | candle (`apply_repeat_penalty`) | `repeat_penalty` | (a) sign-branch, raw logits (textbook) | HashSet-dedup of last `repeat_last_n` (examples default 128) | yes | yes | [utils.rs:25-46](https://github.com/huggingface/candle/blob/31f35b14/candle-transformers/src/utils.rs#L25-L46) @ `31f35b1` |
| 11a | llama-cpp-python | `repeat_penalty` | (f) → llama.cpp (`llama_sampler_init_penalties`) | `last_n_tokens_size` default 64; default θ=1.0 (off) | yes | yes | `llama_cpp/_internals.py:1060-1067` @ `346853c` |
| 11b | text-generation-webui — llama.cpp loader | `repetition_penalty` | (f) → llama.cpp | `repetition_penalty_range` → `repeat_last_n`, default 1024 | yes | yes | `modules/llama_cpp_server.py:85-86` @ `ed888c7` |
| 11c | text-generation-webui — Transformers & ExLlamav3_HF loaders | `repetition_penalty` | (a) **own** sign-branch (monkeypatches HF's processor with a range-aware clone) | `repetition_penalty_range` default 1024 | yes | yes | `modules/sampler_hijack.py:495` @ `ed888c7` |
| 11d | text-generation-webui — ExLlamav3 native / TensorRT-LLM loaders | `repetition_penalty` | (f) → ExLlamaV3 / TRT-LLM | those engines' | yes | yes | `modules/exllamav3.py:330-331`, `modules/tensorrt_llm.py:40` @ `ed888c7` |
| 11e | LM Studio (closed source) | `repeat_penalty` | (f) → llama.cpp (default engine); MLX engine on Mac → mlx-lm ⇒ (a) both ways | engine defaults | yes | yes | docs only: lmstudio.ai/docs/app (engine-only note; no source) |
| 11f | GPT4All | `repeat_penalty` | (f) → llama.cpp (`llama_sampler_init_penalties`) | `repeat_last_n` 64, default θ=1.1 | yes | yes | `gpt4all-backend/src/llamamodel.cpp` `initSampler` ~L1025-1033 @ main `b666d16` (lines approx.) |
| 11g | Jan | `repeat_penalty` | (f) → llama.cpp (vendored, cortex.llamacpp lineage) — exact file:line UNVERIFIED | llama.cpp's | yes | yes | deepwiki.com/menloresearch/jan/4.2-llamacpp-extension |
| 12a | OpenAI API | `presence_penalty`, `frequency_penalty` (−2..2) | (c) subtractive, formula documented | count/membership | no | no | developers.openai.com/api/docs/guides/advanced-usage |
| 12b | Anthropic API | — | (g) no penalty knobs at all (temp/top_p/top_k only) | n/a | no | no | platform.claude.com/docs/en/api/messages |
| 12c | Google Gemini | `presencePenalty`, `frequencyPenalty` (−2..<2) | (c) subtractive ("applied to logprobs", count-scaled) | count/membership | no | no | Firebase AI Logic model-parameters docs; generative-ai-js SDK reference |

Sign-branch boundary trivia: vLLM/aphrodite/ExLlamaV2/candle test `z > 0` (or `>= 0`) → divide;
HF/TGI/SGLang/TRT-LLM/LMDeploy/mlx-lm/mistral.rs/Ollama-MLX test `z < 0` → multiply; KoboldCpp
tests `z <= 0` → multiply. All identical except at exactly `z = 0`, which is a fixed point of both
branches — one operator everywhere.

---

## 2. Per-stack detail

### 2.1 Ollama (commit `cecd265`, main, 2026-07-01)

The premise "new Go engine ollamarunner
accepts repeat_penalty but never wires it into the sampler" was true until late May 2026. Then
**PR #16031** ("runner: Remove CGO engines, use llama-server exclusively for GGML models",
dhiltgen, **merged 2026-05-29**) deleted the Go-native GGML engine entirely. `sample/samplers.go`
and `runner/ollamarunner/` no longer exist on main (grep confirms zero matches for `ollamarunner`
/ `NewSampler`). All GGML/GGUF models are now served by an upstream **llama.cpp `llama-server`
subprocess**, penalty params forwarded verbatim.

Three paths today:

**(1) GGML/GGUF → llama-server — class (f) ⇒ (a).** `llm/server.go:106` ("All GGML models are
served via the upstream llama-server subprocess."); `llm/llama_server.go:1482-1485`:

```go
RepeatPenalty:   req.Options.RepeatPenalty,
RepeatLastN:     req.Options.RepeatLastN,
FreqPenalty:     req.Options.FrequencyPenalty,
PresPenalty:     req.Options.PresencePenalty,
```

llama.cpp then applies its CTRL sign-branch (verified separately in the paper at `4fc4ec55`).

**(2) MLX safetensors → Go `x/mlxrunner` — class (a), own code.** The one surviving Go-native
sampler replicated the CTRL sign-branch verbatim. `x/mlxrunner/sample/sample.go:888-915`:

```go
factor := mlx.Where(
    adjusted.Less(mlx.FromValue(float32(0))),
    mlx.FromValue(ctx.opts.RepeatPenalty),
    mlx.FromValue(1/ctx.opts.RepeatPenalty),
)
adjusted = adjusted.Multiply(factor)
```

i.e. `z < 0 → z·θ`, else `z·(1/θ)` — applied to raw logits (called from `baseScores` at
`sample.go:818`, pre-softmax). Presence = flat subtract; frequency = count-scaled subtract via
`ScatterAddAxis`. Window: per-sequence ring buffer of width `RepeatLastN`; `0` → penalty off,
`-1` → whole context. Routed via `m.IsMLX()` (`server/sched.go:531`); currently the experimental
mac/nvfp4 models (`laguna-xs.2`, `qwen3.5:2b-nvfp4`, `gemma4:e2b-nvfp4`).

**(3) Image-gen → `x/imagegen` — class (g), N/A.**

**Options plumbing and defaults** — `api/types.go:593-597, 1101-1105`: `RepeatLastN: 64`,
**`RepeatPenalty: 1.1`** (i.e. the penalty is ON by default for every Ollama user),
`PresencePenalty: 0`, `FrequencyPenalty: 0`.

**Issue/PR status (key deliverable):**
- **Issue #15783 — OPEN** (created 2026-04-24): "Go sampler (ollamarunner) silently ignores
  repeat_penalty, frequency_penalty, and presence_penalty." Documents class-(e) behavior of the
  now-deleted engine; reports severe repetition loops (transcription WER 84–93% vs 0–33%).
  References PR #15784 and issue #9278. https://github.com/ollama/ollama/issues/15783
- **PR #15784 — OPEN, unmerged, now orphaned** (created 2026-04-24): "sample: implement repeat,
  frequency, and presence penalties in Go sampler." Its own description confirms it copies the
  CTRL form: repeat_penalty "divides positive logits / multiplies negative logits." It patches
  `sample/samplers.go` + `runner/ollamarunner/runner.go`, **both deleted by #16031**, so it cannot
  merge as written. https://github.com/ollama/ollama/pull/15784
- **PR #16031 — MERGED 2026-05-29**: deleted the Go GGML engine; GGUF inference delegates to
  llama-server. https://github.com/ollama/ollama/pull/16031

Status: the Go GGML sampler no longer exists, so GGUF-path penalties are llama.cpp's. The MLX Go
sampler is new first-party code carrying the same sign-branch; issue #15783 remains open.

UNVERIFIED residue: the exact vendored llama.cpp version pin and its window handling were not
re-read in this pass (params confirmed forwarded verbatim; llama.cpp's own form verified in the
paper). No runtime testing — source reading only.

### 2.2 HF text-generation-inference (TGI) — class (a) [+ (f) for its llama.cpp backend]

Commit `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` (main, 2026-03-21). **No penalty math in Rust**:
`router/src/validation.rs:246-247` validates, `backends/llamacpp/src/backend.rs:183` forwards to
llama.cpp as `penalty_repeat` (that backend = (f)). The server-side sampler is Python —
`server/text_generation_server/utils/logits_process.py:108-117`,
`HeterogeneousRepetitionPenaltyLogitsProcessor`:

```python
score = torch.gather(scores, 1, input_ids)
# if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
score = torch.where(
    score < 0, score * self.penalty_tensor, score / self.penalty_tensor
)
scores.scatter_(1, input_ids, score)
```

Non-batched path (`utils/tokens.py:21,46`) uses upstream transformers'
`RepetitionPenaltyLogitsProcessor` — identical operator. Applied to raw logits before
temperature/top-k warpers (`tokens.py:351` vs `:357`). Seen-set: full sequence (prompt +
generated), membership-based (θ once, not θ^N). A copy exists in the Gaudi backend.

### 2.3 SGLang — class (a)

Commit `926140d789c8c7ec6622a47630b5894b37fbdc1b` (main, 2026-07-02).
`python/sglang/srt/sampling/penaltylib/repetition_penalty.py:10-16`:

```python
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )
```

`BatchedRepetitionPenalizer` (`is_multiplicative = True`) scatter-overwrites θ into seen
positions. **Seen-set: generated output tokens only — the prompt is not penalized** (unlike
HF/vLLM/TGI/TRT-LLM). Membership-based (scatter overwrites, no accumulation). No window.

### 2.4 NVIDIA TensorRT-LLM — class (a)

Commit `46054021101811caaa57a378d45da6ca5bdc8ff1` (main, 2026-07-02).
`cpp/tensorrt_llm/kernels/penaltyKernels.cu:199-206`, combined `batchApplyPenalty` kernel:

```cpp
if (ifPresenceInFullSeq > 0)
{
    // Repetition
    if (repetitionPenalties != nullptr)
    {
        logit = logit < 0.0f ? logit * repetitionPenalty : logit / repetitionPenalty;
    }
}
```

**Ordering nuance:** temperature is applied first (`logit *= invTemperature`, line 193), so the
sign test runs on the temperature-scaled logit — but temperature > 0 preserves sign, so it is
equivalent to sign-branching on the raw logit. Presence (`-= presencePenalty`, L213) and frequency
(`-= frequencyPenalty * numOccurences`, L218) are separate subtractive ops after. Seen-set: full
sequence (prompt bitmap + generated occurrence counts, `penaltyWorkspace`), membership-based;
optional `promptIgnoreLengths` can exclude part of the prompt. No sliding window.

### 2.5 ExLlamaV2 — class (a)

Commit `7dc12af3a81f34ac3f27cd7602ed539b638933ca` (master, 2026-03-04).
`exllamav2/exllamav2_ext/cpp/sampling.cpp:85-88` (CPU-only C++; there is no CUDA copy — full-history
search of `exllamav2_ext/cuda/` finds none, correcting an earlier note in this survey. Comment at
line 48 says "as in HF repetition penalty"):

```cpp
if (logits[t] > 0.0) logits[t] /= rep_p;  // Multiplicative penalty
else logits[t] *= rep_p;
logits[t] -= pres_p;  // Additive penalty
```

Plus per-occurrence `-= freq_p` (line 95). Window: iterates backwards over
`sustain (= token_repetition_range; −1 = whole seq)` + `decay` tokens; over the decay tail θ ramps
linearly back to 1.0 (`d_rep_p = (1 − rep_p)/decay`). Multiplicative + presence hit each token
once (`g_rep_mask`); frequency accumulates. Wiring: `exllamav2/generator/sampler.py:412-418`.

### 2.6 mlx-lm — class (a)

Commit `2ed22318cd6a2fcc5c2e0caa1e1fb0ddeb7cafd5` (main, 2026-06-24).
`mlx_lm/sample_utils.py:300-310`:

```python
tokens = tokens[-context_size:]
selected_logits = logits[:, tokens]
selected_logits = mx.where(
    selected_logits < 0,
    selected_logits * penalty,
    selected_logits / penalty,
)
logits[:, tokens] = selected_logits
```

Docstring calls it "A (sign-aware) multiplicative penalty" and cites CTRL (arXiv:1909.05858)
explicitly. Window: last `context_size` tokens, **default 20** — by far the smallest default
window in the survey. Separate additive presence/frequency processors also exist (lines 315+).
(Note: an earlier second-hand characterization of mlx-lm as non-sign-branched, picked up while
surveying LM Studio, is wrong — this is the direct source read.)

### 2.7 LMDeploy — class (a) in both engines

Commit `5868a57f19162f1973dc05a01975c2b94a48480f` (main, 2026-07-01).

TurboMind CUDA — `src/turbomind/kernels/sampling_penalty_kernels.cu:167-171`:

```cpp
if (masks[di / 32] & (1U << (di % 32))) {
    const float logit = logits[di];
    logits[di]        = logit < 0.f ? logit * penalty : logit / penalty;
}
```

Seen-set: shared-memory bitmask over the entire sequence (prompt + generated), once per token.

PyTorch engine — `lmdeploy/pytorch/engine/logits_process.py:59-65` — a byte-for-byte HF pattern:
`torch.gather` / `torch.where(score < 0, score * penalty, score / penalty)` / `scatter_`. Full
context ids.

### 2.8 aphrodite-engine — class (a), vLLM-inherited

Commit `14e8de147ac71fc0af72a64ff7f05770d67691bf` (2026-05-07). Torch path
`aphrodite/_custom_ops.py:707-718`:

```python
penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
# If logits are positive, divide by penalty, otherwise multiply by penalty.
scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
logits *= scaling
```

CUDA kernel `csrc/sampler.cu:36-42` identical (`logit > 0 ? logit/penalty : logit*penalty`).
Additive freq/presence applied alongside in `layers/utils.py:81-82`. Seen-set:
`prompt_mask | output_mask` — full context, no window. Structure inherited from vLLM.

### 2.9 KoboldCpp — class (a), own implementation (not llama.cpp's sampler)

Commit `0c163a9b4c0cecdad508576d96bb5f03b71e2c8a` (branch `concedo`, 2026-06-28). Although built
on llama.cpp, KoboldCpp does **not** use `llama_sampler_init_penalties` — it runs its own rep-pen
in both code paths (single-sequence `sample_rep_pen()` via `SampleLogits`, and a custom
`llama_sampler_i` `batch_rep_pen_init` plugged into the batched chain at `gpttype_adapter.cpp:4261`;
identical math). Core — `gpttype_adapter.cpp:1730-1738`:

```cpp
// The academic publication that described this technique actually just only divided, but that
// would cause tokens with negative logits to become more likely, which is obviously wrong.
// This is common fix for this problem, which is to multiply by the penalty instead of dividing.
if (candidates->data[i].logit <= 0) {
    candidates->data[i].logit *= penalty;
} else {
    candidates->data[i].logit /= penalty;
}
candidates->data[i].logit -= presence_penalty;
```

The comment documents, in shipped source, the understanding that the sign-branch is the fix for
CTRL's naive divide. Slope/range semantics (this commit): window
`last_n_repeat = min(history, rep_pen_range, n_ctx)`; the window is split in half; the recent half
gets full `rep_pen`, the older half gets `rep_pen_reduced = 1 + (rep_pen − 1)·rep_pen_slope`
(a 2-tier bucket, not a smooth ramp); slope clamped to (0,1].

### 2.10 mistral.rs — class (a) (with an ordering quirk)

Commit `15986c037bbe3ee31085d1c73abd2ea3cb11f094` (2026-06-28).
`mistralrs-core/src/sampler.rs:1230-1243` (CPU reference path):

```rust
*logit = *logit
    - count * frequency_penalty
    - if count > 0.0 { 1. } else { 0. } * presence_penalty;

if repetition_penalty != 1.0 && count > 0.0 {
    if *logit > 0.0 {
        *logit /= repetition_penalty;
    } else {
        *logit *= repetition_penalty;
    }
}
```

Sign-branch on raw logits when freq/pres are 0 (the default). **Quirk:** when subtractive
penalties are nonzero they are applied first, so the sign test reads the already-shifted logit —
the operators do not commute, and the subtractive penalty moves tokens across the kink. GPU paths
(`cuda_apply_sparse_penalties_f32`, metal equivalent) UNVERIFIED at kernel level but gated on the
same parameter.

### 2.11 candle — class (a), textbook

Commit `31f35b147389700ed2a178ee66a91c3cc25cc80d` (2026-06-26).
`candle-transformers/src/utils.rs:25-46`, `apply_repeat_penalty`:

```rust
if *logit >= 0. {
    *logit /= penalty
} else {
    *logit *= penalty
}
```

HashSet-dedup (once per token). Examples pass `&tokens[len − repeat_last_n ..]`; llama example
default `repeat_last_n = 128` (`candle-examples/examples/llama/main.rs:120-121, 257-266`).

### 2.12 Wrappers

- **llama-cpp-python** — (f) → llama.cpp. `llama_cpp/_internals.py:1060-1067` calls
  `llama_cpp.llama_sampler_init_penalties(...)` directly; the legacy Python
  `LlamaSamplingContext.sample` raises `NotImplementedError` (dead code). Defaults:
  `repeat_penalty = 1.0` (off), window `last_n_tokens_size = 64`. Commit `346853c`.
- **text-generation-webui** (commit `ed888c7`) — mixed:
  - llama.cpp loader → (f): `modules/llama_cpp_server.py:85-86` maps `repetition_penalty` →
    `repeat_penalty` and `repetition_penalty_range` → `repeat_last_n` (default range 1024).
  - **Transformers & ExLlamav3_HF loaders → (a), TGW's own code**: `sampler_hijack.py:574-580`
    monkeypatches out HF's processor and substitutes `RepetitionPenaltyLogitsProcessorWithRange`
    whose core is `torch.where(score < 0, score * self.penalty, score / self.penalty)`
    (`sampler_hijack.py:495`). Same operator, independent copy, plus a range window HF lacks.
  - ExLlamav3 native loader → (f) to ExLlamaV3 (`modules/exllamav3.py:330-331`, `SS_RepP` stage);
    TensorRT-LLM loader → (f) to TRT-LLM (`modules/tensorrt_llm.py:40`).
  - (The old ExLlamaV2 loaders are gone from current main.)
- **LM Studio** — closed source; engine-only classification. Default engine is llama.cpp ⇒ (a)
  inherited; on Apple silicon the MLX engine uses mlx-lm ⇒ also (a) (mlx-lm verified above,
  §2.6). No first-party penalty code readable. Docs: lmstudio.ai/docs/app.
- **GPT4All** — (f) → llama.cpp: `gpt4all-backend/src/llamamodel.cpp` `initSampler` (~L1025-1033,
  read via GitHub blob, line numbers approximate) adds `llama_sampler_init_penalties` to the
  chain with `repeat_last_n`/`repeat_penalty`. Public defaults θ=1.1, window 64. Repo main
  `b666d16`.
- **Jan** — (f) → llama.cpp (bundled engine, cortex.llamacpp lineage folded into
  menloresearch/llama.cpp). Delegation well-established; exact file:line UNVERIFIED (vendored C++
  not opened).

### 2.13 Hosted APIs — subtractive-only or nothing (verified July 2026)

- **OpenAI** — `presence_penalty`, `frequency_penalty`, both −2.0..2.0; **no multiplicative
  repetition_penalty exists**. Formula documented (their "Frequency and presence penalties"
  guide): `mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence` — a pure
  logit subtraction, shift-invariant, immune to A1. Doc:
  developers.openai.com/api/docs/guides/advanced-usage (the exact code block did not extract
  cleanly through automated fetch; formula is OpenAI's long-standing documented form — for a
  screenshot-grade citation, capture the page in a browser).
- **Anthropic** — Messages API exposes **no penalty parameters at all** (only `max_tokens`,
  `temperature`, `top_p`, `top_k`). Doc: platform.claude.com/docs/en/api/messages (note:
  docs.anthropic.com redirects there now).
- **Google Gemini** — `presencePenalty`, `frequencyPenalty` (−2.0 to <2.0), subtractive
  ("Frequency penalty applied to the next token's logprobs, multiplied by the number of times
  each token has been seen in the respponse so far" — typo in original, generative-ai-js SDK
  reference); **no multiplicative knob**. No full symbolic formula published (prose only). Docs:
  Firebase AI Logic model-parameters page; generative-ai-js GenerationConfig reference.

---

## 3. Cross-stack observations

1. **Ollama.** PR #16031 (merged 2026-05-29) deleted the Go GGML engine; GGUF inference now
   delegates to llama.cpp's `llama-server`. PR #15784 (sign-branch penalties for the deleted
   engine) is open but orphaned; issue #15783 is still open. The MLX Go sampler
   (`x/mlxrunner/sample/sample.go:888-915`) is first-party code carrying the same sign-branch.
   Ollama defaults `repeat_penalty` to 1.1, so the penalty is on by default (also on by default:
   ExLlamaV2 1.025, candle's examples 1.1, GPT4All 1.1; llama-cpp-python defaults 1.0/off).

2. **No stack normalizes first.** Twelve implementations (different companies, languages, and
   hardware targets: Python, Go, Rust, C++, CUDA, Metal/MLX) reproduce the identical sign-branch
   on raw logits, several with comments citing HF or CTRL as the authority (ExLlamaV2: "as in HF
   repetition penalty"; mlx-lm cites arXiv:1909.05858; KoboldCpp's comment narrates the
   naive-divide bug and calls the sign-branch "the common fix"). Each engine keeps its own copy,
   so no single upstream patch reaches them all; STACKS-HISTORY.md traces how each copy arrived.

3. **Seen-set semantics vary while the operator is uniform.** Same knob name, same nominal θ, but: SGLang
   penalizes generated tokens only (prompt exempt); HF/TGI/vLLM/TRT-LLM/LMDeploy penalize prompt
   + output; llama.cpp/Ollama window 64; candle examples 128; TGW default range 1024; mlx-lm
   default **20**; ExLlamaV2 adds a linear decay tail; KoboldCpp a two-tier slope. A user moving
   `repetition_penalty=1.3` between stacks changes both the operator's gauge exposure and which
   tokens it hits.

4. **Two ordering issues, found incidentally.** (i) mistral.rs applies subtractive
   freq/pres penalties BEFORE the sign test (`sampler.rs:1230-1243`), so with nonzero
   freq/pres the multiplicative branch reads an already-shifted logit — the subtractive knob
   moves tokens across the kink, a concrete demonstration that the operators don't commute.
   (ii) TensorRT-LLM applies temperature before the sign test (`penaltyKernels.cu:193` vs
   `:204`) — benign today (temperature > 0 preserves sign) but the branch is one refactor away
   from reading a sign-corrupted value.

5. **TGW's monkeypatch is a fourth independent copy in the HF ecosystem.** text-generation-webui
   replaces HF's processor with its own range-aware sign-branch clone (`sampler_hijack.py:495`),
   so fixing HF upstream does NOT fix TGW's Transformers loader — it needs its own patch. Same
   for TGI, whose batched processor is a separate copy of the HF pattern
   (`logits_process.py:108-117`), and LMDeploy's PyTorch engine (another verbatim clone). The
   gather/where/scatter three-liner has been copy-pasted at least four times.

6. **Hosted APIs.** OpenAI and Google expose subtractive penalties only (formula documented by
   OpenAI); Anthropic exposes none. The multiplicative knob is specific to open-weights and
   self-hosted stacks.

## 4. Verification caveats

Directly read from pinned source (high confidence): TGI, SGLang, TensorRT-LLM, ExLlamaV2, mlx-lm,
LMDeploy (both engines), aphrodite (torch + CUDA), KoboldCpp, mistral.rs (CPU path), candle,
llama-cpp-python, text-generation-webui, Ollama (all three current paths + issue/PR status).

UNVERIFIED / caveats:
- mistral.rs GPU kernels (CUDA/Metal sparse-penalty kernels not opened; same param gate).
- GPT4All: delegation to `llama_sampler_init_penalties` verified via GitHub blob but line numbers
  approximate; quoted signature may lag the current llama.cpp API.
- Jan: llama.cpp delegation established from architecture docs, not a file:line read.
- LM Studio: closed source; engine-inheritance only.
- Ollama: vendored llama.cpp version pin / window handling not re-read (forwarding verified).
- OpenAI formula: the mu[j] code block resisted clean automated extraction from the live page;
  formula quoted is OpenAI's documented form; capture the page in a browser before citing it.
- All classifications are from source reading at one pinned commit per repo; no binaries were run.
