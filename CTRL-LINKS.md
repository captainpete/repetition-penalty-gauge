# Where the CTRL sign-branch lives: verified source permalinks

Commit-pinned GitHub links to the current source of the multiplicative (CTRL) repetition-penalty
sign-branch in every stack that carries its own copy. Each was verified by resolving the repo's
default-branch HEAD (`git ls-remote`) and confirming the operator at the pinned line range against
the raw file (`grep`/`sed`), not GitHub's summarized views. HEADs as of **2026-07-07**. Companion
to `STACKS-SURVEY.md` (what/how) and `STACKS-HISTORY.md` (the git genealogy of how each got it).

The operator: for a previously-seen token with raw logit `z`, divide if positive (`z/θ`), multiply
if negative (`z·θ`) — a sign-branch on the un-normalized logit.

## Own-implementation stacks

| Stack | Permalink |
|---|---|
| HuggingFace transformers | https://github.com/huggingface/transformers/blob/eee480d59810135f45280f8db99f14d0136bed82/src/transformers/generation/logits_process.py#L409-L412 |
| vLLM (torch) | https://github.com/vllm-project/vllm/blob/5769a7382cb111288a9c2927342ca908943b9793/vllm/_custom_ops.py#L278-L291 |
| vLLM (CUDA) | https://github.com/vllm-project/vllm/blob/5769a7382cb111288a9c2927342ca908943b9793/csrc/libtorch_stable/sampler.cu#L36-L41 |
| llama.cpp | https://github.com/ggml-org/llama.cpp/blob/ee445f93d8a0a5033a46d1960e901ef5caec9a41/src/llama-sampler.cpp#L2688-L2694 |
| Ollama (MLX sampler) | https://github.com/ollama/ollama/blob/67b6a1c2d45321e0cb3c04a18073f9818de7724b/x/mlxrunner/sample/sample.go#L897-L902 |
| HF text-generation-inference | https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/utils/logits_process.py#L109-L116 |
| SGLang | https://github.com/sgl-project/sglang/blob/6c1fb8a937dfd135996a024e97b1a4b23202f570/python/sglang/srt/sampling/penaltylib/repetition_penalty.py#L10-L15 |
| NVIDIA TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM/blob/a0c406ff88c4a9736b5ce2f3c5eacbacdd0926d1/cpp/tensorrt_llm/kernels/penaltyKernels.cu#L202-L205 |
| ExLlamaV2 (CPU only; no CUDA copy) | https://github.com/turboderp-org/exllamav2/blob/7dc12af3a81f34ac3f27cd7602ed539b638933ca/exllamav2/exllamav2_ext/cpp/sampling.cpp#L85-L86 |
| mlx-lm | https://github.com/ml-explore/mlx-lm/blob/2ed22318cd6a2fcc5c2e0caa1e1fb0ddeb7cafd5/mlx_lm/sample_utils.py#L304-L308 |
| LMDeploy (PyTorch) | https://github.com/InternLM/lmdeploy/blob/99c215fb005263031198824ad83bcd902c9d05ec/lmdeploy/pytorch/engine/logits_process.py#L61-L64 |
| LMDeploy (TurboMind CUDA) | https://github.com/InternLM/lmdeploy/blob/99c215fb005263031198824ad83bcd902c9d05ec/src/turbomind/kernels/sampling_penalty_kernels.cu#L167-L171 |
| aphrodite-engine (torch) | https://github.com/aphrodite-engine/aphrodite-engine/blob/03d018b952dfc3a523cfed8a2063a9cf2ff45fa5/aphrodite/_custom_ops.py#L526-L537 |
| aphrodite-engine (CUDA) | https://github.com/aphrodite-engine/aphrodite-engine/blob/03d018b952dfc3a523cfed8a2063a9cf2ff45fa5/csrc/libtorch_stable/sampler.cu#L36-L41 |
| KoboldCpp (branch `concedo`) | https://github.com/LostRuins/koboldcpp/blob/0c163a9b4c0cecdad508576d96bb5f03b71e2c8a/gpttype_adapter.cpp#L1730-L1737 |
| mistral.rs | https://github.com/EricLBuehler/mistral.rs/blob/4f6042b41ef56dd9d22360b39a8d381ac070700d/mistralrs-core/src/sampler.rs#L1236-L1242 |
| candle | https://github.com/huggingface/candle/blob/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-transformers/src/utils.rs#L25-L44 |
| text-generation-webui | https://github.com/oobabooga/text-generation-webui/blob/ed888c71f221df552750e1834b3654abab8ae345/modules/sampler_hijack.py#L492-L496 |

## Wrappers (no own copy — they run llama.cpp's operator above)

llama-cpp-python, GPT4All, Jan, LM Studio (default engine), TGI's llama.cpp backend, Ollama's GGUF
path, and text-generation-webui's llama.cpp loader all delegate to `llama_sampler_init_penalties`,
i.e. the **llama.cpp** row. Hosted APIs (OpenAI, Google) expose only subtractive
presence/frequency penalties; Anthropic exposes none. None of these carry the multiplicative
sign-branch.
