#!/usr/bin/env python3
"""A1 gauge-(non)invariance probe, measured INSIDE vLLM's own sampler.

Design mirrors ../run_a1.py: greedy decode; add a constant gauge shift c to
EVERY output logit BEFORE vLLM's repetition penalty is applied; compare c=+5 vs
c=-5 token-by-token. Softmax is shift-invariant, so at theta=1 the two runs are
provably identical (validity gate: flip rate MUST be exactly 0). The sign-branched
multiplicative penalty (positive/theta, negative*theta on RAW logits) makes the
same theta>1 act differently under different c -> flips.

Route: vLLM 0.8.5 V1 sampler REJECTS per-request logits_processors
(v1/engine/processor.py raises ValueError). V0 applies them in
LogitsProcessor.forward (model_executor/layers/logits_processor.py:83) BEFORE the
Sampler runs apply_penalties (model_executor/layers/sampler.py:262). So we FORCE
V0 (VLLM_USE_V1=0) and inject the shift as a per-request logits_processor, which
provably precedes the penalty. Verified by reading installed source; see REPORT.md.

Run:
  .venv/bin/python run_a1_vllm.py --model openai-community/gpt2-large
"""
import os, json, argparse
os.environ["VLLM_USE_V1"] = "0"          # V0 engine: needed for per-request logits_processors
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams

PROMPTS = [
    "The history of the Roman Empire began",
    "Here is a list of my favorite things:",
    "My morning routine is simple. First,",
    "The most important rule of cooking is",
    "In the small town where I grew up,",
    "To whom it may concern, I am writing to",
    "The weather today is",
    "She opened the door and saw",
    "The best advice I ever received was",
    "Once upon a time, in a land far away,",
    "The instructions were clear:",
    "Breaking news this morning:",
    "I have always believed that",
    "The recipe calls for the following ingredients:",
    "After many years of research, scientists have",
    "Dear diary, today was",
]


def make_shift(c):
    # V0 calls a 2-param processor as (past_output_token_ids, logits_row).
    # We add a constant to the whole vocab row BEFORE penalties -> softmax no-op at theta=1.
    def _shift(past_ids, logits):
        return logits + c
    return _shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2-large")
    ap.add_argument("--dtype", default="auto")      # 'auto' -> fp16 for gpt2, bf16 forced for 7B via flag
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--thetas", default="1.0,1.15,1.3")
    ap.add_argument("--cs", default="5,-5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="../../results/smoke_vllm/a1_raw.json")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=0, help="cap ctx (0=model default)")
    args = ap.parse_args()

    thetas = [float(x) for x in args.thetas.split(",")]
    cs = [float(x) for x in args.cs.split(",")]
    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS

    llm_kwargs = dict(model=args.model, dtype=args.dtype, enforce_eager=True,
                      gpu_memory_utilization=args.gpu_mem_frac, seed=0)
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    mc = llm.llm_engine.model_config
    dtype = str(mc.dtype)
    rev = getattr(mc, "revision", None)

    # Build one flat request list: (theta, c, prompt_idx) -> SamplingParams with its own shift.
    reqs, meta = [], []
    for theta in thetas:
        for c in cs:
            sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                ignore_eos=True, repetition_penalty=theta,
                                logits_processors=[make_shift(c)])
            for pi, ptext in enumerate(prompts):
                reqs.append((ptext, sp))
                meta.append({"theta": theta, "c": c, "prompt_idx": pi})

    outs = llm.generate([r[0] for r in reqs], [r[1] for r in reqs])
    records = []
    for m, o in zip(meta, outs):
        toks = list(o.outputs[0].token_ids)
        records.append({**m, "token_ids": toks, "n_tokens": len(toks),
                        "text": o.outputs[0].text})

    # Flip rate: for each theta, align c=+5 vs c=-5 per prompt, compare token ids.
    by = {}
    for r in records:
        by[(r["theta"], r["c"], r["prompt_idx"])] = r["token_ids"]
    cpos, cneg = cs[0], cs[1]
    summary = {}
    for theta in thetas:
        flips = total = 0
        per_prompt = []
        for pi in range(len(prompts)):
            a = by[(theta, cpos, pi)]
            b = by[(theta, cneg, pi)]
            n = min(len(a), len(b))
            f = sum(1 for i in range(n) if a[i] != b[i])
            flips += f; total += n
            per_prompt.append({"prompt_idx": pi, "aligned": n, "flips": f})
        summary[str(theta)] = {"flips": flips, "positions": total,
                               "flip_rate": (flips / total) if total else 0.0,
                               "per_prompt": per_prompt}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"model": args.model, "dtype": dtype, "revision": rev,
               "engine": "V0", "max_tokens": args.max_tokens,
               "thetas": thetas, "cs": cs, "n_prompts": len(prompts),
               "shift_route": "per-request logits_processors (V0), applied before penalties",
               "seen_set": "vLLM repetition_penalty penalizes prompt+output tokens",
               "summary": summary, "records": records},
              open(args.out, "w"))
    print(f"\n=== A1 vLLM ({args.model}, dtype={dtype}) ===")
    for theta in thetas:
        s = summary[str(theta)]
        print(f"  theta={theta}: flip_rate={s['flip_rate']:.4f} "
              f"({s['flips']}/{s['positions']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
