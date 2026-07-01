#!/usr/bin/env python3
"""A1 — Repetition-penalty gauge (non-)invariance. Generation + metrics. See PREREG.md.

Adds a constant gauge shift c to every output logit (== lm_head.bias += c) BEFORE the
repetition penalty, which uses exact HF semantics (positive logits /theta, negative
*theta, over all previously-seen ids). Softmax is shift-invariant so c is a provable
no-op at theta=1; the sign-branched penalty makes the same theta>1 act differently
under different c. Writes runs/A1/raw.json.

Example:
  python code/run_a1.py
CPU smoke test:
  python code/run_a1.py --device cpu --max-new 8 --limit 2
"""
import os, json, argparse
os.environ.setdefault("HF_HUB_CACHE", "/hf/hub")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


FIX = False  # set by --fix: apply penalty to log_softmax(logits) (normalize-before-penalize)


def apply_rep_penalty(logits, seen_idx, theta):
    """HF RepetitionPenaltyLogitsProcessor semantics on the seen-id set."""
    if theta == 1.0 or seen_idx.numel() == 0:
        return logits
    logits = logits.clone()
    vals = logits[seen_idx]
    vals = torch.where(vals < 0, vals * theta, vals / theta)
    logits[seen_idx] = vals
    return logits


@torch.no_grad()
def generate(model, prompt_ids, c, theta, max_new, mode, seed, device):
    seen = set(prompt_ids)
    seen_idx = torch.tensor(sorted(seen), device=device, dtype=torch.long)
    gen, ents = [], []
    g = None
    if mode == "sample":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    cur = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    past = None
    for step in range(max_new):
        feed = cur if past is None else cur[:, -1:]
        out = model(feed, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        logits = logits + c                                  # gauge shift (== lm_head.bias += c)
        if FIX:
            logits = torch.log_softmax(logits, dim=-1)       # PROPOSED FIX: normalize before penalizing
        logits = apply_rep_penalty(logits, seen_idx, theta)  # penalty AFTER shift
        p = torch.softmax(logits, dim=-1)
        ents.append(float(-(p * torch.log(p + 1e-12)).sum()))
        if mode == "greedy":
            nxt = int(torch.argmax(logits))
        else:
            nxt = int(torch.multinomial(p, 1, generator=g))
        if nxt not in seen:
            seen.add(nxt)
            seen_idx = torch.cat([seen_idx, torch.tensor([nxt], device=device)])
        gen.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
    return gen, sum(ents) / len(ents)


def distinct_n(ids, n):
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def rep_rate(prompt_ids, gen_ids):
    seen, rep = set(prompt_ids), 0
    for t in gen_ids:
        if t in seen:
            rep += 1
        seen.add(t)
    return rep / max(1, len(gen_ids))


def max_run(ids):
    best = cur = 1 if ids else 0
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="use only first N prompts (smoke test)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="runs/A1/raw.json")
    ap.add_argument("--thetas", default="1.0,1.15,1.3")
    ap.add_argument("--cs", default="-5,-3,-1,0,1,3,5")
    ap.add_argument("--fix", action="store_true", help="normalize (log_softmax) before penalizing")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="bfloat16 to fit a 7B on 24GB; the gauge-shift + argmax decision stays fp32 "
                         "(generate() casts logits to float()), so the no-op gate is unaffected")
    args = ap.parse_args()

    global FIX; FIX = args.fix
    thetas = [float(x) for x in args.thetas.split(",")]
    cs = [float(x) for x in args.cs.split(",")]
    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"

    records = []
    for pi, ptext in enumerate(prompts):
        pids = tok(ptext)["input_ids"]
        for mode in ("greedy", "sample"):
            for theta in thetas:
                for c in cs:
                    gen, mean_ent = generate(model, pids, c, theta, args.max_new,
                                             mode, args.seed, args.device)
                    records.append({
                        "prompt_idx": pi, "mode": mode, "theta": theta, "c": c,
                        "gen_ids": gen,
                        "distinct1": distinct_n(gen, 1),
                        "distinct2": distinct_n(gen, 2),
                        "distinct3": distinct_n(gen, 3),
                        "rep_rate": rep_rate(pids, gen),
                        "mean_entropy": mean_ent,
                        "max_run": max_run(gen),
                        "text": tok.decode(gen),
                    })
        print(f"  prompt {pi + 1}/{len(prompts)} done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "revision": rev, "seed": args.seed,
               "max_new": args.max_new, "thetas": thetas, "cs": cs,
               "n_prompts": len(prompts), "records": records},
              open(args.out, "w"))
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
