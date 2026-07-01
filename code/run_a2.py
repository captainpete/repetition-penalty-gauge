#!/usr/bin/env python3
"""A2 — repetition penalty as a threshold-sharp attack on structured output. See PREREG.md.

Greedy-decodes JSON / Python / prose continuations under a theta sweep, instrumenting every
step: pre-penalty top1 logit z_top, runner-up gap, the HF penalty's post-penalty argmax, and
whether the flip matches the closed-form threshold g < z_top(1-1/theta). Writes runs/A2/raw.json.

Example:
  python code/run_a2.py
CPU smoke (uses gpt2, not a code model — validates plumbing only):
  python code/run_a2.py --model gpt2 --device cpu --max-new 12 --limit 3
"""
import os, json, argparse
os.environ.setdefault("HF_HUB_CACHE", "/hf/hub")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JSON_PROMPTS = [
    '[\n  {"id": 1, "name": "Alice", "roles": ["admin", "user"], "active": true},\n  {"id": 2, "name": "Bob", "roles": [',
    '{\n  "config": {\n    "server": {"host": "localhost", "port": 8080, "options": {',
    '{\n  "users": [\n    {"name": "x", "tags": ["a", "b"]},\n',
    '{"matrix": [[1, 2, 3], [4, 5, 6], [',
    '{\n  "menu": {\n    "items": [\n      {"label": "File", "submenu": [',
    '[{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x":',
]
PY_PROMPTS = [
    'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [',
    'class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        ',
    'def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        ',
    'def merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        ',
    'import json\n\ndef load_config(path):\n    with open(path) as f:\n        data = json.load(f)\n    return {',
    'def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        ',
]
PROSE_PROMPTS = [
    "The history of the Roman Empire began with the founding of the city of Rome. Over the centuries,",
    "She walked into the room and immediately noticed that something was different. The furniture",
    "The most important lesson I learned that year was about patience. When I first started,",
    "Climate change is one of the defining challenges of our time. Scientists around the world",
    "In the early morning light, the harbor was quiet. A few fishermen",
    "The recipe had been in her family for generations. It called for simple ingredients, but",
]
PROMPTS = ([("json", p) for p in JSON_PROMPTS]
           + [("python", p) for p in PY_PROMPTS]
           + [("prose", p) for p in PROSE_PROMPTS])

STRUCT_CHARS = set('{}[]()"\',:;=')


def is_structural(s):
    t = s.strip()
    if t == "":
        return True  # whitespace / indent / newline
    return all(ch in STRUCT_CHARS for ch in t)


FIX = False  # set by --fix: penalize log_softmax(logits) (normalize-before-penalize)


def apply_rep_penalty(logits, seen_idx, theta):
    if theta == 1.0 or seen_idx.numel() == 0:
        return logits
    out = logits.clone()
    v = out[seen_idx]
    out[seen_idx] = torch.where(v < 0, v * theta, v / theta)
    return out


@torch.no_grad()
def generate(model, tok, prompt_ids, theta, max_new, device):
    seen = set(prompt_ids)
    seen_idx = torch.tensor(sorted(seen), device=device, dtype=torch.long)
    positions = []
    gen = []
    cur = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    past = None
    for _ in range(max_new):
        feed = cur if past is None else cur[:, -1:]
        out = model(feed, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()                 # pre-penalty, upcast to fp32
        # top1 via argmax (the SAME op as `emitted`) so θ=1.0 gives 0 flips exactly —
        # topk vs argmax disagree on bf16 ties and manufacture spurious flips otherwise.
        top1 = int(torch.argmax(logits))
        z_top = float(logits[top1])
        masked = logits.clone()
        masked[top1] = float("-inf")
        runnerup = int(torch.argmax(masked))
        z_2 = float(logits[runnerup])
        plog = apply_rep_penalty(torch.log_softmax(logits, dim=-1) if FIX else logits, seen_idx, theta)  # HF semantics (FIX: normalize first)
        emitted = int(torch.argmax(plog))
        top1_str = tok.decode([top1])
        positions.append({
            "top1": top1, "z_top": z_top, "runnerup": runnerup, "z2": z_2,
            "gap": z_top - z_2,
            "top1_seen": top1 in seen, "top1_pos": z_top > 0,
            "top2_seen": runnerup in seen,
            "structural": is_structural(top1_str), "top1_str": top1_str,
            "emitted": emitted, "flip": emitted != top1,
            "emitted_is_runnerup": emitted == runnerup,
        })
        if emitted not in seen:
            seen.add(emitted)
            seen_idx = torch.cat([seen_idx, torch.tensor([emitted], device=device)])
        gen.append(emitted)
        cur = torch.cat([cur, torch.tensor([[emitted]], device=device)], dim=1)
    return positions, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bigcode/starcoder2-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts per domain (smoke)")
    ap.add_argument("--thetas", default="1.0,1.1,1.2,1.3,1.4,1.5")
    ap.add_argument("--out", default="runs/A2/raw.json")
    ap.add_argument("--fix", action="store_true", help="normalize (log_softmax) before penalizing")
    args = ap.parse_args()
    global FIX; FIX = args.fix

    thetas = [float(x) for x in args.thetas.split(",")]
    if args.limit:
        sel = []
        for dom in ("json", "python", "prose"):
            sel += [(d, p) for (d, p) in PROMPTS if d == dom][:args.limit]
        prompts = sel
    else:
        prompts = PROMPTS

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"

    records = []
    for pi, (dom, ptext) in enumerate(prompts):
        pids = tok(ptext)["input_ids"]
        for theta in thetas:
            positions, gen = generate(model, tok, pids, theta, args.max_new, args.device)
            records.append({
                "prompt_idx": pi, "domain": dom, "theta": theta,
                "prompt": ptext, "gen_text": tok.decode(gen),
                "positions": positions,
            })
        print(f"  prompt {pi + 1}/{len(prompts)} ({dom}) done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "revision": rev, "max_new": args.max_new,
               "thetas": thetas, "records": records}, open(args.out, "w"))
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
