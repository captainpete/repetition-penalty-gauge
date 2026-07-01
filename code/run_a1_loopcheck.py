#!/usr/bin/env python3
"""A1 (rep-penalty) — does the FIXED operator still suppress degenerate loops?/repetition-penalty.md. R3-W4 objection: the normalize-before-penalize fix might merely weaken
the penalty toward a no-op. This sweeps theta under {raw, fix} on the loop-prone gpt2 and reports a
degeneration metric (repetition rate / distinct-2 / longest single-token run). If under the fix the
repetition rate still falls monotonically as theta rises, the fix is a usable penalty, not a disabled one.

  python code/run_a1_loopcheck.py --model gpt2
"""
import os, json, argparse
os.environ.setdefault("HF_HUB_CACHE", "/hf/hub")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_a1 import PROMPTS, apply_rep_penalty


@torch.no_grad()
def greedy(model, prompt_ids, theta, max_new, device, fix):
    seen = set(prompt_ids); seen_idx = torch.tensor(sorted(seen), device=device, dtype=torch.long)
    cur = torch.tensor([prompt_ids], device=device, dtype=torch.long); past = None; gen = []
    for _ in range(max_new):
        feed = cur if past is None else cur[:, -1:]
        out = model(feed, past_key_values=past, use_cache=True); past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        if fix:
            logits = torch.log_softmax(logits, dim=-1)         # normalize before penalizing
        logits = apply_rep_penalty(logits, seen_idx, theta)
        tok = int(logits.argmax()); gen.append(tok)
        if tok not in seen:
            seen.add(tok); seen_idx = torch.cat([seen_idx, torch.tensor([tok], device=device)])
        cur = torch.cat([cur, torch.tensor([[tok]], device=device)], dim=1)
    return gen


def degen(prompt_ids, gen):
    seen = set(prompt_ids); rep = 0
    for t in gen:
        if t in seen:
            rep += 1
        seen.add(t)
    big = [(gen[i], gen[i + 1]) for i in range(len(gen) - 1)]
    d2 = len(set(big)) / max(1, len(big))
    lr = mr = 1
    for i in range(1, len(gen)):
        mr = mr + 1 if gen[i] == gen[i - 1] else 1
        lr = max(lr, mr)
    return rep / max(1, len(gen)), d2, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--thetas", default="1.0,1.05,1.1,1.2,1.3,1.5")
    ap.add_argument("--out", default="runs/fix_loopcheck/raw.json")
    args = ap.parse_args()
    thetas = [float(x) for x in args.thetas.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    pids = [tok(p)["input_ids"] for p in PROMPTS]

    res = {}     # f"{op}_theta{th}" -> {rep_rate, distinct2, longest_run}
    for op, fix in [("raw", False), ("fix", True)]:
        for th in thetas:
            rr = d2 = lr = 0.0
            for p in pids:
                g = greedy(model, p, th, args.max_new, args.device, fix)
                a, b, c = degen(p, g); rr += a; d2 += b; lr += c
            n = len(pids)
            res[f"{op}_theta{th}"] = {"rep_rate": rr / n, "distinct2": d2 / n, "longest_run": lr / n}
        print(f"  {op}: rep_rate by theta " +
              " ".join(f"{th}:{res[f'{op}_theta{th}']['rep_rate']:.3f}" for th in thetas), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "thetas": thetas, "metrics": res}, open(args.out, "w"), indent=2)
    # verdict: under the fix, does rep_rate fall monotonically as theta rises (loops still break)?
    fr = [res[f"fix_theta{th}"]["rep_rate"] for th in thetas]
    breaks = fr[-1] < fr[0] - 0.02 and all(fr[i] >= fr[i + 1] - 0.03 for i in range(len(fr) - 1))
    print(f"  FIX breaks loops (rep_rate {fr[0]:.3f}@θ{thetas[0]} -> {fr[-1]:.3f}@θ{thetas[-1]}, "
          f"monotone-down): {breaks} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
