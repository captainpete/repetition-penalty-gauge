#!/usr/bin/env python3
"""A1 (rep-penalty) — REAL cross-checkpoint logit zero-point + natural-Δc flip-rate./repetition-penalty.md. The paper's headline claim ("repetition_penalty=1.3 already denotes
different interventions across real models, no synthetic shift required") is currently asserted from a
synthetic c=±5. This measures where each real model actually sits relative to the sign boundary the
penalty branches on, and the A1 flip-rate at the NATURAL offset between real models.

Per model: (1) zero-point stats — fraction of seen/penalized-token logits that are positive (divide
branch) vs negative (×θ branch), and mean/median top-1 logit; (2) a c-sweep greedy flip-rate vs c=0 at
θ∈{1.15,1.3} so the analyzer can read off the flip-rate at the measured natural Δc.

  python code/run_a1_zeropoint.py --model gpt2
"""
import os, json, argparse, statistics
os.environ.setdefault("HF_HUB_CACHE", "/hf/hub")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_a1 import PROMPTS, apply_rep_penalty       # reuse the prompt set + exact HF penalty semantics


@torch.no_grad()
def greedy(model, prompt_ids, c, theta, max_new, device, collect_zp=False):
    """greedy decode with gauge shift c + HF rep-penalty; optionally collect zero-point stats."""
    seen = set(prompt_ids)
    seen_idx = torch.tensor(sorted(seen), device=device, dtype=torch.long)
    cur = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    past, gen = None, []
    zp_pos, zp_tot, ztops = 0, 0, []
    for _ in range(max_new):
        feed = cur if past is None else cur[:, -1:]
        out = model(feed, past_key_values=past, use_cache=True); past = out.past_key_values
        logits = out.logits[0, -1, :].float() + c
        if collect_zp:
            ztops.append(float(logits.max()))
            sv = logits[seen_idx]
            zp_pos += int((sv > 0).sum()); zp_tot += sv.numel()
        tok = int(apply_rep_penalty(logits, seen_idx, theta).argmax())
        gen.append(tok)
        if tok not in seen:
            seen.add(tok); seen_idx = torch.cat([seen_idx, torch.tensor([tok], device=device)])
        cur = torch.cat([cur, torch.tensor([[tok]], device=device)], dim=1)
    return gen, (zp_pos, zp_tot, ztops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--thetas", default="1.15,1.3")
    ap.add_argument("--out", default="runs/A1_zeropoint/raw.json")
    args = ap.parse_args()
    thetas = [float(x) for x in args.thetas.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"
    pids = [tok(p)["input_ids"] for p in PROMPTS]

    # (1) zero-point: c=0, theta=1 (no penalty) — measure sign distribution + top-1 logit scale
    zp_pos = zp_tot = 0; ztops = []
    base_gen = {}
    for i, p in enumerate(pids):
        g, (pp, tt, zt) = greedy(model, p, 0.0, 1.0, args.max_new, args.device, collect_zp=True)
        zp_pos += pp; zp_tot += tt; ztops += zt
    frac_pos = zp_pos / max(1, zp_tot)
    zp = {"frac_seen_logit_positive": frac_pos, "mean_top1_logit": statistics.mean(ztops),
          "median_top1_logit": statistics.median(ztops)}
    print(f"{args.model}: frac seen-logit positive {frac_pos:.3f}, "
          f"mean/median top1 logit {zp['mean_top1_logit']:.2f}/{zp['median_top1_logit']:.2f}", flush=True)

    # (2) flip-rate vs c=0 (raw gauge) at offsets RELATIVE to this model's own zero-point. c_natural =
    # -median brings the model to a mean-centred gauge (a real gauge the penalty could see); compared to
    # the synthetic c=5 the paper currently uses. Token mismatch fraction vs the raw (c=0) decode.
    med = zp["median_top1_logit"]
    c_natural = -med                                         # centre the model's zero-point to 0
    offsets = {"natural(-median)": c_natural, "half_natural": c_natural / 2,
               "synth_5": 5.0, "synth_-5": -5.0}
    flip = {}
    for theta in thetas:
        ref = [greedy(model, p, 0.0, theta, args.max_new, args.device)[0] for p in pids]
        for name, c in offsets.items():
            mism = tots = 0
            for i, p in enumerate(pids):
                g = greedy(model, p, c, theta, args.max_new, args.device)[0]
                mism += sum(1 for a, b in zip(ref[i], g) if a != b); tots += len(ref[i])
            flip[f"theta{theta}_{name}"] = mism / max(1, tots)
        print(f"  theta={theta}: natural(c={c_natural:.1f}) flip "
              f"{flip[f'theta{theta}_natural(-median)']:.3f} vs synth_5 {flip[f'theta{theta}_synth_5']:.3f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "revision": rev, "thetas": thetas, "c_natural": c_natural,
               "zero_point": zp, "flip_rate": flip}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
