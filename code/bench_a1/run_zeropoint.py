#!/usr/bin/env python3
"""A1 zero-point run on WikiText-103 prefixes (paper-note Table 2).

Matches run_a1_zeropoint.py's metric definitions exactly:
  (1) base greedy decode at c=0, theta=1 over the 200 prefixes; collect
      - frac_seen_logit_positive = fraction of seen-token logits > 0 at decode time
        (pooled over every step and every seen id, measured BEFORE that step's token is added),
      - mean_top1_logit / median_top1_logit = mean/median of the per-step top-1 (max) logit,
        where median = statistics.median over all (prefix, step) top-1 logits (as in the original).
  (2) natural-gauge flip rate at theta=1.3: c_natural = -median_top1_logit (re-centre the model to
      its own median), greedy-decode at (c=c_natural, theta) and compare token-for-token against the
      raw-gauge (c=0, theta) decode. flip = fraction of aligned positions that differ. Also records
      the synthetic c=+5 flip for reference. Pooled over prefixes.

  .venv/bin/python run_zeropoint.py --model gpt2 --dtype float32
  .venv/bin/python run_zeropoint.py --model bigcode/starcoder2-7b --dtype bfloat16 --batch-size 16
"""
import os, json, argparse, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bench_lib import batched_greedy, load_prefix_ids


def flip_rate(ref, alt):
    tot = mism = 0
    for a, b in zip(ref, alt):
        n = min(len(a), len(b))
        mism += sum(1 for i in range(n) if a[i] != b[i])
        tot += n
    return mism / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--thetas", default="1.3")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefixes", default="results/bench_a1/prefixes.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    thetas = [float(x) for x in args.thetas.split(",")]
    label = args.model.split("/")[-1]
    out = args.out or f"results/bench_a1/raw_zp_{label}.json"

    tok = AutoTokenizer.from_pretrained(args.model)
    _texts, pids = load_prefix_ids(tok, args.prefixes)
    if args.limit:
        pids = pids[:args.limit]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"

    # (1) zero-point at c=0, theta=1
    base_gen, zp = batched_greedy(model, pids, 0.0, 1.0, args.max_new, args.device,
                                  batch_size=args.batch_size, collect_zp=True)
    frac_pos = zp["zp_pos"] / max(1, zp["zp_tot"])
    med = statistics.median(zp["ztops"])
    zero_point = {"frac_seen_logit_positive": frac_pos,
                  "mean_top1_logit": statistics.mean(zp["ztops"]),
                  "median_top1_logit": med}
    print(f"{args.model}: frac_pos={frac_pos:.3f} mean/median top1={zero_point['mean_top1_logit']:.2f}/"
          f"{med:.2f}", flush=True)

    # (2) natural-gauge flip vs raw gauge
    c_natural = -med
    flip = {}
    for theta in thetas:
        ref, _ = batched_greedy(model, pids, 0.0, theta, args.max_new, args.device,
                                batch_size=args.batch_size)
        nat, _ = batched_greedy(model, pids, c_natural, theta, args.max_new, args.device,
                                batch_size=args.batch_size)
        syn, _ = batched_greedy(model, pids, 5.0, theta, args.max_new, args.device,
                                batch_size=args.batch_size)
        flip[f"theta{theta}_natural(-median)"] = flip_rate(ref, nat)
        flip[f"theta{theta}_synth_5"] = flip_rate(ref, syn)
        print(f"  theta={theta}: natural(c={c_natural:.1f}) flip "
              f"{flip[f'theta{theta}_natural(-median)']:.3f} vs synth_5 "
              f"{flip[f'theta{theta}_synth_5']:.3f}", flush=True)

    payload = {"model": args.model, "revision": rev, "dtype": args.dtype, "thetas": thetas,
               "c_natural": c_natural, "n_prompts": len(pids), "max_new": args.max_new,
               "batch_size": args.batch_size, "zero_point": zero_point, "flip_rate": flip}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
