#!/usr/bin/env python3
"""A1 flip-rate run on WikiText-103 prefixes (paper-note Table 1; Table 3 fix leg with --fix).

For a model: greedy-decode the 200 manifest prefixes with c=+5 and c=-5 at theta in {1.0, 1.3},
200 new tokens each. Writes results/bench_a1/raw_<label>.json with the per-prefix gen ids for both
c values, so analyze.py can compute the flip rate (fraction of aligned positions where c=+5 and
c=-5 differ) and its bootstrap CI.

  .venv/bin/python run_flip.py --model gpt2 --dtype float32
  .venv/bin/python run_flip.py --model Qwen/Qwen2.5-7B --dtype bfloat16 --batch-size 16
  # fix leg:
  .venv/bin/python run_flip.py --model gpt2 --dtype float32 --fix
  # subtractive (presence-style) control leg, alpha=1.0 (single condition, no theta grid):
  .venv/bin/python run_flip.py --model gpt2 --dtype float32 --subtractive 1.0
"""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bench_lib import batched_greedy, load_prefix_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--thetas", default="1.0,1.3")
    ap.add_argument("--cs", default="5,-5")
    ap.add_argument("--fix", action="store_true", help="normalize (log_softmax) before penalizing")
    ap.add_argument("--subtractive", type=float, default=None,
                    help="presence-style control: z_i -= ALPHA for every seen id (replaces the "
                         "sign-branch penalty; single condition, no theta grid). Try 1.0.")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="use only first N prefixes (smoke)")
    ap.add_argument("--prefixes", default="results/bench_a1/prefixes.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    thetas = [float(x) for x in args.thetas.split(",")]
    cs = [float(x) for x in args.cs.split(",")]
    suffix = "_subtractive" if args.subtractive is not None else ("_fix" if args.fix else "")
    label = args.model.split("/")[-1] + suffix
    out = args.out or f"results/bench_a1/raw_flip_{label}.json"

    tok = AutoTokenizer.from_pretrained(args.model)
    _texts, pids = load_prefix_ids(tok, args.prefixes)
    if args.limit:
        pids = pids[:args.limit]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"

    records = []  # one per (prefix, theta|alpha, c)
    if args.subtractive is not None:
        # Subtractive presence-style control: single alpha condition, no theta grid.
        alpha = args.subtractive
        for c in cs:
            gens, _ = batched_greedy(model, pids, c, 1.0, args.max_new, args.device,
                                     subtractive=alpha, batch_size=args.batch_size)
            for pi, g in enumerate(gens):
                records.append({"prompt_idx": pi, "alpha": alpha, "c": c, "gen_ids": g})
            print(f"  alpha={alpha} c={c} done ({len(gens)} prefixes)", flush=True)
    else:
        for theta in thetas:
            for c in cs:
                gens, _ = batched_greedy(model, pids, c, theta, args.max_new, args.device,
                                         fix=args.fix, batch_size=args.batch_size)
                for pi, g in enumerate(gens):
                    records.append({"prompt_idx": pi, "theta": theta, "c": c, "gen_ids": g})
                print(f"  theta={theta} c={c} done ({len(gens)} prefixes)", flush=True)

    payload = {"model": args.model, "revision": rev, "dtype": args.dtype, "fix": args.fix,
               "subtractive": args.subtractive, "max_new": args.max_new,
               "thetas": thetas, "cs": cs, "n_prompts": len(pids),
               "batch_size": args.batch_size, "records": records}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"))
    print(f"wrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
