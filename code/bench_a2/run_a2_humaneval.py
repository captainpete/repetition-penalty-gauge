#!/usr/bin/env python3
"""Part B — A2 threshold + delimiter-flip experiment on the 164 official HumanEval prompts
as benchmark-sourced code seeds. Reuses code/run_a2.py's EXACT instrumentation (generate(),
apply_rep_penalty(), is_structural(), the fp32-upcast + argmax-tie discipline). Only the seed
prompts change: hand-written JSON/PY/prose -> HumanEval `prompt` field (all tagged domain
"python", since HumanEval is Python).

  .venv/bin/python run_a2_humaneval.py [--fix]
"""
import os, sys, json, argparse
os.environ.setdefault("HF_HOME", "$HF_HUB_CACHE")
sys.path.insert(0, "code")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import run_a2  # exact instrumentation lives here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bigcode/starcoder2-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N HumanEval prompts (smoke)")
    ap.add_argument("--thetas", default="1.0,1.1,1.2,1.3,1.4,1.5")
    ap.add_argument("--out", default="results/bench_a2/humaneval_raw.json")
    ap.add_argument("--fix", action="store_true", help="normalize (log_softmax) before penalizing")
    args = ap.parse_args()
    run_a2.FIX = args.fix  # generate() reads this module global (exact original wiring)

    thetas = [float(x) for x in args.thetas.split(",")]

    from datasets import load_dataset
    he = load_dataset("openai/openai_humaneval", split="test")
    n = min(args.limit, len(he)) if args.limit else len(he)
    prompts = [(i, he[i]["prompt"]) for i in range(n)]

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"

    # incremental checkpoint (JSONL, one record per prompt x theta) so a killed run resumes
    ckpt = args.out + ".jsonl"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    records = []
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["prompt_idx"], r["theta"]))
                records.append(r)
        print(f"resuming: {len(done)} (prompt,theta) records already in {ckpt}", flush=True)

    with open(ckpt, "a") as cf:
        for pi, ptext in prompts:
            pids = tok(ptext)["input_ids"]
            for theta in thetas:
                if (pi, theta) in done:
                    continue
                positions, gen = run_a2.generate(model, tok, pids, theta, args.max_new, args.device)
                rec = {"prompt_idx": pi, "domain": "python", "theta": theta,
                       "prompt": ptext, "gen_text": tok.decode(gen), "positions": positions}
                records.append(rec)
                cf.write(json.dumps(rec) + "\n")
                cf.flush()
            print(f"  prompt {pi + 1}/{len(prompts)} done", flush=True)

    records.sort(key=lambda r: (r["prompt_idx"], r["theta"]))
    json.dump({"model": args.model, "revision": rev, "max_new": args.max_new,
               "fix": args.fix, "thetas": thetas, "records": records}, open(args.out, "w"))
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
