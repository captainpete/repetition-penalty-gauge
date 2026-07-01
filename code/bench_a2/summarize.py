#!/usr/bin/env python3
"""Combine Part A (JSONSchemaBench validity) + Part B (HumanEval threshold/delimiter) into
results/bench_a2/summary.json. Also computes the theta=1.0-conditioned JSON valid rates
(restricted to schemas the model got right at theta=1.0 raw — the analogue of the original
hand-written experiment, which was at 100% at theta=1.0).

  .venv/bin/python summarize.py
"""
import os, json, argparse

BASE = "results/bench_a2"


def part_a(path):
    d = json.load(open(path))
    # unconditioned rates straight from the run summary
    uncond = d["summary"]
    # theta=1.0-conditioned: restrict to ids valid at theta=1.0 raw
    ok_ids = {r["id"] for r in d["runs"]["raw_theta1.0"] if r["valid"]}
    cond = {}
    for key, recs in d["runs"].items():
        sub = [r for r in recs if r["id"] in ok_ids]
        cond[key] = {"valid_rate": sum(r["valid"] for r in sub) / len(sub),
                     "n": len(sub), "n_valid": sum(r["valid"] for r in sub)}
    return {"model": d["model"], "n_schemas": d["n_schemas"], "max_new": d["max_new"],
            "unconditioned": uncond, "n_theta1_valid": len(ok_ids), "conditioned_on_theta1": cond}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-raw", default=os.path.join(BASE, "json_raw.json"))
    ap.add_argument("--he-summary", default=os.path.join(BASE, "humaneval_summary_humaneval.json"))
    ap.add_argument("--he-qwen-summary", default=os.path.join(BASE, "humaneval_summary_humaneval_qwen.json"))
    ap.add_argument("--out", default=os.path.join(BASE, "summary.json"))
    args = ap.parse_args()

    out = {}
    if os.path.exists(args.json_raw):
        out["json_schemabench"] = part_a(args.json_raw)
    if os.path.exists(args.he_summary):
        out["humaneval_threshold"] = json.load(open(args.he_summary))
    if os.path.exists(args.he_qwen_summary):
        out["humaneval_threshold_qwen"] = json.load(open(args.he_qwen_summary))

    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: (v if k == "json_schemabench" else {kk: v[kk] for kk in ("raw", "fix") if kk in v})
                      for k, v in out.items()}, indent=2, default=str)[:4000])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
