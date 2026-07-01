#!/usr/bin/env python3
"""Diagnostic: do the provably-equivalent c=+5 and c=-5 greedy sequences actually
diverge? distinct-2 saturated at 1.0, so measure divergence directly from raw.json."""
import json, statistics, sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/A1/raw.json"))
R = d["records"]
NP = d["n_prompts"]


def seqs(theta, mode):
    m = {}
    for r in R:
        if r["theta"] == theta and r["mode"] == mode:
            m[(r["prompt_idx"], r["c"])] = r["gen_ids"]
    return m


for theta in d["thetas"]:
    g = seqs(theta, "greedy")
    flips = tot = 0
    firstdiv = []
    for p in range(NP):
        a, b = g[(p, 5.0)], g[(p, -5.0)]
        n = min(len(a), len(b))
        diff = [i for i in range(n) if a[i] != b[i]]
        tot += n
        flips += len(diff)
        firstdiv.append(diff[0] if diff else -1)
    nd = sum(1 for x in firstdiv if x >= 0)
    med = statistics.median(sorted(x for x in firstdiv if x >= 0)) if nd else None
    pct = 100 * flips / tot if tot else 0
    print(f"theta={theta}: c=+5 vs c=-5 greedy -> {flips}/{tot} tokens differ "
          f"({pct:.1f}%); {nd}/{NP} prompts diverge; median first-divergence idx={med}")
