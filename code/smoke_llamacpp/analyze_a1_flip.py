#!/usr/bin/env python3
"""A1 flip-rate analysis for the llama.cpp smoke replication.

flip rate = fraction of aligned greedy positions where the token id differs
between the c=+5 and c=-5 runs, at a given theta. VALIDITY GATE: at theta=1.0
the flip rate must be EXACTLY 0 (uniform logit bias is a softmax/argmax no-op,
and the repeat penalty is disabled at theta=1)."""
import json, sys
from collections import defaultdict

raw = sys.argv[1] if len(sys.argv) > 1 else \
    "../../results/smoke_llamacpp/a1_raw.json"
out = sys.argv[2] if len(sys.argv) > 2 else \
    "../../results/smoke_llamacpp/a1_summary.json"

d = json.load(open(raw))
recs = d["records"]
g = defaultdict(dict)                    # (prompt_idx, theta) -> {c: gen_ids}
thetas, cs = set(), set()
for r in recs:
    g[(r["prompt_idx"], round(r["theta"], 4))][round(r["c"], 4)] = r["gen_ids"]
    thetas.add(round(r["theta"], 4)); cs.add(round(r["c"], 4))
thetas = sorted(thetas)
cpos, cneg = max(cs), min(cs)

per_theta = {}
for th in thetas:
    tot = fl = 0
    per_prompt = {}
    for (pi, t), m in g.items():
        if t != th:
            continue
        a, b = m[cneg], m[cpos]
        n = min(len(a), len(b))
        pf = sum(a[i] != b[i] for i in range(n))
        per_prompt[pi] = pf / n if n else 0.0
        fl += pf; tot += n
    per_theta[th] = {"flip_rate": fl / tot if tot else 0.0,
                     "positions": tot, "flips": fl,
                     "per_prompt_mean": sum(per_prompt.values()) / len(per_prompt)}

gate_theta = 1.0
gate_pass = abs(per_theta.get(1.0, {}).get("flip_rate", 1.0)) == 0.0

summary = {
    "model": d["model"], "max_new": d["max_new"],
    "penalty_last_n": d["penalty_last_n"], "n_vocab": d.get("n_vocab"),
    "c_pos": cpos, "c_neg": cneg,
    "validity_gate_theta": gate_theta,
    "validity_gate_pass": gate_pass,
    "flip_rate_by_theta": {str(th): per_theta[th] for th in thetas},
    "hf_reference_flip_theta1.3": 0.941,
}
json.dump(summary, open(out, "w"), indent=2)

print(f"model: {d['model']}  (penalty_last_n={d['penalty_last_n']}, max_new={d['max_new']})")
print(f"gauge: c=+{cpos} vs c={cneg}")
print(f"VALIDITY GATE (theta=1.0 flip rate == 0): {'PASS' if gate_pass else 'FAIL'}"
      f"  (flip={per_theta[1.0]['flip_rate']:.4f})")
print("theta | flip_rate | positions | per-prompt-mean")
for th in thetas:
    p = per_theta[th]
    print(f"{th:>5} | {p['flip_rate']:.4f}    | {p['positions']:>6}    | {p['per_prompt_mean']:.4f}")
print(f"HF reference flip @ theta=1.3 (gpt2-large): 0.941")
