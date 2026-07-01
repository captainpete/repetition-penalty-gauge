#!/usr/bin/env python3
"""Smoke: (1) theta=1.0 gate = 0 (c=+5 vs c=-5 token-identical);
(2) batch-size invariance (batch=1 vs batch=8 identical tokens) at theta=1.3;
(3) cross-check vs the ORIGINAL per-step run_a1.generate on a couple prompts (semantics match)."""
import os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bench_lib import batched_greedy, load_prefix_ids

sys.path.insert(0, "code")
from run_a1 import generate as orig_generate  # per-step reference

dev = "cuda"
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).to(dev).eval()
_texts, pids = load_prefix_ids(tok, "results/bench_a1/prefixes.json")
pids = pids[:8]
MN = 40

# (1) theta=1.0 gate
gp, _ = batched_greedy(model, pids, 5.0, 1.0, MN, dev, batch_size=8)
gm, _ = batched_greedy(model, pids, -5.0, 1.0, MN, dev, batch_size=8)
flips = sum(sum(1 for a, b in zip(x, y) if a != b) for x, y in zip(gp, gm))
print(f"(1) theta=1.0 gate: total flips (c=+5 vs c=-5) = {flips} / {8*MN}  -> gate {'PASS' if flips==0 else 'FAIL'}")

# (2) batch-size invariance at theta=1.3, c=5
b1, _ = batched_greedy(model, pids, 5.0, 1.3, MN, dev, batch_size=1)
b8, _ = batched_greedy(model, pids, 5.0, 1.3, MN, dev, batch_size=8)
diff = sum(sum(1 for a, b in zip(x, y) if a != b) for x, y in zip(b1, b8))
print(f"(2) batch invariance theta=1.3 c=5: diffs batch1 vs batch8 = {diff} / {8*MN}  -> {'PASS' if diff==0 else 'DIFFERS'}")

# (3) vs original per-step generate (theta=1.3, c=5 and c=-5)
for c in (5.0, -5.0):
    bref, _ = batched_greedy(model, pids, c, 1.3, MN, dev, batch_size=8)
    mism = 0
    for i, p in enumerate(pids):
        og, _ = orig_generate(model, p, c, 1.3, MN, "greedy", 1234, dev)
        mism += sum(1 for a, b in zip(og, bref[i]) if a != b)
    print(f"(3) vs original per-step, theta=1.3 c={c}: token mismatches = {mism} / {8*MN}")
