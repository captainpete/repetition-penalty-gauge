#!/usr/bin/env python3
"""Analyze the benchmark-sourced A1 re-run -> summary.json + REPORT.md.

Reads results/bench_a1/:
  raw_flip_<label>.json          (Table 1: flip rate c=+5 vs c=-5, theta in {1.0,1.3})
  raw_flip_<label>_fix.json      (Table 3 row 1: fix leg)
  raw_zp_<label>.json            (Table 2: zero-point + natural flip)
and prints/writes the new numbers side-by-side with the old 16-prompt numbers.

  python analyze.py
"""
import os, json, glob, random, argparse
from collections import defaultdict

BOOT = 10000
TEST_THETA = 1.3
CPOS, CNEG = 5.0, -5.0
RESULTS = "results/bench_a1"

# --- old 16-hand-written-prompt numbers (paper-note.tex) for side-by-side comparison ---
OLD_FLIP = {  # model label -> flip @ theta=1.3 (from results/a1b + a1b_scale summaries)
    "gpt2": 0.4966, "gpt2-large": 0.9409, "pythia-2.8b": 0.9050,
    "Qwen2.5-7B": 0.8706, "Qwen2.5-7B-Instruct": 0.8969,
}
OLD_ZP = {  # label -> (frac_pos, median_top1, natural_flip@1.3)  from results/a1_zeropoint
    "gpt2": (0.091, -97.5, 0.569), "gpt2-large": (0.746, 11.9, 0.781),
    "starcoder2-7b": (0.828, 15.5, 0.858), "pythia-2.8b": (0.938, 17.6, 0.815),
    "Qwen2.5-Coder-7B": (0.986, 24.2, 0.796),
}
OLD_ZP_POOLED_NAT13 = 0.764  # mean across the 5 models (paper NOTE)
FLIP_ORDER = ["gpt2-large", "pythia-2.8b", "gpt2", "Qwen2.5-7B", "Qwen2.5-7B-Instruct"]
ZP_ORDER = ["gpt2", "gpt2-large", "starcoder2-7b", "pythia-2.8b", "Qwen2.5-Coder-7B"]


def boot_ci(xs, n=BOOT, seed=0):
    r = random.Random(seed)
    m = len(xs)
    means = sorted(sum(xs[r.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def flip_by_theta(recs, np_):
    g = defaultdict(dict)
    thetas = set()
    for r in recs:
        g[(r["theta"], r["prompt_idx"])][r["c"]] = r["gen_ids"]
        thetas.add(r["theta"])
    out = {}
    for theta in sorted(thetas):
        per, flips, tot = [], 0, 0
        for p in range(np_):
            a, b = g[(theta, p)][CPOS], g[(theta, p)][CNEG]
            k = min(len(a), len(b))
            df = sum(1 for i in range(k) if a[i] != b[i])
            per.append(df / k if k else 0.0)
            flips += df
            tot += k
        out[theta] = (flips / tot if tot else 0.0, per)
    return out


def label_of(path):
    lbl = os.path.basename(path).replace("raw_flip_", "").replace("raw_zp_", "").replace(".json", "")
    return lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RESULTS)
    D = ap.parse_args().dir

    # ---- Table 1: flip rates ----
    flip_models = {}
    for rp in sorted(glob.glob(D + "/raw_flip_*.json")):
        if rp.endswith("_fix.json"):
            continue
        d = json.load(open(rp))
        lbl = d["model"].split("/")[-1]
        fbt = flip_by_theta(d["records"], d["n_prompts"])
        gate = abs(fbt.get(1.0, (1, []))[0])
        test = fbt.get(TEST_THETA)
        lo, hi = boot_ci(test[1]) if test else (0, 0)
        flip_models[lbl] = {
            "revision": d.get("revision"), "dtype": d.get("dtype"), "n_prompts": d["n_prompts"],
            "gate_flip_theta1.0": gate, "gate_pass": gate == 0.0,
            "flip_theta1.3": test[0] if test else None, "ci_theta1.3": [lo, hi],
            "flip_by_theta": {str(t): fbt[t][0] for t in sorted(fbt)},
        }

    # ---- Table 3: fix leg ----
    fix_models = {}
    for rp in sorted(glob.glob(D + "/raw_flip_*_fix.json")):
        d = json.load(open(rp))
        lbl = d["model"].split("/")[-1]
        fbt = flip_by_theta(d["records"], d["n_prompts"])
        fix_models[lbl] = {"flip_by_theta": {str(t): fbt[t][0] for t in sorted(fbt)},
                           "gate_pass": abs(fbt.get(1.0, (1, []))[0]) == 0.0}

    # ---- Table 2: zero-point ----
    zp_models = {}
    for rp in sorted(glob.glob(D + "/raw_zp_*.json")):
        d = json.load(open(rp))
        lbl = d["model"].split("/")[-1]
        zp_models[lbl] = {
            "revision": d.get("revision"), "dtype": d.get("dtype"),
            "frac_seen_logit_positive": d["zero_point"]["frac_seen_logit_positive"],
            "mean_top1_logit": d["zero_point"]["mean_top1_logit"],
            "median_top1_logit": d["zero_point"]["median_top1_logit"],
            "c_natural": d["c_natural"],
            "natural_flip_theta1.3": d["flip_rate"].get("theta1.3_natural(-median)"),
            "synth5_flip_theta1.3": d["flip_rate"].get("theta1.3_synth_5"),
        }
    nat13 = [zp_models[m]["natural_flip_theta1.3"] for m in ZP_ORDER if m in zp_models]
    pooled_nat13 = sum(nat13) / len(nat13) if nat13 else None

    summary = {
        "protocol": "200 32-token prefixes sampled from the WikiText-103 test set, seed 0",
        "n_prompts": 200, "n_positions_per_model": 40000, "max_new": 200,
        "flip": flip_models, "fix": fix_models,
        "zeropoint": {"models": zp_models, "pooled_natural_flip_theta1.3": pooled_nat13,
                      "pooled_definition": "mean of per-model natural flip@theta=1.3 (matches paper NOTE)"},
        "old_16prompt": {"flip": OLD_FLIP, "zeropoint": OLD_ZP,
                         "zeropoint_pooled_natural_flip_theta1.3": OLD_ZP_POOLED_NAT13},
    }
    json.dump(summary, open(D + "/summary.json", "w"), indent=2)

    # ---- REPORT.md ----
    L = ["# A1 on benchmark-sourced prefixes — RESULT\n",
         "**Protocol.** 200 32-token prefixes sampled from the WikiText-103 test set, seed 0 "
         "(SimCTG / contrastive-search open-ended-generation protocol). Prefixes defined as TEXT "
         "(first 32 GPT-2 BPE tokens of each sampled prose segment, decoded back) so every model "
         "shares them. All runs greedy, 200 new tokens, HF sign-branch penalty, gauge shift added "
         "before the penalty, fp32 for the shift/penalty/argmax. See `REPORT.md` header table for "
         "old (16 hand-written prompts) vs new numbers.\n",
         "## Table 1 — gauge flip rate (c=+5 vs c=-5)\n",
         "| Model | gate flip(θ=1.0) | flip(θ=1.3) NEW | 95% CI (θ=1.3) | flip(θ=1.3) OLD (16-prompt) |",
         "|---|---|---|---|---|"]
    all_gate = True
    for m in FLIP_ORDER:
        if m not in flip_models:
            continue
        r = flip_models[m]
        all_gate = all_gate and r["gate_pass"]
        ci = r["ci_theta1.3"]
        L.append(f"| {m} | {r['gate_flip_theta1.0']:.3f} {'PASS' if r['gate_pass'] else 'FAIL'} | "
                 f"**{r['flip_theta1.3']:.3f}** | [{ci[0]:.3f}, {ci[1]:.3f}] | {OLD_FLIP.get(m,float('nan')):.3f} |")
    L.append(f"\nValidity gate (θ=1.0 flip == 0 for every model): **{'PASS' if all_gate else 'FAIL'}**\n")

    L += ["## Table 2 — zero-point (five checkpoints)\n",
          "| Model | frac seen logit>0 NEW | (OLD) | median top-1 NEW | (OLD) | nat flip@1.3 NEW | (OLD) |",
          "|---|---|---|---|---|---|---|"]
    for m in ZP_ORDER:
        if m not in zp_models:
            continue
        r = zp_models[m]
        o = OLD_ZP.get(m, (float('nan'),) * 3)
        L.append(f"| {m} | {r['frac_seen_logit_positive']:.3f} | ({o[0]:.3f}) | "
                 f"{r['median_top1_logit']:.2f} | ({o[1]:.1f}) | "
                 f"**{r['natural_flip_theta1.3']:.3f}** | ({o[2]:.3f}) |")
    L.append(f"\nPooled natural flip@θ=1.3 (mean of 5 models): **{pooled_nat13:.3f}** "
             f"(OLD {OLD_ZP_POOLED_NAT13:.3f})\n" if pooled_nat13 is not None else "")

    L += ["## Table 3 (row 1) — fix leg (log_softmax before penalty)\n",
          "| Model | flip(θ=1.0) | flip(θ=1.3) | expected |", "|---|---|---|---|"]
    for m in ["gpt2-large", "pythia-2.8b", "gpt2"]:
        if m not in fix_models:
            continue
        fb = fix_models[m]["flip_by_theta"]
        L.append(f"| {m} | {fb.get('1.0', float('nan')):.3f} | {fb.get('1.3', float('nan')):.3f} | 0.000 |")

    open(D + "/REPORT.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {D}/summary.json and {D}/REPORT.md")


if __name__ == "__main__":
    main()
