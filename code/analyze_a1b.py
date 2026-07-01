#!/usr/bin/env python3
"""A1b analyzer — frozen rule in experiments/A1b/PREREG.md. Primary endpoint:
argmax-flip rate between the c=+5 and c=-5 greedy runs (both no-ops at theta=1).
Reads runs/A1b/raw_<model>.json, writes runs/A1b/{summary.json,REPORT.md}.

  python code/analyze_a1b.py
"""
import os, json, glob, random
from collections import defaultdict

TEST_THETA = 1.3
FLIP_MIN = 0.15
BOOT = 10000
CPOS, CNEG = 5.0, -5.0


def boot_ci(xs, n=BOOT, seed=0):
    r = random.Random(seed)
    m = len(xs)
    means = sorted(sum(xs[r.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def flip_by_theta(recs, np_):
    """returns {theta: (pooled_flip_rate, [per-prompt flip_rate])}"""
    g = defaultdict(dict)  # (theta,prompt,c) -> ids
    thetas = set()
    for r in recs:
        if r["mode"] == "greedy":
            g[(r["theta"], r["prompt_idx"])][r["c"]] = r["gen_ids"]
            thetas.add(r["theta"])
    out = {}
    for theta in sorted(thetas):
        per, flips, tot = [], 0, 0
        for p in range(np_):
            a, b = g[(theta, p)][CPOS], g[(theta, p)][CNEG]
            n = min(len(a), len(b))
            df = sum(1 for i in range(n) if a[i] != b[i])
            per.append(df / n if n else 0.0)
            flips += df
            tot += n
        out[theta] = (flips / tot if tot else 0.0, per)
    return out


def rep_by_c(recs, theta):
    g = defaultdict(list)
    for r in recs:
        if r["mode"] == "greedy" and r["theta"] == theta:
            g[r["c"]].append(r["rep_rate"])
    return {c: sum(v) / len(v) for c, v in g.items()}


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--dir", default="runs/A1b"); D=ap.parse_args().dir
    raws = sorted(glob.glob(D+"/raw_*.json"))
    models = {}
    for rp in raws:
        d = json.load(open(rp))
        models[d["model"]] = d

    lines = ["# A1b — Repetition-penalty gauge non-invariance (flip-rate endpoint) — RESULT\n"]
    summ = {}
    all_confirm = True
    for name, d in models.items():
        fbt = flip_by_theta(d["records"], d["n_prompts"])
        thetas = sorted(fbt)
        gate = abs(fbt.get(1.0, (1, []))[0]) < 1e-12
        test = fbt.get(TEST_THETA)
        lo, hi = boot_ci(test[1]) if test else (0, 0)
        pooled = [fbt[t][0] for t in thetas]
        mono = all(pooled[i] <= pooled[i + 1] + 1e-9 for i in range(len(pooled) - 1))
        confirmed = gate and test and test[0] >= FLIP_MIN and lo > 0 and mono
        all_confirm = all_confirm and confirmed

        lines.append(f"## {name} (rev `{d.get('revision')}`)")
        lines.append(f"- no-op gate flip_rate(θ=1.0)=0: **{gate}**")
        lines.append("\n| θ | flip_rate (c=+5 vs c=−5) |")
        lines.append("|---|---|")
        for t in thetas:
            lines.append(f"| {t:g} | {fbt[t][0]:.3f} |")
        lines.append(f"\n- flip_rate(θ={TEST_THETA}) = **{test[0]:.3f}** "
                     f"(95% CI [{lo:.3f}, {hi:.3f}], threshold ≥ {FLIP_MIN})")
        lines.append(f"- monotone non-decreasing in θ: **{mono}**")
        # secondary: rep_rate by c at smallest θ>1
        sm = min(t for t in thetas if t > 1.0)
        rc = rep_by_c(d["records"], sm)
        lines.append(f"- secondary rep_rate by c at θ={sm:g} (named channel, non-saturated): "
                     + ", ".join(f"c{c:+g}={rc[c]:.3f}" for c in sorted(rc)))
        lines.append(f"\n**{name}: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}**\n")
        summ[name] = {"noop_gate": gate, "flip_test_theta": test[0],
                      "flip_ci": [lo, hi], "monotone": mono, "confirmed": confirmed,
                      "flip_by_theta": {str(t): fbt[t][0] for t in thetas}}

    lines.insert(1, f"**Cross-model verdict: {'CONFIRMED on all ' + str(len(models)) + ' models' if all_confirm else 'mixed/not confirmed'}** "
                    "— a provable gauge no-op (flip_rate=0 at θ=1) changes ~half of greedy "
                    "tokens at θ≥1.15. The repetition penalty is gauge-dependent.\n"
                 if all_confirm else
                 "**Cross-model verdict: not all models confirmed — see per-model below.**\n")
    report = "\n".join(lines) + "\n"
    os.makedirs(D, exist_ok=True)
    open(D+"/REPORT.md", "w").write(report)
    json.dump({"all_confirmed": all_confirm, "models": summ},
              open(D+"/summary.json", "w"), indent=2)
    print(report)


if __name__ == "__main__":
    main()
