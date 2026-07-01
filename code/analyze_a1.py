#!/usr/bin/env python3
"""A1 analyzer — applies the frozen decision rule in PREREG.md to runs/A1/raw.json,
writes runs/A1/summary.json + runs/A1/REPORT.md, prints the verdict.

  python code/analyze_a1.py
"""
import os, json, argparse, random
from collections import defaultdict

NOOP_THETA = 1.0
TEST_THETA = 1.3
GAP_MIN = 0.15           # frozen: min absolute distinct-2 gap c=+5 vs c=-5
BOOT = 10000


def boot_ci(diffs, n=BOOT, seed=0):
    r = random.Random(seed)
    m = len(diffs)
    means = []
    for _ in range(n):
        s = sum(diffs[r.randrange(m)] for _ in range(m)) / m
        means.append(s)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="runs/A1/raw.json")
    ap.add_argument("--out-dir", default="runs/A1")
    args = ap.parse_args()

    d = json.load(open(args.raw))
    recs = d["records"]
    cs = d["cs"]

    # ---- No-op leg: at theta=1.0, all c token-identical per (prompt, mode) ----
    by_pm = defaultdict(dict)   # (prompt,mode) -> {c: gen_ids}
    for r in recs:
        if r["theta"] == NOOP_THETA:
            by_pm[(r["prompt_idx"], r["mode"])][r["c"]] = tuple(r["gen_ids"])
    noop_total = noop_ok = 0
    noop_fail = []
    for (pi, mode), m in by_pm.items():
        noop_total += 1
        ref = m[cs[0]]
        if all(m[c] == ref for c in cs):
            noop_ok += 1
        else:
            noop_fail.append((pi, mode))
    noop_pass = (noop_total > 0 and noop_ok == noop_total)

    # ---- Inversion leg: greedy, theta=TEST_THETA, distinct-2 by c ----
    def grid(metric, mode, theta):
        g = defaultdict(dict)   # c -> {prompt_idx: value}
        for r in recs:
            if r["mode"] == mode and r["theta"] == theta:
                g[r["c"]][r["prompt_idx"]] = r[metric]
        return g

    d2 = grid("distinct2", "greedy", TEST_THETA)
    rr = grid("rep_rate", "greedy", TEST_THETA)
    en = grid("mean_entropy", "greedy", TEST_THETA)

    def cmean(g, c):
        v = g[c]
        return sum(v.values()) / len(v)

    cpos, cneg = max(cs), min(cs)
    prompts = sorted(d2[cpos].keys())
    diffs = [d2[cpos][p] - d2[cneg][p] for p in prompts]   # per-prompt distinct-2 gap
    gap = sum(diffs) / len(diffs)
    lo, hi = boot_ci(diffs)

    means_d2 = [(c, cmean(d2, c)) for c in cs]
    ordered = all(means_d2[i][1] <= means_d2[i + 1][1] + 1e-9 for i in range(len(means_d2) - 1))
    rr_top_at_neg = (max(cs, key=lambda c: cmean(rr, c)) == cneg)

    inv_confirmed = (gap >= GAP_MIN) and (lo > 0) and ordered

    # ---- write report ----
    lines = []
    lines.append("# A1 — Repetition-penalty gauge (non-)invariance — RESULT\n")
    lines.append(f"Model: `{d['model']}` (rev `{d.get('revision')}`), "
                 f"{d['n_prompts']} prompts x {d['max_new']} tokens, seed {d['seed']}.\n")

    lines.append("## No-op leg (c must be behaviorally invisible at theta=1)")
    verdict = "PASS" if noop_pass else "FAIL"
    lines.append(f"**{verdict}** — {noop_ok}/{noop_total} (prompt,mode) pairs token-identical "
                 f"across c={cs} at theta=1.0.")
    if noop_fail:
        lines.append(f"  failures: {noop_fail}")
    lines.append("")

    lines.append(f"## Inversion leg (greedy, theta={TEST_THETA})")
    lines.append("distinct-2 / rep_rate / entropy as a function of the gauge c "
                 "(both extremes are provable no-ops at theta=1):\n")
    lines.append("| c | distinct-2 | rep_rate | mean entropy (nats) |")
    lines.append("|---|---|---|---|")
    for c in cs:
        lines.append(f"| {c:+g} | {cmean(d2,c):.3f} | {cmean(rr,c):.3f} | {cmean(en,c):.3f} |")
    lines.append("")
    lines.append(f"- distinct-2 gap (c={cpos:+g} − c={cneg:+g}): **{gap:.3f}** "
                 f"(95% CI [{lo:.3f}, {hi:.3f}], threshold ≥ {GAP_MIN})")
    lines.append(f"- distinct-2 monotone non-decreasing in c: **{ordered}**")
    lines.append(f"- rep_rate maximal at c={cneg:+g} (penalty weakest): **{rr_top_at_neg}**")
    lines.append("")

    if noop_pass and inv_confirmed:
        v = (f"**CONFIRMED.** c is a provable no-op at theta=1 (no-op leg passed), yet at "
             f"theta={TEST_THETA} the same penalty produces a distinct-2 gap of {gap:.3f} "
             f"(CI excludes 0), ordered in c. A behavior-preserving reparametrization changes "
             f"generation — the repetition penalty is gauge-dependent. Consensus falsified.")
    elif noop_pass and not inv_confirmed:
        v = ("**NOT CONFIRMED.** No-op leg passed but the theta>1 c-dependence did not clear the "
             "frozen threshold (gap/CI/order). Consistent with consensus on this model/grid.")
    else:
        v = ("**INVALID.** No-op leg failed — c was not behaviorally invisible at theta=1, so the "
             "setup (not the claim) is wrong. Debug before interpreting the inversion leg.")
    lines.append("## Verdict")
    lines.append(v)
    report = "\n".join(lines) + "\n"

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "REPORT.md"), "w").write(report)
    json.dump({
        "model": d["model"], "revision": d.get("revision"),
        "noop_pass": noop_pass, "noop_ok": noop_ok, "noop_total": noop_total,
        "test_theta": TEST_THETA, "distinct2_gap": gap, "gap_ci": [lo, hi],
        "ordered": ordered, "rep_rate_top_at_cneg": rr_top_at_neg,
        "inversion_confirmed": bool(noop_pass and inv_confirmed),
        "distinct2_by_c": {str(c): cmean(d2, c) for c in cs},
        "rep_rate_by_c": {str(c): cmean(rr, c) for c in cs},
        "entropy_by_c": {str(c): cmean(en, c) for c in cs},
    }, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

    print(report)


if __name__ == "__main__":
    main()
