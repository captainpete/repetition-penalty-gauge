#!/usr/bin/env python3
"""Control-leg analysis for the A1 gauge probe -> summary_controls.json.

Two independent pieces, both reading results/bench_a1/raw_flip_*.json:

  (A) SUBTRACTIVE control (presence-style penalty, z_i -= alpha for every seen id).
      Reads raw_flip_<label>_subtractive.json (c=+5 vs c=-5, single alpha). Reports the flip
      rate over the 40,000 aligned positions. A subtractive penalty is gauge-invariant (a scalar
      shift c added before subtracting the same alpha from the same seen-set can never move the
      argmax), so the expected flip rate is EXACTLY 0 — a true control, like the theta=1.0 gate.
      We also cite that theta=1.0 gate (alpha=0 duplicates it, so it is not rerun).

  (B) CASCADE-HONESTY stats on the existing CTRL (sign-branch) flip runs at theta=1.3 (and 1.0):
      per model, over the 200 prefixes x 200 aligned positions (c=+5 vs c=-5):
        - frac_diverging: fraction of prefixes whose two gauge-runs differ at any position,
        - first-divergence position (0-indexed generated-token offset): median + IQR (Q1,Q3),
          among diverging prefixes,
        - mean flip rate BEFORE first divergence (0 by construction — verified) vs AFTER
          (how unaligned the two continuations are once they have seeded a difference).
      These separate the per-decision gauge seed from cascade amplification. At theta=1.0 no
      prefix should diverge (the gate); we report that too.

  python analyze_controls.py   # no GPU
"""
import os, json, glob, statistics
import numpy as np

RESULTS = "results/bench_a1"
CPOS, CNEG = 5.0, -5.0
FLIP_ORDER = ["gpt2-large", "pythia-2.8b", "gpt2", "Qwen2.5-7B", "Qwen2.5-7B-Instruct"]


def records_by_prefix(recs, key, keyval):
    """{prompt_idx: {c: gen_ids}} for records whose recs[key]==keyval."""
    byp = {}
    for r in recs:
        if r.get(key) != keyval:
            continue
        byp.setdefault(r["prompt_idx"], {})[r["c"]] = r["gen_ids"]
    return byp


def flip_rate(byp):
    flips = tot = 0
    for cv in byp.values():
        a, b = cv[CPOS], cv[CNEG]
        k = min(len(a), len(b))
        flips += sum(1 for i in range(k) if a[i] != b[i])
        tot += k
    return (flips / tot if tot else 0.0), flips, tot


def cascade_stats(byp):
    """First-divergence + before/after flip stats over the prefixes in byp (a single theta)."""
    n = len(byp)
    firstdiv, after_rates = [], []
    before_flips = before_tot = 0
    for cv in byp.values():
        a, b = cv[CPOS], cv[CNEG]
        k = min(len(a), len(b))
        fd = next((i for i in range(k) if a[i] != b[i]), None)
        if fd is None:
            continue
        firstdiv.append(fd)
        before_flips += sum(1 for i in range(fd) if a[i] != b[i])  # 0 by construction
        before_tot += fd
        af = sum(1 for i in range(fd, k) if a[i] != b[i])
        after_rates.append(af / (k - fd))
    out = {
        "n_prefixes": n,
        "n_diverging": len(firstdiv),
        "frac_diverging": len(firstdiv) / n if n else 0.0,
        "before_first_div_flips": before_flips,   # must be 0
        "before_first_div_positions": before_tot,
        "before_first_div_flip_rate": (before_flips / before_tot) if before_tot else 0.0,
    }
    if firstdiv:
        fd = np.array(firstdiv)
        out.update({
            "first_div_pos_median": float(np.median(fd)),
            "first_div_pos_q1": float(np.percentile(fd, 25)),
            "first_div_pos_q3": float(np.percentile(fd, 75)),
            "first_div_pos_min": int(fd.min()),
            "first_div_pos_max": int(fd.max()),
            "after_first_div_flip_rate_mean": float(np.mean(after_rates)),
        })
    return out


def main():
    # ---- (A) subtractive control ----
    subtractive = {}
    for rp in sorted(glob.glob(RESULTS + "/raw_flip_*_subtractive.json")):
        d = json.load(open(rp))
        lbl = d["model"].split("/")[-1]
        alpha = d.get("subtractive")
        byp = records_by_prefix(d["records"], "alpha", alpha)
        rate, flips, tot = flip_rate(byp)
        subtractive[lbl] = {
            "revision": d.get("revision"), "dtype": d.get("dtype"), "alpha": alpha,
            "n_prompts": d["n_prompts"], "flip_rate": rate, "flips": flips, "positions": tot,
            "gate_pass": flips == 0,
        }

    # ---- (B) cascade stats on CTRL sign-branch runs ----
    cascade = {}
    for rp in sorted(glob.glob(RESULTS + "/raw_flip_*.json")):
        base = os.path.basename(rp)
        if base.endswith("_fix.json") or base.endswith("_subtractive.json"):
            continue
        d = json.load(open(rp))
        lbl = d["model"].split("/")[-1]
        entry = {}
        for theta in sorted(set(r["theta"] for r in d["records"])):
            byp = records_by_prefix(d["records"], "theta", theta)
            entry[f"theta{theta}"] = cascade_stats(byp)
        cascade[lbl] = {"revision": d.get("revision"), "dtype": d.get("dtype"), **entry}

    summary = {
        "protocol": "200 32-token WikiText-103 prefixes, seed 0; 200 greedy tokens; c=+5 vs c=-5.",
        "subtractive_control": {
            "penalty": "presence-style: z_i -= alpha for every previously-seen id (prompt+gen), "
                       "gauge shift c applied BEFORE it, fp32; single alpha, no theta grid.",
            "expected_flip_rate": 0.0,
            "note": "gauge-invariant control; alpha=0 duplicates the existing theta=1.0 gate "
                    "(0/40,000 on all Table-1 models) and is not rerun.",
            "models": subtractive,
        },
        "cascade_honesty": {
            "definition": "per model at each theta: frac of 200 prefixes whose c=+5 and c=-5 "
                          "gauge-runs diverge at all; first-divergence position (0-indexed "
                          "generated-token offset) median/IQR among diverging prefixes; mean "
                          "per-prefix flip rate before (=0 by construction) vs after first "
                          "divergence.",
            "models": cascade,
        },
    }
    out = RESULTS + "/summary_controls.json"
    json.dump(summary, open(out, "w"), indent=2)

    # ---- console tables ----
    print("=== Subtractive (presence-style) control: flip rate c=+5 vs c=-5 ===")
    print(f"{'model':22} {'alpha':>5} {'flip_rate':>10} {'flips/pos':>14} gate")
    for m in FLIP_ORDER:
        if m in subtractive:
            s = subtractive[m]
            print(f"{m:22} {s['alpha']:>5} {s['flip_rate']:>10.4f} "
                  f"{str(s['flips'])+'/'+str(s['positions']):>14} "
                  f"{'PASS' if s['gate_pass'] else 'FAIL'}")

    print("\n=== Cascade honesty (CTRL sign-branch), theta=1.3 ===")
    print(f"{'model':22} {'frac_div':>9} {'med_fd':>7} {'IQR':>13} {'after_flip':>11} {'before':>7}")
    for m in FLIP_ORDER:
        if m not in cascade:
            continue
        c = cascade[m].get("theta1.3", {})
        iqr = f"[{c.get('first_div_pos_q1',0):.0f},{c.get('first_div_pos_q3',0):.0f}]"
        print(f"{m:22} {c['frac_diverging']:>9.3f} {c.get('first_div_pos_median',0):>7.0f} "
              f"{iqr:>13} {c.get('after_first_div_flip_rate_mean',0):>11.3f} "
              f"{c['before_first_div_flip_rate']:>7.3f}")

    print("\n=== theta=1.0 gate (frac diverging should be 0) ===")
    for m in FLIP_ORDER:
        if m in cascade:
            c = cascade[m].get("theta1.0", {})
            print(f"{m:22} frac_diverging={c['frac_diverging']:.3f}  "
                  f"n_diverging={c['n_diverging']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
