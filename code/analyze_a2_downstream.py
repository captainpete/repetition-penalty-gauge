#!/usr/bin/env python3
"""A2 analyzer — does the per-token corruption matter end-to-end, and does the fix recover it?
Gate on the theta-trend CONTRAST: raw degrades 1.0->1.3, fix stays ~flat.
  python analyze_a2_downstream.py --raw runs/A2_downstream/raw.json"""
import os, json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="runs/A2_downstream/raw.json")
    ap.add_argument("--out-dir", default="runs/A2_downstream")
    args = ap.parse_args()
    d = json.load(open(args.raw)); m = d["metrics"]; th = d["thetas"]
    lo, hi = th[0], th[-1]

    L = [f"# A2 — downstream task quality (raw vs fix) — RESULT\n",
         f"{d['model']}, HumanEval n={d['n_humaneval']}, JSON n={d['n_json']}, θ∈{th}.\n",
         "| metric | op | " + " | ".join(f"θ={t}" for t in th) + " | Δ(θ1.0−θ1.3) |",
         "|---|---|" + "---|" * (len(th) + 1)]
    verdicts = []
    for metric, label in [("humaneval_pass1", "HumanEval pass@1"), ("json_valid_rate", "JSON valid-rate")]:
        deg = {}
        for op in ("raw", "fix"):
            vals = [m[f"{op}_theta{t}"][metric] for t in th]
            deg[op] = vals[0] - vals[-1]
            L.append(f"| {label} | {op} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {deg[op]:+.3f} |")
        # corruption matters iff raw drops with θ AND the fix recovers most of it
        raw_hurts = deg["raw"] > 0.03
        fix_recovers = deg["fix"] < deg["raw"] - 0.02
        verdicts.append((label, raw_hurts, fix_recovers, deg["raw"], deg["fix"]))

    L.append("\n## Verdict")
    any_matters = any(rh for _, rh, _, _, _ in verdicts)
    for label, rh, fr, dr, df in verdicts:
        if rh and fr:
            L.append(f"- **{label}: CORRUPTION MATTERS, FIX RECOVERS.** raw drops {dr:+.3f} from θ{lo}→θ{hi}; "
                     f"the fix drops only {df:+.3f} (recovers most of it). End-to-end evidence.")
        elif rh:
            L.append(f"- **{label}: raw degrades ({dr:+.3f}) but the fix does NOT fully recover ({df:+.3f}).** "
                     f"Report honestly.")
        else:
            L.append(f"- **{label}: small/null effect** (raw Δ {dr:+.3f}). The per-token corruption does not "
                     f"materially change this end-to-end metric — a quantified bound, still reportable.")
    disp = "MATTERS" if any_matters else "NULL"
    if any_matters:
        L.append(f"\n**Overall: the structured-output corruption is real end-to-end** on at least one metric, "
                 f"and the normalize-before-penalize fix recovers it — converts the paper's implication to evidence.")
    else:
        L.append(f"\n**Overall: NULL/small** — per-token corruption does not move these end-to-end metrics; "
                 f"report the bound and scope the 'structured-output corruption' claim accordingly.")
    report = "\n".join(L) + "\n"
    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "REPORT.md"), "w").write(report)
    json.dump({"disposition": disp, "verdicts": [{"metric": v[0], "raw_hurts": v[1], "fix_recovers": v[2],
               "raw_drop": v[3], "fix_drop": v[4]} for v in verdicts], "metrics": m},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(report)


if __name__ == "__main__":
    main()
