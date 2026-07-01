#!/usr/bin/env python3
"""A1 analyzer — do REAL checkpoints occupy different points on the penalty's sign boundary, and
is the A1 flip-rate at the NATURAL cross-model offset clearly > 0 (converting the title from assertion to
measurement)?  python analyze_a1_zeropoint.py --out-dir runs/A1_zeropoint"""
import os, json, glob, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/A1_zeropoint")
    args = ap.parse_args()
    raws = sorted(glob.glob(os.path.join(args.out_dir, "raw_*.json")))
    models = [json.load(open(r)) for r in raws]

    L = ["# A1 — real cross-checkpoint zero-point + natural-Δc flip-rate — RESULT\n",
         "Headline under test: `repetition_penalty=1.3` already denotes different interventions across "
         "REAL models, no synthetic shift required. Tests where each checkpoint sits on the penalty's "
         "sign boundary and the flip-rate at the measured natural offset.\n",
         "## (1) Zero-point per model (where it sits relative to the sign boundary)",
         "| model | frac seen-logit >0 (÷θ branch) | mean top-1 logit | median top-1 logit |",
         "|---|---|---|---|"]
    meds = []
    for d in models:
        z = d["zero_point"]; meds.append((d["model"], z["median_top1_logit"]))
        L.append(f"| {d['model'].split('/')[-1]} | {z['frac_seen_logit_positive']:.3f} | "
                 f"{z['mean_top1_logit']:.2f} | {z['median_top1_logit']:.2f} |")

    thetas = models[0]["thetas"]
    lo_m, hi_m = min(meds, key=lambda x: x[1]), max(meds, key=lambda x: x[1])
    spread = hi_m[1] - lo_m[1]                                # gauge offset between the most-different models
    fps = [d["zero_point"]["frac_seen_logit_positive"] for d in models]
    fp_spread = max(fps) - min(fps)
    L.append(f"\n## (2) Natural offset between real models")
    L.append(f"- median-top1-logit spans **{lo_m[1]:.2f}** ({lo_m[0].split('/')[-1]}) → **{hi_m[1]:.2f}** "
             f"({hi_m[0].split('/')[-1]}) = a **{spread:.1f}-logit** natural gauge spread")
    L.append(f"- fraction-of-seen-logit-positive spans {min(fps):.3f}→{max(fps):.3f} (spread **{fp_spread:.2f}**) "
             f"— at the SAME repetition_penalty, some checkpoints ÷θ almost nothing, others ÷θ much more")
    L.append(f"\n## Flip-rate: raw gauge vs each model's mean-centred (NATURAL) gauge, vs synthetic c=5")
    L.append("| θ | flip @ natural (raw vs −median) | flip @ synthetic c=5 |")
    L.append("|---|---|---|")
    nat_flips = []
    for th in thetas:
        fn = sum(d["flip_rate"][f"theta{th}_natural(-median)"] for d in models) / len(models)
        f5 = sum(d["flip_rate"][f"theta{th}_synth_5"] for d in models) / len(models)
        nat_flips.append(fn)
        L.append(f"| {th} | **{fn:.3f}** | {f5:.3f} |")

    headline_theta = max(thetas)
    nat_hi = sum(d["flip_rate"][f"theta{headline_theta}_natural(-median)"] for d in models) / len(models)
    # MEASURED iff real checkpoints branch differently (frac-positive spread clearly >0) AND the natural
    # (raw-vs-centred) gauge causes flips. Gate on contrasts (spreads, flip>0), not absolute magnitudes.
    measured = (fp_spread > 0.1) and (nat_hi > 0.02)
    L.append(f"\n## Verdict")
    if measured:
        L.append(f"**MEASURED (title holds).** Real checkpoints occupy materially different points on the "
                 f"penalty's sign boundary (frac-positive spread **{fp_spread:.2f}**, median-logit spread "
                 f"**{spread:.1f}**), so repetition_penalty={headline_theta} already applies a different "
                 f"intervention to each — no synthetic shift required. A model's own mean-centred gauge (a "
                 f"real gauge it could ship in) flips **{nat_hi:.1%}** of greedy tokens vs its raw gauge.")
        disp = "MEASURED"
    else:
        L.append(f"**SCOPE THE CLAIM.** Checkpoints sit at similar sign-boundary points (frac-positive "
                 f"spread {fp_spread:.2f}) and/or the natural-gauge flip-rate is low ({nat_hi:.1%}); soften "
                 f"the headline — the dramatic gauge effect needs offsets larger than real checkpoints show.")
        disp = "SCOPE"
    report = "\n".join(L) + "\n"
    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "REPORT.md"), "w").write(report)
    json.dump({"disposition": disp, "median_logit_spread": spread, "frac_pos_spread": fp_spread,
               "flip_at_natural_by_theta": dict(zip(map(str, thetas), nat_flips)),
               "zero_points": {d["model"]: d["zero_point"] for d in models}},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(report)


if __name__ == "__main__":
    main()
