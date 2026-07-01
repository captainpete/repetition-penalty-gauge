#!/usr/bin/env python3
"""Part B analyzer — reuses code/analyze_a2.py's EXACT metric definitions (clean-position
selection, threshold predicate g < z_top(1-1/theta), balanced accuracy / TPR / TNR, runner-up
fraction, structural-token tagging + code structural-flip rate) on the HumanEval-seeded run.
HumanEval is all Python, so the code-vs-prose domain control is not applicable (no prose seeds);
we report the code-side metrics + the overall flip fraction pooled over theta in [1.1,1.5].

  .venv/bin/python analyze_a2_humaneval.py --raw humaneval_raw.json [--raw-fix humaneval_fix_raw.json]
"""
import os, sys, json, argparse
sys.path.insert(0, "code")
import analyze_a2 as A  # exact constants + helpers

TEST_THETAS = A.TEST_THETAS   # (1.1,1.2,1.3,1.4,1.5)


def struct_flip_rate(records, domains=("python",)):
    """identical to analyze_a2.main's inner struct_flip_rate (per-record rate, then meaned)."""
    rates = []
    for r in records:
        if r["domain"] in domains and r["theta"] in TEST_THETAS:
            sf = sum(1 for p in r["positions"] if p["flip"] and p["structural"])
            rates.append(sf / max(1, len(r["positions"])))
    return rates


def analyze_raw(R):
    # ---- validity gate: theta=1.0 must have zero flips ----
    gate_flips = sum(p["flip"] for r in R if r["theta"] == 1.0 for p in r["positions"])
    gate_ok = (gate_flips == 0)

    clean_pred, clean_act = [], []
    runnerup_hits = runnerup_tot = 0
    zt_flip, zt_noflip = [], []
    flip_pool = pos_pool = 0            # overall flip fraction pooled theta in [1.1,1.5], code
    for r in R:
        if r["domain"] != "python" or r["theta"] not in TEST_THETAS:
            continue
        th = r["theta"]
        for p in r["positions"]:
            pos_pool += 1
            flip_pool += 1 if p["flip"] else 0
            if not (p["top1_seen"] and p["top1_pos"]):
                continue
            (zt_flip if p["flip"] else zt_noflip).append(p["z_top"])
            if p["top1_seen"] and p["top1_pos"] and not p["top2_seen"]:
                pred = p["gap"] < p["z_top"] * (1 - 1.0 / th)
                clean_pred.append(pred)
                clean_act.append(p["flip"])
                if p["flip"]:
                    runnerup_tot += 1
                    runnerup_hits += p["emitted_is_runnerup"]

    tp = sum(1 for a, b in zip(clean_pred, clean_act) if a and b)
    tn = sum(1 for a, b in zip(clean_pred, clean_act) if not a and not b)
    fp = sum(1 for a, b in zip(clean_pred, clean_act) if a and not b)
    fn = sum(1 for a, b in zip(clean_pred, clean_act) if not a and b)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    bal_acc = (tpr + tnr) / 2 if (tp + fn and tn + fp) else float("nan")
    runnerup_frac = runnerup_hits / runnerup_tot if runnerup_tot else float("nan")
    zt_lo, zt_hi = A.boot_ci_diff(zt_flip, zt_noflip) if zt_flip and zt_noflip else (0, 0)

    code_rates = struct_flip_rate(R)
    code_mean = sum(code_rates) / len(code_rates) if code_rates else 0.0

    return {"gate_ok": gate_ok, "gate_flips": gate_flips,
            "n_clean": len(clean_act), "n_clean_flips": sum(clean_act),
            "bal_acc": bal_acc, "tpr": tpr, "tnr": tnr,
            "runnerup_frac": runnerup_frac, "runnerup_tot": runnerup_tot,
            "zt_flip_mean": (sum(zt_flip)/len(zt_flip)) if zt_flip else None,
            "zt_noflip_mean": (sum(zt_noflip)/len(zt_noflip)) if zt_noflip else None,
            "zt_diff_ci": [zt_lo, zt_hi],
            "code_struct_flip_rate": code_mean,
            "overall_flip_fraction": (flip_pool / pos_pool) if pos_pool else float("nan"),
            "n_positions_pooled": pos_pool}


def valid_cliff(R, thetas):
    """descriptive: py_ok validity + bracket error vs theta (code)."""
    curve, be = {}, {}
    for th in thetas:
        vals = [1.0 if A.py_ok(r["prompt"] + r["gen_text"]) else 0.0
                for r in R if r["domain"] == "python" and r["theta"] == th]
        curve[th] = sum(vals) / len(vals) if vals else float("nan")
        bvals = [A.bracket_error(r["prompt"] + r["gen_text"])
                 for r in R if r["domain"] == "python" and r["theta"] == th]
        be[th] = sum(bvals) / len(bvals) if bvals else float("nan")
    return {"py_valid_rate": curve, "bracket_err": be}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/bench_a2/humaneval_raw.json")
    ap.add_argument("--raw-fix", default="results/bench_a2/humaneval_fix_raw.json")
    ap.add_argument("--out-dir", default="results/bench_a2")
    args = ap.parse_args()

    draw = json.load(open(args.raw))
    R = draw["records"]
    raw = analyze_raw(R)
    cliff = valid_cliff(R, draw["thetas"])

    fix = None
    if os.path.exists(args.raw_fix):
        dfix = json.load(open(args.raw_fix))
        fix = analyze_raw(dfix["records"])

    out = {"model": draw["model"], "revision": draw.get("revision"),
           "max_new": draw["max_new"], "thetas": draw["thetas"],
           "n_prompts": len({r["prompt_idx"] for r in R}),
           "raw": raw, "fix": fix, "cliff": cliff}
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.raw).replace("_raw.json", "").replace(".json", "")
    json.dump(out, open(os.path.join(args.out_dir, f"humaneval_summary_{tag}.json"), "w"), indent=2)

    print(f"=== {draw['model']} (rev {draw.get('revision')}) — HumanEval seeds ===")
    print(f"validity gate theta=1.0 -> {raw['gate_flips']} flips  [{'PASS' if raw['gate_ok'] else 'FAIL'}]")
    print(f"clean positions n={raw['n_clean']} (flips {raw['n_clean_flips']})")
    print(f"threshold formula: bal_acc {raw['bal_acc']:.4f}  TPR {raw['tpr']:.4f}  TNR {raw['tnr']:.4f}")
    print(f"runner-up frac: {raw['runnerup_frac']:.4f} (n={raw['runnerup_tot']})")
    print(f"code structural-flip rate (raw): {raw['code_struct_flip_rate']:.4f}")
    if fix:
        print(f"code structural-flip rate (fix): {fix['code_struct_flip_rate']:.4f}"
              f"  [gate {fix['gate_flips']} flips]")
    print(f"overall flip fraction pooled theta[1.1,1.5]: {raw['overall_flip_fraction']:.4f}"
          f" (n_pos={raw['n_positions_pooled']})")
    print("py-valid vs theta:", {f"{k:g}": round(v, 3) for k, v in cliff["py_valid_rate"].items()})


if __name__ == "__main__":
    main()
