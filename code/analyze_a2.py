#!/usr/bin/env python3
"""A2 analyzer — frozen rule in experiments/A2/PREREG.md.
Primary: does g < z_top(1-1/theta) predict the actual penalty-induced flip on clean
penalized positions? Plus confidence-first, domain control, and a descriptive validity
cliff. Reads runs/A2/raw.json -> runs/A2/{summary.json,REPORT.md}.
"""
import os, json, random, ast, argparse
from collections import defaultdict

CODE = ("json", "python")
TEST_THETAS = (1.1, 1.2, 1.3, 1.4, 1.5)
BAL_ACC_MIN = 0.90
RUNNERUP_MIN = 0.90
DOMAIN_RATIO_MIN = 5.0
BOOT = 10000


def boot_ci_diff(a, b, n=BOOT, seed=0):
    r = random.Random(seed)
    ma, mb = len(a), len(b)
    out = sorted((sum(a[r.randrange(ma)] for _ in range(ma)) / ma)
                 - (sum(b[r.randrange(mb)] for _ in range(mb)) / mb) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def bracket_error(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    st, bad = [], 0
    for ch in text:
        if ch in "([{":
            st.append(ch)
        elif ch in ")]}":
            if st and st[-1] == pairs[ch]:
                st.pop()
            else:
                bad += 1
    return bad + len(st)


def json_ok(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def py_ok(text):
    try:
        ast.parse(text)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="runs/A2/raw.json")
    ap.add_argument("--out-dir", default="runs/A2")
    args = ap.parse_args()
    d = json.load(open(args.raw))
    R = d["records"]

    # ---- validity gate: theta=1.0 must have zero flips ----
    gate_flips = sum(p["flip"] for r in R if r["theta"] == 1.0 for p in r["positions"])
    gate_ok = (gate_flips == 0)

    # ---- gather penalized positions on code prompts at test thetas ----
    clean_pred, clean_act = [], []     # clean slice: top1_seen & top1_pos & not top2_seen
    runnerup_hits = runnerup_tot = 0
    zt_flip, zt_noflip = [], []        # confidence-first (penalized positive top1)
    for r in R:
        if r["domain"] not in CODE or r["theta"] not in TEST_THETAS:
            continue
        th = r["theta"]
        for p in r["positions"]:
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

    # balanced accuracy of the formula on the clean slice
    tp = sum(1 for a, b in zip(clean_pred, clean_act) if a and b)
    tn = sum(1 for a, b in zip(clean_pred, clean_act) if not a and not b)
    fp = sum(1 for a, b in zip(clean_pred, clean_act) if a and not b)
    fn = sum(1 for a, b in zip(clean_pred, clean_act) if not a and b)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    bal_acc = (tpr + tnr) / 2 if (tp + fn and tn + fp) else float("nan")
    runnerup_frac = runnerup_hits / runnerup_tot if runnerup_tot else float("nan")

    zt_lo, zt_hi = boot_ci_diff(zt_flip, zt_noflip) if zt_flip and zt_noflip else (0, 0)
    conf_first = zt_lo > 0

    # ---- domain control: structural flips per generated token, code vs prose ----
    def struct_flip_rate(domains):
        rates = []
        for r in R:
            if r["domain"] in domains and r["theta"] in TEST_THETAS:
                sf = sum(1 for p in r["positions"] if p["flip"] and p["structural"])
                rates.append(sf / max(1, len(r["positions"])))
        return rates
    code_rates = struct_flip_rate(CODE)
    prose_rates = struct_flip_rate(("prose",))
    code_mean = sum(code_rates) / len(code_rates) if code_rates else 0.0
    prose_mean = sum(prose_rates) / len(prose_rates) if prose_rates else 0.0
    ratio = code_mean / prose_mean if prose_mean > 0 else float("inf")
    dlo, dhi = boot_ci_diff(code_rates, prose_rates) if code_rates and prose_rates else (0, 0)
    domain_ok = (ratio >= DOMAIN_RATIO_MIN) and (dlo > 0)

    # ---- supporting: validity vs theta (descriptive cliff) ----
    cliff = {}
    for dom, fn_ok in (("json", json_ok), ("python", py_ok)):
        curve = {}
        for th in d["thetas"]:
            vals = []
            for r in R:
                if r["domain"] == dom and r["theta"] == th:
                    full = r["prompt"] + r["gen_text"]
                    vals.append(1.0 if fn_ok(full) else 0.0)
            curve[th] = sum(vals) / len(vals) if vals else float("nan")
        be = {}
        for th in d["thetas"]:
            vals = [bracket_error(r["prompt"] + r["gen_text"])
                    for r in R if r["domain"] == dom and r["theta"] == th]
            be[th] = sum(vals) / len(vals) if vals else float("nan")
        cliff[dom] = {"valid_rate": curve, "bracket_err": be}

    confirmed = gate_ok and (bal_acc >= BAL_ACC_MIN) and (runnerup_frac >= RUNNERUP_MIN) \
        and conf_first and domain_ok

    # ---- report ----
    L = ["# A2 — Repetition penalty as a threshold attack on structured output — RESULT\n"]
    L.append(f"Model `{d['model']}` (rev `{d.get('revision')}`), greedy, "
             f"{d['max_new']} tokens, θ∈{d['thetas']}.\n")
    L.append(f"**Validity gate (θ=1.0 → 0 flips):** {'PASS' if gate_ok else 'FAIL ('+str(gate_flips)+' flips)'}\n")
    L.append("## 1. Threshold formula on clean penalized positions (primary)")
    L.append(f"clean positions n={len(clean_act)} | flips: {sum(clean_act)}")
    L.append(f"- formula `g < z_top(1−1/θ)` vs actual flip: balanced accuracy "
             f"**{bal_acc:.3f}** (TPR {tpr:.3f}, TNR {tnr:.3f}; threshold ≥ {BAL_ACC_MIN})")
    L.append(f"- emitted == pre-penalty runner-up when flipped: **{runnerup_frac:.3f}** "
             f"(n={runnerup_tot}; threshold ≥ {RUNNERUP_MIN})\n")
    L.append("## 2. Confidence-first")
    L.append(f"- mean z_top flipped {sum(zt_flip)/len(zt_flip):.2f} (n={len(zt_flip)}) vs "
             f"non-flipped {sum(zt_noflip)/len(zt_noflip):.2f} (n={len(zt_noflip)}); "
             f"diff 95% CI [{zt_lo:.2f}, {zt_hi:.2f}] → confident-first: **{conf_first}**\n")
    L.append("## 3. Domain specificity (control)")
    L.append(f"- structural-flip rate/token: code {code_mean:.4f} vs prose {prose_mean:.4f} "
             f"→ ratio **{ratio:.1f}×** (threshold ≥ {DOMAIN_RATIO_MIN}×; diff CI [{dlo:.4f},{dhi:.4f}]) "
             f"→ **{domain_ok}**\n")
    L.append("## 4. Validity vs θ (supporting, descriptive — not gating)")
    for dom in ("json", "python"):
        vr = cliff[dom]["valid_rate"]
        L.append(f"- {dom} valid-rate: " + ", ".join(f"θ{th:g}={vr[th]:.2f}" for th in d["thetas"]))
        be = cliff[dom]["bracket_err"]
        L.append(f"  {dom} bracket-err: " + ", ".join(f"θ{th:g}={be[th]:.1f}" for th in d["thetas"]))
    L.append("")
    L.append("## Verdict")
    if confirmed:
        L.append(f"**CONFIRMED.** The closed-form threshold predicts penalty-induced flips "
                 f"(bal-acc {bal_acc:.3f}), flips go to the runner-up ({runnerup_frac:.3f}), "
                 f"corruption is confidence-first, and structural damage is {ratio:.1f}× "
                 f"code-vs-prose. A 'modest' penalty is a predictable attack on syntax. "
                 f"Consensus falsified.")
    elif not gate_ok:
        L.append("**INVALID.** Validity gate failed (flips at θ=1.0) — harness bug, debug first.")
    else:
        fails = []
        if bal_acc < BAL_ACC_MIN: fails.append(f"formula bal-acc {bal_acc:.3f}<{BAL_ACC_MIN}")
        if runnerup_frac < RUNNERUP_MIN: fails.append(f"runner-up {runnerup_frac:.3f}<{RUNNERUP_MIN}")
        if not conf_first: fails.append("confidence-first CI includes 0")
        if not domain_ok: fails.append(f"domain ratio {ratio:.1f}<{DOMAIN_RATIO_MIN} or CI includes 0")
        L.append("**NOT CONFIRMED** — failed: " + "; ".join(fails))
    report = "\n".join(L) + "\n"

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "REPORT.md"), "w").write(report)
    json.dump({"model": d["model"], "revision": d.get("revision"),
               "gate_ok": gate_ok, "bal_acc": bal_acc, "tpr": tpr, "tnr": tnr,
               "runnerup_frac": runnerup_frac, "conf_first": conf_first,
               "zt_flip_mean": (sum(zt_flip)/len(zt_flip)) if zt_flip else None,
               "zt_noflip_mean": (sum(zt_noflip)/len(zt_noflip)) if zt_noflip else None,
               "code_struct_flip_rate": code_mean, "prose_struct_flip_rate": prose_mean,
               "domain_ratio": ratio, "domain_ok": domain_ok,
               "confirmed": bool(confirmed), "cliff": cliff},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(report)


if __name__ == "__main__":
    main()
