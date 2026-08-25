#!/usr/bin/env python3
"""ZPREACH analyzer (rep-penalty ZPREACH) -- applies the FROZEN decision rule in PREREG.md Sec 5.

Assembles the n=7 per-model table from data that already exists on disk (no model loads, no GPU, no
new decode):
  (i)   zero-point           <- the a1_zeropoint run raws   (zero_point.frac_seen_logit_positive, median_top1_logit)
  (ii)  raw steepness        <- FIXCAL raws  (raw_grid[1.00].rep_rate - raw_grid[1.05].rep_rate)
  (iii) reachability @ 1.1   <- FIXCAL raws  (bisection.rep_rate entry with theta_raw == 1.1)

...then adjudicates P1/P2/P3 by ORDERING CONTRASTS (concordant vs discordant pairs), reports Spearman
rho as a DESCRIPTIVE statistic only (hand-rolled, no scipy), runs the mandatory within-group check that
the harness TRIAGE power caveat requires, sets effective_n / family_confounded, and fires the frozen
disposition. Missing models are handled gracefully -> PENDING.

  python3 code/analyze_zpreach.py \
      --zeropoint-dirs results/a1_zeropoint \
      --fixcal-dir results/fix_calibration --out-dir results/zeropoint_reachability
"""
import os, json, glob, math, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # code/ -> repo root

# ---- frozen cohort (PREREG Sec 3), in a stable canonical order
COHORT = ["gpt2", "gpt2-large", "pythia-2.8b", "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
          "Qwen2.5-Coder-7B", "starcoder2-7b"]
FAMILY = {"gpt2": "gpt2", "gpt2-large": "gpt2", "pythia-2.8b": "pythia",
          "Qwen2.5-7B": "qwen", "Qwen2.5-7B-Instruct": "qwen", "Qwen2.5-Coder-7B": "qwen",
          "starcoder2-7b": "starcoder"}

DEPLOYED_ANCHOR = 1.1          # Ollama / GPT4All default -- the anchor ZPREACH names
STEEP_LO, STEEP_HI = 1.0, 1.05  # steepness = rep_rate(1.0) - rep_rate(1.05)
PRIMARY = "rep_rate"           # FIXCAL's pre-registered primary suppression metric
EXTREME_CUT = 0.5              # PREREG Sec 3b: frac_seen_logit_positive < 0.5 == multiply-branch dominant
TOL = 1e-9

DEFAULT_ZP_DIRS = ",".join([os.path.join(REPO, "runs", "A1_zeropoint"),
                            "results/a1_zeropoint",
                            os.path.expanduser("~/paper-repetition-penalty/results/a1_zeropoint")])


# ----------------------------------------------------------------------------- helpers
def frozen_rule():
    """Sec 5 of PREREG.md, read VERBATIM from the file next to this analyzer (no hand-synced copy)."""
    p = os.path.join(HERE, "PREREG.md")
    try:
        txt = open(p).read()
        s = txt.index("\n", txt.index("## 5. FROZEN DECISION RULE")) + 1
        return txt[s:txt.index("\n## ", s)].strip()
    except Exception as e:                                          # pragma: no cover
        return f"(could not read PREREG.md Sec 5 next to the analyzer: {e})"


def slug(model_field):
    """Model identity = last path component of the JSON's `model` field (never the filename:
    'Qwen2.5-7B' is a prefix of 'Qwen2.5-7B-Instruct')."""
    return str(model_field).split("/")[-1]


def fnum(x, p=3, inf="inf"):
    if x is None:
        return "--"
    if isinstance(x, float):
        if math.isinf(x):
            return inf
        return f"{x:.{p}f}"
    return str(x)


def midranks(vals):
    """Ascending midranks (ties share the mean of the ranks they span). rank 1 = smallest.
    Values may include +inf (all +inf tie at the top). PREREG Sec 3a."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = mid
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None                                   # degenerate (all tied on one axis)
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    """Hand-rolled tie-correct Spearman: Pearson on midranks. DESCRIPTIVE ONLY (frozen rule 1)."""
    if len(x) < 2:
        return None
    return pearson(midranks(x), midranks(y))


def concordance(x, y):
    """Frozen rule 2: count concordant / discordant pairs against the predicted POSITIVE direction.
    Pairs tied on either variable are excluded and counted as T. HOLDS iff C > D."""
    n, C, D, T = len(x), 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or y[i] == y[j]:
                T += 1
                continue
            if (x[i] < x[j]) == (y[i] < y[j]):
                C += 1
            else:
                D += 1
    comp = (C + D) > 0
    return {"C": C, "D": D, "T": T, "n": n, "computable": comp,
            "holds": bool(comp and C > D), "strict_monotone": bool(comp and D == 0)}


def median_low(vals):
    """Lower median -- order statistic at index ceil(n/2)-1. Well-defined when some values are +inf."""
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, math.ceil(len(s) / 2) - 1)]


def rank_sides(models, vals):
    """PREREG Sec 4: 'side' is computed on MIDRANKS, not values -- reach_score is censored at +inf, so a
    value-median can itself be +inf and the P3 clause would pass unconditionally (a gate that cannot
    fail is not a gate). Ranks stay discriminating under censoring."""
    r = midranks(vals)
    med = median_low(r)
    sides = {m: ("high" if rr > med else "low") for m, rr in zip(models, r)}
    return sides, dict(zip(models, r)), med, bool(len(set(r)) <= 1)


def jsonable(o):
    """Strict-JSON-safe: +/-inf -> the string '+inf'/'-inf' (json.dump would emit bare Infinity, which
    is not valid JSON for non-Python readers)."""
    if isinstance(o, float):
        if math.isinf(o):
            return "+inf" if o > 0 else "-inf"
        if math.isnan(o):
            return None
        return o
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def grid_key(grid, theta):
    for k in grid:
        try:
            if abs(float(k) - theta) < TOL:
                return k
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------------------- loading
def load_zeropoints(dirs):
    """(i) the a1_zeropoint run zero-points. First directory supplying a model WINS (PREREG Sec 3); a later directory
    disagreeing on the numbers is recorded and makes the run INVALID."""
    zp, disagree, dirs_used, dirs_missing = {}, [], [], []
    for d in dirs:
        if not os.path.isdir(d):
            dirs_missing.append(d)
            continue
        found = 0
        for p in sorted(glob.glob(os.path.join(d, "raw_*.json"))):
            try:
                j = json.load(open(p))
            except Exception as e:
                disagree.append({"path": p, "error": f"unreadable: {e}"})
                continue
            if "zero_point" not in j or "model" not in j:
                continue
            s, z = slug(j["model"]), j["zero_point"]
            rec = {"frac_seen_logit_positive": z.get("frac_seen_logit_positive"),
                   "median_top1_logit": z.get("median_top1_logit"),
                   "mean_top1_logit": z.get("mean_top1_logit"),
                   "revision": j.get("revision"), "source": p}
            found += 1
            if s not in zp:
                zp[s] = rec
            else:
                a, b = zp[s], rec
                for f in ("frac_seen_logit_positive", "median_top1_logit"):
                    if a[f] is None or b[f] is None or abs(a[f] - b[f]) > 1e-9:
                        disagree.append({"model": s, "field": f, "kept": a[f], "kept_from": a["source"],
                                         "other": b[f], "other_from": b["source"]})
        dirs_used.append({"dir": d, "raws_with_zero_point": found})
    return zp, disagree, dirs_used, dirs_missing


def load_fixcal(d):
    """(ii) raw steepness + (iii) reachability at theta_raw = 1.1, from the FIXCAL raws."""
    fc, notes = {}, []
    if not os.path.isdir(d):
        return fc, [f"fixcal dir does not exist: {d}"]
    for p in sorted(glob.glob(os.path.join(d, "raw_*.json"))):
        try:
            j = json.load(open(p))
        except Exception as e:
            notes.append(f"unreadable {p}: {e}")
            continue
        if "raw_grid" not in j or "bisection" not in j or "model" not in j:
            continue
        s = slug(j["model"])
        rec = {"source": p, "revision": j.get("revision"), "theta_max": j.get("theta_max"),
               "steepness": None, "rep_rate_1.00": None, "rep_rate_1.05": None,
               "reach_verdict": None, "reach_theta_fix": None, "reach_ratio": None,
               "reach_achieved": None, "reach_target": None, "reach_shortfall": None,
               "reach_raw_suppresses": None, "reach_converged": None, "reach_score": None,
               "controls_ok": None, "control_detail": {}}

        # --- (ii) steepness
        g = j["raw_grid"]
        klo, khi = grid_key(g, STEEP_LO), grid_key(g, STEEP_HI)
        if klo and khi and PRIMARY in g[klo] and PRIMARY in g[khi]:
            rec["rep_rate_1.00"], rec["rep_rate_1.05"] = g[klo][PRIMARY], g[khi][PRIMARY]
            rec["steepness"] = g[klo][PRIMARY] - g[khi][PRIMARY]
        else:
            notes.append(f"{s}: raw_grid lacks theta={STEEP_LO}/{STEEP_HI} {PRIMARY} -- steepness missing")

        # --- (iii) reachability at the deployed anchor
        row = None
        for r in j["bisection"].get(PRIMARY, []):
            if abs(float(r.get("theta_raw", -1)) - DEPLOYED_ANCHOR) < TOL:
                row = r
                break
        if row is None:
            notes.append(f"{s}: no bisection.{PRIMARY} entry at theta_raw={DEPLOYED_ANCHOR} "
                         f"-- reachability missing")
        else:
            v = row.get("verdict")
            rec.update(reach_verdict=v, reach_theta_fix=row.get("theta_fix"),
                       reach_achieved=row.get("achieved"), reach_target=row.get("target"),
                       reach_shortfall=row.get("shortfall"),
                       reach_raw_suppresses=row.get("raw_suppresses"),
                       reach_converged=row.get("converged"))
            # PREREG Sec 3a: theta' if MATCHED, 1.0 if TRIVIAL, +inf if UNREACHABLE. LOWER = more reachable.
            if v == "MATCHED" and row.get("theta_fix"):
                rec["reach_score"] = float(row["theta_fix"])
                rec["reach_ratio"] = float(row["theta_fix"]) / DEPLOYED_ANCHOR
            elif v == "TRIVIAL":
                rec["reach_score"] = 1.0
                rec["reach_ratio"] = 1.0 / DEPLOYED_ANCHOR
            elif v == "UNREACHABLE":
                rec["reach_score"] = float("inf")
            else:
                notes.append(f"{s}: unrecognised verdict {v!r} at theta_raw={DEPLOYED_ANCHOR}")

        # --- FIXCAL's own mandatory controls (frozen rule 0: a failed no-op => INVALID)
        c = j.get("controls", {})
        rec["control_detail"] = {"noop_raw1_eq_fix1": c.get("noop_raw1_eq_fix1"),
                                 "instrumented_eq_raw1": c.get("instrumented_eq_raw1")}
        if rec["control_detail"]["noop_raw1_eq_fix1"] is not None:
            rec["controls_ok"] = bool(c.get("noop_raw1_eq_fix1")) and bool(c.get("instrumented_eq_raw1"))
        fc[s] = rec
    return fc, notes


# ----------------------------------------------------------------------------- adjudication
def ordering(models, rows, xf, yf):
    """C/D/T + descriptive rho for one prediction over a set of models (predicted direction positive)."""
    ms = [m for m in models
          if rows[m].get(xf) is not None and rows[m].get(yf) is not None]
    x = [rows[m][xf] for m in ms]
    y = [rows[m][yf] for m in ms]
    out = concordance(x, y) if len(ms) >= 2 else {"C": 0, "D": 0, "T": 0, "n": len(ms),
                                                  "computable": False, "holds": False,
                                                  "strict_monotone": False}
    out["models"] = ms
    out["rho_descriptive"] = spearman(x, y) if len(ms) >= 2 else None
    out["x_ranks"] = dict(zip(ms, midranks(x))) if ms else {}
    out["y_ranks"] = dict(zip(ms, midranks(y))) if ms else {}
    # ordering of models by the predictor, ascending -- the "monotone ordering" the request gates on
    out["by_predictor"] = [m for _, m in sorted(zip(x, ms), key=lambda t: t[0])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeropoint-dirs", default=DEFAULT_ZP_DIRS,
                    help="comma-separated; FIRST dir supplying a model wins (PREREG Sec 3)")
    ap.add_argument("--fixcal-dir", default=os.path.join(REPO, "runs", "FIXCAL"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "runs", "ZPREACH"))
    args = ap.parse_args()

    zp_dirs = [d for d in (s.strip() for s in args.zeropoint_dirs.split(",")) if d]
    # de-duplicate while preserving precedence order (the default list carries /work + repo aliases,
    # which are the SAME directory inside the lab container -- realpath collapses them)
    seen_d, dedup = set(), []
    for d in zp_dirs:
        rp = os.path.realpath(d)
        if rp not in seen_d:
            seen_d.add(rp)
            dedup.append(d)
    zp_dirs = dedup
    fixcal_dir = args.fixcal_dir
    if not os.path.isdir(fixcal_dir) and os.path.isdir("results/fix_calibration"):
        fixcal_dir = "results/fix_calibration"

    zp, disagree, zp_dirs_used, zp_dirs_missing = load_zeropoints(zp_dirs)
    fc, fc_notes = load_fixcal(fixcal_dir)

    # ---- assemble the frozen-cohort table
    rows, missing = {}, {}
    for m in COHORT:
        z, f = zp.get(m), fc.get(m)
        fp = z["frac_seen_logit_positive"] if z else None
        r = {"model": m, "family": FAMILY[m],
             "frac_pos": fp,
             "frac_neg": (None if fp is None else 1.0 - fp),
             "median_top1_logit": z["median_top1_logit"] if z else None,
             "group": (None if fp is None else ("EXTREME" if fp < EXTREME_CUT else "NON-EXTREME")),
             "zp_source": z["source"] if z else None,
             "fixcal_source": f["source"] if f else None}
        for k in ("steepness", "rep_rate_1.00", "rep_rate_1.05", "reach_verdict", "reach_theta_fix",
                  "reach_ratio", "reach_achieved", "reach_target", "reach_shortfall",
                  "reach_raw_suppresses", "reach_converged", "reach_score", "controls_ok",
                  "control_detail"):
            r[k] = f[k] if f else None
        lack = []
        if r["frac_pos"] is None:
            lack.append("(i) zero-point")
        if r["steepness"] is None:
            lack.append("(ii) raw steepness")
        if r["reach_score"] is None:
            lack.append("(iii) reachability@1.1")
        if lack:
            missing[m] = lack
        r["complete"] = not lack
        rows[m] = r

    present = [m for m in COHORT if rows[m]["complete"]]
    extras = sorted((set(zp) | set(fc)) - set(COHORT))

    # ---- validity (frozen rule 0)
    ctl_fail = [m for m in COHORT if rows[m]["controls_ok"] is False]
    invalid = bool(disagree) or bool(ctl_fail)

    # ---- orderings: overall (frozen rule 2) -- P1 and P2
    P1 = ordering(present, rows, "frac_neg", "steepness")
    P2 = ordering(present, rows, "steepness", "reach_score")

    # ---- P3 (frozen rule 3), on the cohort actually present; sides on MIDRANKS (PREREG Sec 4)
    g2, g2l = rows["gpt2"], rows["gpt2-large"]
    computable = bool("gpt2" in present and "gpt2-large" in present)
    if present:
        s_side, s_rank, s_med, s_deg = rank_sides(present, [rows[m]["steepness"] for m in present])
        r_side, r_rank, r_med, r_deg = rank_sides(present, [rows[m]["reach_score"] for m in present])
    else:
        s_side = r_side = s_rank = r_rank = {}
        s_med = r_med = None
        s_deg = r_deg = True
    P3 = {"computable": computable, "n_cohort_present": len(present),
          "median_low_steepness_rank": s_med, "median_low_reach_score_rank": r_med,
          "steepness_degenerate": s_deg, "reach_score_degenerate": r_deg,
          "gpt2": {"steepness": g2["steepness"], "steepness_rank": s_rank.get("gpt2"),
                   "steepness_side": s_side.get("gpt2"), "reach_verdict": g2["reach_verdict"],
                   "reach_score": g2["reach_score"], "reach_score_rank": r_rank.get("gpt2"),
                   "reach_score_side": r_side.get("gpt2")},
          "gpt2-large": {"steepness": g2l["steepness"], "steepness_rank": s_rank.get("gpt2-large"),
                         "steepness_side": s_side.get("gpt2-large"),
                         "reach_verdict": g2l["reach_verdict"], "reach_score": g2l["reach_score"],
                         "reach_score_rank": r_rank.get("gpt2-large"),
                         "reach_score_side": r_side.get("gpt2-large")}}
    P3["P3b_same_side_steepness"] = bool(computable and P3["gpt2"]["steepness_side"] is not None and
                                         P3["gpt2"]["steepness_side"] == P3["gpt2-large"]["steepness_side"])
    P3["P3c_same_side_reach"] = bool(computable and P3["gpt2"]["reach_score_side"] is not None and
                                     P3["gpt2"]["reach_score_side"] == P3["gpt2-large"]["reach_score_side"])
    P3["holds"] = bool(computable and P3["P3b_same_side_steepness"] and P3["P3c_same_side_reach"])
    # P3a: premise check only -- REPORTED, NOT A GATE (PREREG Sec 4 / D1)
    P3["P3a_premise_gpt2large_extreme"] = (None if g2l["frac_pos"] is None
                                           else bool(g2l["frac_pos"] < EXTREME_CUT))
    P3["P3a_frac_pos_gpt2large"] = g2l["frac_pos"]

    # ---- within-group orderings (frozen rule 4)
    groups = {}
    for g in ("EXTREME", "NON-EXTREME"):
        ms = [m for m in present if rows[m]["group"] == g]
        groups[g] = {"models": ms, "n": len(ms), "eligible": len(ms) >= 3,
                     "P1": ordering(ms, rows, "frac_neg", "steepness") if len(ms) >= 2 else None,
                     "P2": ordering(ms, rows, "steepness", "reach_score") if len(ms) >= 2 else None}
    eligible = [g for g in groups if groups[g]["eligible"]]
    largest = max(eligible, key=lambda g: (groups[g]["n"], g)) if eligible else None
    if largest:
        lp1, lp2 = groups[largest]["P1"], groups[largest]["P2"]
        within_computable = bool(lp1 and lp2 and lp1["computable"] and lp2["computable"])
        within_holds = bool(within_computable and lp1["holds"] and lp2["holds"])
    else:
        within_computable = within_holds = False
    within = {"split_rule": f"EXTREME = frac_seen_logit_positive < {EXTREME_CUT} (multiply-branch "
                            f"dominant) vs NON-EXTREME >= {EXTREME_CUT}",
              "groups": groups, "largest_eligible_group": largest,
              "computable": within_computable, "holds": within_holds}

    # ---- family diagnostic (never a gate): does the split reproduce family membership?
    fam_groups = {}
    for f in sorted({FAMILY[m] for m in present}):
        ms = [m for m in present if FAMILY[m] == f]
        fam_groups[f] = {"models": ms, "n": len(ms),
                         "P1": ordering(ms, rows, "frac_neg", "steepness") if len(ms) >= 2 else None,
                         "P2": ordering(ms, rows, "steepness", "reach_score") if len(ms) >= 2 else None}

    # ---- gate 4: family confound (the TRIAGE power caveat, as a gate)
    overall_holds = bool(P1["holds"] and P2["holds"])
    family_confounded = bool(overall_holds and not within_holds)
    n_groups = len([g for g in groups if groups[g]["n"] > 0])
    effective_n = (n_groups if family_confounded
                   else (groups[largest]["n"] if (within_holds and largest) else len(present)))

    # ---- disposition (frozen rule 6)
    if invalid:
        disposition, branch = "INVALID", "input self-inconsistent -- debug, do not interpret"
    elif len(present) < len(COHORT):
        disposition, branch = "PENDING", "cohort incomplete -- PROVISIONAL numbers only, do not cite"
    elif overall_holds and P3["holds"] and within_holds and not family_confounded:
        disposition, branch = "MECHANISM-SUPPORTED", "frozen rule 6 -- gauge-artifact reading may be asserted (scoped)"
    else:
        disposition, branch = "MECHANISM-NOT-ESTABLISHED", "frozen rule 7 -- failure branch"

    summ = {
        "experiment": "ZPREACH", "req": "ZPREACH",
        "question": "Does the post-hoc gauge-artifact reading (zero-point extremity -> raw steepness -> "
                    "fix un-reachability at theta_raw=1.1) survive a pre-registered ordering test at n=7?",
        "disposition": disposition, "branch": branch,
        "cohort": COHORT, "present": present, "missing": missing,
        "n_present": len(present), "n_cohort": len(COHORT),
        "effective_n": effective_n, "family_confounded": family_confounded,
        "reachability_score_definition": (
            "reach_score = theta' if MATCHED at theta_raw=1.1 (rep_rate) | 1.0 if TRIVIAL | +inf if "
            "UNREACHABLE. LOWER = MORE reachable. Ranked by ascending midranks (rank 1 = most "
            "reachable); all UNREACHABLE models tie at +inf in one midrank block and are never "
            "separated by their shortfalls; pairs tied on either variable are excluded from C/D and "
            "counted as T."),
        "group_split_rule": within["split_rule"],
        "steepness_definition": (f"raw_grid['{STEEP_LO:.9f}'].{PRIMARY} - "
                                 f"raw_grid['{STEEP_HI:.9f}'].{PRIMARY}"),
        "deployed_anchor": DEPLOYED_ANCHOR,
        "validity": {"invalid": invalid, "duplicate_disagreements": disagree,
                     "fixcal_control_failures": ctl_fail,
                     "controls_by_model": {m: rows[m]["control_detail"] for m in COHORT
                                           if rows[m]["control_detail"]}},
        "models": rows,
        "P1": {"statement": "raw steepness increases with zero-point extremity (frac-seen-logit-NEGATIVE "
                            "-- multiply-branch dominance)",
               "x": "frac_neg", "y": "steepness", "direction": "positive", **P1},
        "P2": {"statement": "reachability at 1.1 decreases with raw steepness (i.e. the matching-cost "
                            "score increases with steepness)",
               "x": "steepness", "y": "reach_score", "direction": "positive", **P2},
        "P3": {"statement": "gpt2-large patterns with gpt2 (both extreme zero-points per the a1_zeropoint run)", **P3},
        "within_group": within,
        "family_diagnostic": fam_groups,
        "inputs": {"zeropoint_dirs_used": zp_dirs_used, "zeropoint_dirs_absent": zp_dirs_missing,
                   "fixcal_dir": fixcal_dir, "fixcal_notes": fc_notes,
                   "models_outside_cohort_seen": extras},
        "frozen_rule_verbatim": frozen_rule(),
    }

    # ------------------------------------------------------------------ REPORT.md
    prov = " (PROVISIONAL -- cohort incomplete, not citable)" if disposition == "PENDING" else ""
    L = [f"# ZPREACH — zero-point → raw-steepness → fix-reachability — "
         f"**{disposition}**{prov}", "",
         "Frozen test of FIXCAL's surviving *post-hoc* reading: raw's deployed-band suppression strength "
         "is a **gauge artifact** of an extreme logit zero-point, and fix-reachability is limited by "
         "**how hard RAW drives the metric**, not by how weak the fix is. Assembly only — no new decode. "
         "Rule frozen in `PREREG.md` §5 (quoted verbatim at the end).", "",
         f"- deployed anchor: **θ_raw = {DEPLOYED_ANCHOR:g}** (Ollama / GPT4All default), primary metric "
         f"`{PRIMARY}`",
         f"- steepness: `raw_grid[{STEEP_LO:.2f}].{PRIMARY} − raw_grid[{STEEP_HI:.2f}].{PRIMARY}`",
         f"- reachability score: {summ['reachability_score_definition']}",
         f"- group split: {within['split_rule']}", ""]

    # presence
    L += ["## Cohort presence (frozen n=7)", "",
          "| model | (i) zero-point | (ii) raw steepness | (iii) reach @1.1 | complete |",
          "|---|:--:|:--:|:--:|:--:|"]
    for m in COHORT:
        r = rows[m]
        L.append(f"| `{m}` | {'yes' if r['frac_pos'] is not None else '**MISSING**'} | "
                 f"{'yes' if r['steepness'] is not None else '**MISSING**'} | "
                 f"{'yes' if r['reach_score'] is not None else '**MISSING**'} | "
                 f"{'**yes**' if r['complete'] else 'no'} |")
    L.append("")
    L.append(f"**{len(present)} / {len(COHORT)} complete.**"
             + ("" if not missing else "  Still awaiting: "
                + "; ".join(f"`{m}` → {', '.join(v)}" for m, v in missing.items())))
    if extras:
        L.append(f"(Raws seen for models outside the frozen cohort, ignored: {', '.join(extras)}.)")
    for n in fc_notes:
        L.append(f"- note: {n}")

    # per-model table
    L += ["", "## Per-model table (sorted by zero-point extremity, most extreme first)", "",
          "| model | family | group | frac-pos | **frac-neg** (P1 x) | median top-1 logit | "
          "rep_rate θ=1.0 | rep_rate θ=1.05 | **steepness** | verdict @1.1 | θ′ | θ′/θ_raw | "
          "achieved vs target (shortfall) | **reach_score** |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|:--:|--:|--:|---|--:|"]
    order_tbl = sorted(COHORT, key=lambda m: (rows[m]["frac_neg"] is None,
                                              -(rows[m]["frac_neg"] or 0.0), m))
    for m in order_tbl:
        r = rows[m]
        if r["reach_verdict"] == "UNREACHABLE":
            ach = (f"{fnum(r['reach_achieved'],4)} vs {fnum(r['reach_target'],4)} "
                   f"(short {fnum(r['reach_shortfall'],4)})")
        elif r["reach_verdict"] in ("MATCHED", "TRIVIAL"):
            ach = f"achieved {fnum(r['reach_achieved'],4)} ≈ target {fnum(r['reach_target'],4)}"
        else:
            ach = "--"
        L.append(f"| `{m}` | {r['family']} | {r['group'] or '--'} | {fnum(r['frac_pos'])} | "
                 f"**{fnum(r['frac_neg'])}** | {fnum(r['median_top1_logit'],2)} | "
                 f"{fnum(r['rep_rate_1.00'],4)} | {fnum(r['rep_rate_1.05'],4)} | "
                 f"**{fnum(r['steepness'],4)}** | {r['reach_verdict'] or '--'} | "
                 f"{fnum(r['reach_theta_fix'])} | {fnum(r['reach_ratio'])} | {ach} | "
                 f"**{fnum(r['reach_score'],3, inf='∞')}** |")
    L += ["", "*(`reach_score`: lower = more reachable; ∞ = UNREACHABLE at θ′≤θ′_max, ranked worst and "
              "tied with every other UNREACHABLE model. `median top-1 logit` is a reported diagnostic "
              "only — PROTOCOL §4 forbids gating on an absolute logit.)*"]

    def ord_block(tag, o, pred, xname, yname):
        if not o or not o["models"]:
            return [f"### {tag} — not computable (no model has both {xname} and {yname})"]
        rho = o["rho_descriptive"]
        b = [f"### {tag} — {pred}", "",
             f"- models used (n={o['n']}): {', '.join('`'+m+'`' for m in o['models'])}",
             f"- ordering by `{xname}` ascending: "
             + " < ".join(f"`{m}`" for m in o["by_predictor"]),
             f"- pairs: **C={o['C']}** concordant, **D={o['D']}** discordant, T={o['T']} tied "
             f"(excluded)",
             f"- **ordering {'HOLDS' if o['holds'] else ('FAILS' if o['computable'] else 'NOT COMPUTABLE')}**"
             f" (gate: C > D)"
             + ("  ·  strictly monotone (D=0), reported descriptively" if o["strict_monotone"] else ""),
             f"- Spearman ρ = **{fnum(rho,3)}** — *DESCRIPTIVE ONLY; no gate reads it* (frozen rule 1)"]
        return b

    L += ["", "## P1 — zero-point extremity → raw steepness", ""]
    L += ord_block("P1 (overall, all present models)", P1,
                   "raw steepness increases with frac-seen-logit-NEGATIVE", "frac_neg", "steepness")
    L += ["", "## P2 — raw steepness → un-reachability at θ_raw=1.1", ""]
    L += ord_block("P2 (overall, all present models)", P2,
                   "the matching cost at 1.1 increases with raw steepness (= reachability decreases)",
                   "steepness", "reach_score")

    # P3
    L += ["", "## P3 — does gpt2-large pattern with gpt2?", ""]
    if not P3["computable"]:
        L += ["**Not computable** — gpt2 and/or gpt2-large is not yet complete."]
    else:
        L += ["Sides are computed on **midranks**, not values (PREREG §4): `reach_score` is censored at "
              "+∞, so a value-median could itself be +∞ and the clause would pass unconditionally.",
              "",
              f"- cohort lower-median rank: steepness {fnum(P3['median_low_steepness_rank'],1)}, "
              f"reach_score {fnum(P3['median_low_reach_score_rank'],1)} (n={P3['n_cohort_present']})",
              f"- `gpt2`: steepness {fnum(P3['gpt2']['steepness'],4)} (rank "
              f"{fnum(P3['gpt2']['steepness_rank'],1)}) → **{P3['gpt2']['steepness_side']}**; "
              f"@1.1 {P3['gpt2']['reach_verdict']} score {fnum(P3['gpt2']['reach_score'],3,inf='∞')} "
              f"(rank {fnum(P3['gpt2']['reach_score_rank'],1)}) → **{P3['gpt2']['reach_score_side']}**",
              f"- `gpt2-large`: steepness {fnum(P3['gpt2-large']['steepness'],4)} (rank "
              f"{fnum(P3['gpt2-large']['steepness_rank'],1)}) → "
              f"**{P3['gpt2-large']['steepness_side']}**; @1.1 {P3['gpt2-large']['reach_verdict']} score "
              f"{fnum(P3['gpt2-large']['reach_score'],3,inf='∞')} (rank "
              f"{fnum(P3['gpt2-large']['reach_score_rank'],1)}) → "
              f"**{P3['gpt2-large']['reach_score_side']}**",
              f"- **P3b (same side on steepness): {'HOLDS' if P3['P3b_same_side_steepness'] else 'FAILS'}**"
              + ("  ⚠ *degenerate: every present model is tied on steepness — this clause carries no "
                 "information*" if P3["steepness_degenerate"] else ""),
              f"- **P3c (same side on reach_score): {'HOLDS' if P3['P3c_same_side_reach'] else 'FAILS'}**"
              + ("  ⚠ *degenerate: every present model is tied on reach_score — this clause carries no "
                 "information*" if P3["reach_score_degenerate"] else ""),
              f"- **P3 {'HOLDS' if P3['holds'] else 'FAILS'}** (both clauses required — frozen rule 3)"]
    p3a = P3["P3a_premise_gpt2large_extreme"]
    L += ["",
          f"**P3a premise check (REPORTED, NOT A GATE):** is gpt2-large's zero-point actually extreme "
          f"(frac-positive < {EXTREME_CUT})? measured frac-positive = "
          f"{fnum(P3['P3a_frac_pos_gpt2large'])} → **{'yes' if p3a else 'NO' if p3a is not None else '--'}**. "
          + ("The request's parenthetical *\"(both extreme zero-points per the a1_zeropoint run)\"* is **contradicted by "
             "the a1_zeropoint run's own table**, which was pre-registered as an expected failure in PREREG §4. Per "
             "PREREG §5-D1 this does **not** rescue a P3 failure: the premise was falsifiable before this "
             "analysis ran, so it cannot be discovered from these results and cannot be used to re-cut "
             "them." if p3a is False else "")]

    # within-group
    L += ["", "## Within-group ordering (frozen rule 4 — the TRIAGE power caveat, as a gate)", "",
          f"Split: {within['split_rule']}", ""]
    for g in ("EXTREME", "NON-EXTREME"):
        G = groups[g]
        L += [f"**{g}** (n={G['n']}): {', '.join('`'+m+'`' for m in G['models']) or '—'}"
              + ("" if G["eligible"] else "  → **<3 models: not computable, does NOT satisfy the gate**")]
        for tag, key, xn, yn in (("P1", "P1", "frac_neg", "steepness"),
                                 ("P2", "P2", "steepness", "reach_score")):
            o = G[key]
            if not o or not o["computable"]:
                L.append(f"  - {tag} within {g}: **not computable** "
                         f"({'n<2' if not o else 'every pair tied on one axis'})")
            else:
                L.append(f"  - {tag} within {g}: C={o['C']} / D={o['D']} / T={o['T']} → "
                         f"**{'HOLDS' if o['holds'] else 'FAILS'}**  ·  ρ = {fnum(o['rho_descriptive'],3)} "
                         f"(descriptive)")
        L.append("")
    L += [f"Largest eligible group: **{largest or 'none'}**  ·  within-group check "
          f"**{'HOLDS' if within_holds else ('FAILS' if within_computable else 'NOT COMPUTABLE')}**.", ""]

    # family diagnostic
    L += ["### Family diagnostic (reported, never a gate)", "",
          "| family | models | P1 within-family | P2 within-family |", "|---|---|---|---|"]
    for f, G in fam_groups.items():
        def cell(o):
            if not o or not o["computable"]:
                return "not computable"
            return (f"C={o['C']}/D={o['D']}/T={o['T']} → {'HOLDS' if o['holds'] else 'FAILS'}")
        L.append(f"| {f} | {', '.join('`'+m+'`' for m in G['models'])} | {cell(G['P1'])} | "
                 f"{cell(G['P2'])} |")

    # power judgement
    L += ["", f"## Power judgement — `effective_n` / `family_confounded`{prov}", "",
          f"- `family_confounded` = **{family_confounded}**",
          f"- `effective_n` = **{effective_n}** (cohort n_present = {len(present)})"]
    if family_confounded:
        L += ["", "> **This is 2 effective points, not 7.** The overall ordering holds only because the "
                  "extreme-zero-point group separates from the rest; with that between-group contrast "
                  "removed, the predicted ordering does not survive inside the larger group. The result "
                  "restates *\"the GPT-2-family model with the extreme zero-point differs from the "
                  "others\"* — it is **not** 7 independent draws, and the mechanism is **NOT** "
                  "established by it, whatever ρ says."]
    elif within_holds:
        L += ["", f"The predicted ordering survives **within** the {largest} group "
                  f"(n={groups[largest]['n']}), so the signal is not purely a between-family contrast."]
    else:
        L += ["", "The within-group ordering is not computable at the present cohort, so no claim about "
                  "family confounding is made yet (see disposition)."]

    # bottom line
    L += ["", "## Bottom line", ""]
    if disposition == "INVALID":
        L += ["**INVALID** — a mandatory input is self-inconsistent; the assembled table describes two "
              "different measurements of the same quantity or a model whose FIXCAL no-op control failed. "
              "Debug the inputs, do not interpret."]
        if disagree:
            L.append("- duplicate zero-point disagreements: " + json.dumps(disagree))
        if ctl_fail:
            L.append(f"- FIXCAL controls failed for: {', '.join(ctl_fail)}")
    elif disposition == "PENDING":
        L += [f"**PENDING** — {len(present)}/{len(COHORT)} of the frozen cohort are complete; data "
              "collection is still running. Everything above is **PROVISIONAL** and may not be cited in "
              "a downstream proposal or the paper: no verdict on P1/P2/P3 is binding until all 7 models carry all "
              "three quantities (frozen rule 0). Re-run this analyzer unchanged when the remaining raws "
              "land — the rule is frozen and is not re-opened on seeing partial numbers.",
              "", "Still awaiting: " + "; ".join(f"`{m}` → {', '.join(v)}" for m, v in missing.items())]
    elif disposition == "MECHANISM-SUPPORTED":
        L += ["**MECHANISM-SUPPORTED** (frozen rule 6). P1 and P2 hold as orderings, P3 holds on the "
              "held-out gpt2-large arm, and the ordering survives **within** the largest group, so the "
              "result is not the family contrast restated.",
              "", "Filings and the paper **may** now assert the gauge-artifact reading — scoped as "
              "pre-registered: descriptive cross-model evidence at n=7, an ordering across 7 "
              "checkpoints, never a within-model causal claim and never a licence to predict a specific "
              "θ′ from a zero-point."]
    else:
        why = []
        if not P1["holds"]:
            why.append("P1 ordering does not hold")
        if not P2["holds"]:
            why.append("P2 ordering does not hold")
        if not P3["holds"]:
            why.append("P3 fails (gpt2-large does not pattern with gpt2)")
        if family_confounded:
            why.append("separation is family-only (`family_confounded=true`, effective_n=2)")
        elif not within_holds:
            why.append("the within-group ordering does not hold / is not computable")
        L += [f"**MECHANISM-NOT-ESTABLISHED** (frozen rule 7): {'; '.join(why) or 'a frozen gate failed'}.",
              "", "**Failure branch, in force verbatim:** the mechanism **stays open**. Filings and the "
              "paper **keep the measured maps** (FIXCAL's calibration table, MATCHED's matched-suppression "
              "head-to-head, the a1_zeropoint run's zero-point table) and **drop the gauge-artifact explanation "
              "entirely** — removed, not softened, not hedged, not relegated to a footnote. No re-cut, no "
              "re-gate, no alternative predictor substituted after seeing this table; any such reading is "
              "post-hoc and non-binding and needs a fresh pre-registration on checkpoints outside this "
              "cohort."]

    L += ["", "---", "", "## Frozen decision rule (verbatim, `PREREG.md` §5)", "",
          summ["frozen_rule_verbatim"], ""]

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "REPORT.md"), "w").write("\n".join(L) + "\n")
    json.dump(jsonable(summ), open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1,
              default=str, allow_nan=False)
    print(f"ZPREACH: disposition={disposition} · present={len(present)}/{len(COHORT)} · "
          f"P1={'HOLDS' if P1['holds'] else 'no'} P2={'HOLDS' if P2['holds'] else 'no'} "
          f"P3={'HOLDS' if P3['holds'] else 'no'} · within={'HOLDS' if within_holds else 'no'} · "
          f"family_confounded={family_confounded} effective_n={effective_n} -> "
          f"{args.out_dir}/REPORT.md, summary.json")
    if missing:
        print("  missing: " + "; ".join(f"{m} -> {', '.join(v)}" for m, v in missing.items()))


if __name__ == "__main__":
    main()
