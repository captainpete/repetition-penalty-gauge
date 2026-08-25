#!/usr/bin/env python3
"""FIXCAL analyzer — applies the FROZEN reporting rule in PREREG.md §5 to the per-model raws.

Emits the migration map theta_raw -> theta'_fix at matched suppression (per model, per metric, with
first-class UNREACHABLE), the merged cross-model table, the deployed-band call-out, the in-loop
confidence + closed-form theta'_required statistics, the metric-consistency contrast (E1) and the
amendment-A prediction check (E2). Gates on CONTRASTS only -- no absolute-magnitude verdict anywhere.

  python code/analyze_fixcal.py \
      --raws 'results/fix_calibration/raw_*.json' --out-dir results/fix_calibration
"""
import os, re, json, glob, math, argparse

METRICS = ("rep_rate", "longest_run", "distinct2")
SIGNS = {"rep_rate": -1.0, "longest_run": -1.0, "distinct2": +1.0}   # suppression = SIGN * metric
PRIMARY = "rep_rate"
# deployed band (FIXCAL): 1.025 is not on the grid -> 1.02 stands in for ExLlama
DEPLOYED = [(1.02, "ExLlama ~1.025 (1.02 = nearest grid point)"),
            (1.05, ""),
            (1.10, "Ollama / GPT4All default"),
            (1.15, "")]


def frozen_rule():
    """The frozen reporting rule, read VERBATIM from PREREG.md §5 (no hand-synced copy to drift)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prereg", "FIXCAL_PREREG.md")
    try:
        txt = open(p).read()
        s = txt.index("\n", txt.index("## 5. FROZEN REPORTING RULE")) + 1
        return txt[s:txt.index("\n## ", s)].strip()
    except Exception as e:                                     # pragma: no cover
        return f"(could not read PREREG.md §5 next to the analyzer: {e})"


def size_hint(name):
    n = name.split("/")[-1].lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", n) or re.search(r"(\d+(?:\.\d+)?)b", n)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)m", n)
    if m:
        return float(m.group(1)) / 1000.0
    return 0.124 if n.startswith("gpt2") else 1.0


def f(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def theta_cell(row):
    """One migration-table cell: matched θ′ or the UNREACHABLE verdict with what it achieved."""
    if row is None:
        return "—"
    if row["verdict"] == "UNREACHABLE":
        return f"**UNREACHABLE** (θ′max: {row['achieved']:.4f} vs {row['target']:.4f})"
    if row["verdict"] == "TRIVIAL":
        return "TRIVIAL (θ′=1.0" + ("" if row.get("raw_suppresses") else "; raw did not suppress") + ")"
    star = "" if row.get("converged", True) else "~"
    return f"{star}{row['theta_fix']:.3f}"


def load(paths):
    models, seen = [], {}
    for p in paths:
        d = json.load(open(p))
        d["_path"] = p
        short = d["model"].split("/")[-1]
        seen[short] = seen.get(short, 0) + 1
        if seen[short] > 1:                    # two raws for the same model: keep both, disambiguate
            short = f"{short}#{seen[short]}"
        d["_short"] = short
        d["_bis"] = {m: {r["theta_raw"]: r for r in d["bisection"][m]} for m in METRICS}
        models.append(d)
    models.sort(key=lambda d: (size_hint(d["model"]), d["_short"]))
    return models


def metric_consistency(d):
    """E1 as a spread CONTRAST (frozen rule 4), never a cutoff."""
    anchors = d["anchors"]
    per_anchor = {}
    agree = True
    for a in anchors:
        vs = [d["_bis"][m].get(a, {}).get("verdict") for m in METRICS]
        same = len(set(vs)) == 1
        agree = agree and same
        ths = [d["_bis"][m].get(a, {}).get("theta_fix") for m in METRICS]
        ratio = None
        if all(t is not None and t > 0 for t in ths) and set(vs) == {"MATCHED"}:
            ratio = max(ths) / min(ths)
        per_anchor[a] = {"verdicts": vs, "verdicts_agree": same, "theta_fix": ths,
                         "across_metric_ratio": ratio}
    prim = [d["_bis"][PRIMARY][a]["theta_fix"] for a in anchors
            if d["_bis"][PRIMARY].get(a, {}).get("verdict") == "MATCHED"
            and d["_bis"][PRIMARY][a]["theta_fix"]]
    across_anchor = (max(prim) / min(prim)) if len(prim) >= 2 else None
    ratios = [v["across_metric_ratio"] for v in per_anchor.values()
              if v["across_metric_ratio"] is not None]
    worst = max(ratios) if ratios else None
    if worst is None or across_anchor is None:
        indep = None                    # not computable (needs MATCHED anchors under all 3 + >=2 primary)
    else:
        indep = bool(agree and worst <= across_anchor)
    return {"per_anchor": per_anchor, "verdicts_agree_all_anchors": bool(agree),
            "max_across_metric_ratio": worst, "across_anchor_ratio_primary": across_anchor,
            "metric_independent": indep}


def theta_req_population(d, field="theta_required"):
    """Sorted finite theta'_required values + the count of infinite ones, from the raw loop tokens."""
    fin = sorted(r[field] for r in d["loop_tokens"] if not r[f"{field}_inf"])
    return fin, sum(1 for r in d["loop_tokens"] if r[f"{field}_inf"])


def quantile(fin, n_inf, q):
    """Quantile over finite-then-infinite population; None means the quantile is infinite."""
    n = len(fin) + n_inf
    if n == 0:
        return None
    i = min(n - 1, max(0, int(math.ceil(q * n)) - 1))
    return fin[i] if i < len(fin) else None


def required_theta_for_anchor(loop_frac, target, fin, n_inf):
    """CLOSED-FORM 'you would need θ′ ≈ X' for one rep_rate anchor (protocol refinement refinement).

    To bring the loop rate from `loop_frac` (the unpenalized rep_rate) down to `target`, a fraction
    q = 1 − target/loop_frac of the loop tokens must flip. Flipping the cheapest first, that needs
    θ′ = quantile_q(θ′_required). STATIC: it reads the UNPENALIZED trajectory and therefore ignores the
    cascade (one flip can derail the rest of the loop), so it is an UPPER bound on what the empirical
    bisection needs. Diagnostic — reported, never a gate."""
    if not loop_frac or loop_frac <= 0:
        return {"q": None, "theta": None, "infinite": None}
    q = 1.0 - target / loop_frac
    if q <= 0:
        return {"q": q, "theta": 1.0, "infinite": False}     # already at/below target unpenalized
    t = quantile(fin, n_inf, min(q, 1.0))
    return {"q": q, "theta": t, "infinite": t is None}


def monotonicity(curve, metric):
    """DESCRIPTIVE only (never a gate, PREREG §5): does suppression rise with theta along a curve?"""
    ks = sorted(curve, key=float)
    vals = [SIGNS[metric] * curve[k][metric] for k in ks]
    viol = [(float(ks[i + 1]), vals[i] - vals[i + 1])
            for i in range(len(vals) - 1) if vals[i + 1] < vals[i] - 1e-12]
    return {"n_points": len(ks), "n_violations": len(viol),
            "max_violation": max((v for _, v in viol), default=0.0),
            "at_theta": [t for t, _ in viol]}


def deployed_rows(d):
    out = []
    for a, note in DEPLOYED:
        if a in d["_bis"][PRIMARY]:
            out.append((a, note, {m: d["_bis"][m][a] for m in METRICS}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raws", required=True,
                    help="glob (quoted) and/or comma-separated raw_<model>.json paths")
    ap.add_argument("--out-dir", default="results/fix_calibration")
    args = ap.parse_args()

    paths = []
    for pat in args.raws.split(","):
        pat = pat.strip()
        hits = sorted(glob.glob(pat))
        paths += hits if hits else ([pat] if os.path.exists(pat) else [])
    if not paths:
        raise SystemExit(f"no raw files matched: {args.raws}")
    models = load(paths)

    # ---------- validity (frozen rule 1)
    bad = [d["_short"] for d in models
           if not (d["controls"]["noop_raw1_eq_fix1"] and d["controls"]["instrumented_eq_raw1"])]
    invalid = len(bad) > 0

    # ---------- per-model analysis
    summ = {"n_models": len(models), "models": {}, "invalid": invalid, "invalid_models": bad}
    for d in models:
        cons = metric_consistency(d)
        dep = deployed_rows(d)
        dep_unreach = [a for a, _, rows in dep if rows[PRIMARY]["verdict"] == "UNREACHABLE"]
        dep_matched = [a for a, _, rows in dep if rows[PRIMARY]["verdict"] == "MATCHED"]
        ls = d["loop_stats"]
        med = ls["theta_required"]["median"]
        over_ceiling = (med is None) or (med > d["theta_max"])   # DIAGNOSTIC, not a gate
        fin, ninf = theta_req_population(d, "theta_required")
        finu, ninfu = theta_req_population(d, "theta_required_unseen")
        d["_req"] = {a: {"spec": required_theta_for_anchor(ls["loop_frac"],
                                                           d["_bis"][PRIMARY][a]["target"], fin, ninf),
                         "unseen": required_theta_for_anchor(ls["loop_frac"],
                                                             d["_bis"][PRIMARY][a]["target"],
                                                             finu, ninfu)}
                     for a in d["anchors"]}
        summ["models"][d["_short"]] = {
            "model": d["model"], "revision": d.get("revision"), "path": d["_path"],
            "device": d["device"], "dtype": d["dtype"], "max_new": d["max_new"],
            "n_prompts": d["n_prompts"], "theta_max": d["theta_max"],
            "n_fix_evals": d["n_fix_evals"], "elapsed_s": d.get("elapsed_s"),
            "controls": {k: v for k, v in d["controls"].items()},
            "raw_grid": {k: {m: v[m] for m in METRICS} for k, v in d["raw_grid"].items()},
            "fix_cache": {k: {m: v[m] for m in METRICS} for k, v in d["fix_cache"].items()},
            "migration": {m: [{kk: vv for kk, vv in r.items() if kk != "per_prompt"}
                              for r in d["bisection"][m]] for m in METRICS},
            "metric_consistency": cons,
            "monotonicity": {"raw": {m: monotonicity(d["raw_grid"], m) for m in METRICS},
                             "fix": {m: monotonicity(d["fix_cache"], m) for m in METRICS}},
            "deployed_unreachable_primary": dep_unreach,
            "deployed_matched_primary": dep_matched,
            "loop_stats": ls,
            "required_theta_per_anchor": d["_req"],
            "amendA": {"in_loop_p_top_mean": ls["p_top_loop"]["mean"],
                       "in_loop_p_top_median": ls["p_top_loop"]["median"],
                       "theta_required_median": med,
                       "theta_required_median_is_infinite": ls["theta_required"]["median_is_infinite"],
                       "theta_required_p90": ls["theta_required"]["p90"],
                       "theta_required_unseen_median": ls["theta_required_unseen"]["median"],
                       "frac_gt_theta_max": ls["theta_required"]["frac_gt_theta_max"],
                       "median_over_ceiling_diagnostic": bool(over_ceiling),
                       "observed_deployed_unreachable": bool(dep_unreach)},
        }

    # ---------- global branch (frozen rules 5/6)
    any_dep_unreach = any(summ["models"][k]["deployed_unreachable_primary"] for k in summ["models"])
    branch = "(ii) UNREACHABLE at a deployed anchor" if any_dep_unreach else "(i) reachable"
    summ["branch"] = branch
    summ["downstream_proposals"] = "PAUSED" if any_dep_unreach else "PROCEED (MATCHED-experiment pairs)"

    # ---------- amendment A (E2) across models — a CROSS-MODEL CONTRAST, never an absolute threshold
    keys = list(summ["models"])
    un = [k for k in keys if summ["models"][k]["amendA"]["observed_deployed_unreachable"]]
    ok = [k for k in keys if not summ["models"][k]["amendA"]["observed_deployed_unreachable"]]

    def ptop(k):                                   # None (no loop tokens at all) sorts to the bottom
        v = summ["models"][k]["amendA"]["in_loop_p_top_mean"]
        return -math.inf if v is None else v

    def tmed(k):                                   # infinite median sorts to the top
        v = summ["models"][k]["amendA"]["theta_required_median"]
        return math.inf if v is None else v

    computable = bool(un) and bool(ok)
    sep_ptop = (min(ptop(k) for k in un) > max(ptop(k) for k in ok)) if computable else None
    sep_theta = (min(tmed(k) for k in un) > max(tmed(k) for k in ok)) if computable else None
    g_un = [ptop(k) for k in un]
    g_ok = [ptop(k) for k in ok]
    summ["amendA_summary"] = {
        "n_models": len(models),
        "unreachable_models": un, "matched_models": ok,
        "computable": computable,
        "separates_on_in_loop_p_top": sep_ptop,
        "separates_on_median_theta_required": sep_theta,
        "holds": bool(sep_ptop and sep_theta) if computable else None,
        "mean_in_loop_p_top_UNREACHABLE_group": (sum(g_un) / len(g_un)) if g_un else None,
        "mean_in_loop_p_top_MATCHED_group": (sum(g_ok) / len(g_ok)) if g_ok else None,
        "group_contrast": ((sum(g_un) / len(g_un)) - (sum(g_ok) / len(g_ok))) if computable else None,
        "note": None if computable else
        "every model fell on the same side of the deployed-anchor verdict -> the cross-model contrast "
        "is not computable; only the per-model diagnostics below are reported"}
    summ["metric_consistency_all_models"] = {
        k: summ["models"][k]["metric_consistency"]["metric_independent"] for k in summ["models"]}

    # ---------------------------------------------------------------- REPORT
    L = ["# FIXCAL — migration map θ_raw → θ′_fix at matched repetition suppression", "",
         f"Models: {len(models)} · "
         + ", ".join(f"`{d['_short']}` ({d['dtype']}, max_new={d['max_new']}, "
                     f"n_prompts={d['n_prompts']}, θ′max={d['theta_max']:g})" for d in models), "",
         f"**Validity: {'INVALID — ' + ', '.join(bad) if invalid else 'controls PASS'}**", "",
         f"**Branch: {branch}** → downstream proposals: **{summ['downstream_proposals']}**", "",
         "## Frozen reporting rule (PREREG.md §5, verbatim)", "", frozen_rule(), ""]

    L += ["## 0. Controls (PROTOCOL §3/§5)", "",
          "| model | no-op raw θ=1 ≡ fix θ′=1 | instrumented θ=1 ≡ raw θ=1 | rep_rate @θ=1 | "
          "fix sweeps |", "|---|:--:|:--:|--:|--:|"]
    for d in models:
        c = d["controls"]
        L.append(f"| {d['_short']} | {'PASS' if c['noop_raw1_eq_fix1'] else '**FAIL**'} | "
                 f"{'PASS' if c['instrumented_eq_raw1'] else '**FAIL**'} | "
                 f"{c['raw_theta1']['rep_rate']:.4f} | {d['n_fix_evals']} |")
    if invalid:
        L += ["", "A control FAILED: the harness is wrong, not the claim. Do not interpret the map."]

    # ---- per model
    for d in models:
        sm = summ["models"][d["_short"]]
        L += ["", f"## {d['_short']}", "",
              "### Raw dense grid (all three suppression metrics)", "",
              "| θ_raw | rep_rate | longest_run | distinct2 |", "|--:|--:|--:|--:|"]
        for k in sorted(d["raw_grid"], key=float):
            r = d["raw_grid"][k]
            tag = " (baseline)" if float(k) == 1.0 else ""
            L.append(f"| {float(k):g}{tag} | {r['rep_rate']:.4f} | {r['longest_run']:.3f} | "
                     f"{r['distinct2']:.4f} |")

        L += ["", "### Fix curve — every evaluated θ′ (bisection cache)", "",
              "| θ′ | rep_rate | longest_run | distinct2 |", "|--:|--:|--:|--:|"]
        for k in sorted(d["fix_cache"], key=float):
            r = d["fix_cache"][k]
            L.append(f"| {float(k):.4f} | {r['rep_rate']:.4f} | {r['longest_run']:.3f} | "
                     f"{r['distinct2']:.4f} |")

        L += ["", "### Migration map θ_raw → θ′_fix (matched suppression, per matching metric)", "",
              "`~` = iteration budget exhausted before the tolerance; θ′ is then the smallest evaluated "
              "θ′ reaching at least the anchor's suppression (bracket in summary.json).", "",
              "| θ_raw | match on rep_rate (primary) | match on longest_run | match on distinct2 | "
              "raw suppresses? | verdicts agree? |", "|--:|---|---|---|:--:|:--:|"]
        for a in d["anchors"]:
            rows = {m: d["_bis"][m].get(a) for m in METRICS}
            pa = sm["metric_consistency"]["per_anchor"][a]
            L.append(f"| {a:g} | {theta_cell(rows['rep_rate'])} | {theta_cell(rows['longest_run'])} | "
                     f"{theta_cell(rows['distinct2'])} | "
                     f"{'yes' if rows[PRIMARY]['raw_suppresses'] else 'NO'} | "
                     f"{'yes' if pa['verdicts_agree'] else 'NO'} |")

        dep = deployed_rows(d)
        if dep:
            L += ["", "### Deployed band (called out)", "",
                  "| θ_raw | engine default | rep_rate target | verdict (primary) | θ′_fix | "
                  "θ′/θ_raw | closed-form θ′ needed (static) |", "|--:|---|--:|---|--:|--:|--:|"]
            for a, note, rows in dep:
                r = rows[PRIMARY]
                ratio = f"{r['theta_fix'] / a:.2f}×" if r["theta_fix"] else "—"
                rq = d["_req"][a]["spec"]
                need = "∞" if rq["infinite"] else (f(rq["theta"], 1) if rq["theta"] else "n/a")
                L.append(f"| {a:g} | {note or '—'} | {r['target']:.4f} | "
                         f"{'**UNREACHABLE**' if r['verdict'] == 'UNREACHABLE' else r['verdict']} | "
                         f"{theta_cell(r)} | {ratio} | {need} |")
            L += ["", "*closed-form θ′ needed* = quantile_q(θ′_required) with q = 1 − target/loop_frac "
                      "(flip the cheapest loop tokens first). STATIC — read off the unpenalized "
                      "trajectory, so it ignores the cascade (one flip can derail the rest of a loop) "
                      "and is an upper bound on what the empirical bisection needs. Diagnostic, not a "
                      "gate; the empirical column is the result."]

        ls = d["loop_stats"]
        tr, tu = ls["theta_required"], ls["theta_required_unseen"]
        L += ["", "### Loop confidence + closed-form required θ′ "
                  "(raw θ=1.0 run; θ′_required = ln p_runner / ln p_top)", "",
              f"- loop tokens: {ls['n_loop_tokens']}/{ls['n_gen_tokens']} "
              f"(loop_frac {f(ls['loop_frac'],4)}; strict, prompt tokens excluded: "
              f"{ls['n_loop_tokens_strict']}, {f(ls['loop_frac_strict'],4)})",
              f"- **in-loop top-1 probability** (amendment A): mean {f(ls['p_top_loop']['mean'],4)}, "
              f"median {f(ls['p_top_loop']['median'],4)}, p10 {f(ls['p_top_loop']['p10'],4)}, "
              f"p90 {f(ls['p_top_loop']['p90'],4)}  ·  all-token mean "
              f"{f(ls['p_top_all_mean'],4)}",
              f"- **θ′_required** (spec form, best non-argmax competitor — a LOWER bound): "
              f"median {'∞' if tr['median'] is None else f(tr['median'],2)}, "
              f"p10 {'∞' if tr['p10'] is None else f(tr['p10'],2)}, "
              f"p90 {'∞' if tr['p90'] is None else f(tr['p90'],2)}; "
              f"frac ∞ {f(tr['frac_infinite'],4)}; frac > θ′max {f(tr['frac_gt_theta_max'],4)}",
              f"- **θ′_required (exact, best UNSEEN competitor)**: "
              f"median {'∞' if tu['median'] is None else f(tu['median'],2)}, "
              f"p90 {'∞' if tu['p90'] is None else f(tu['p90'],2)}; "
              f"frac > θ′max {f(tu['frac_gt_theta_max'],4)}",
              f"- guards: p_top ≥ 1−1e-6 on {ls['n_ptop_saturated']} loop tokens; "
              f"p_runner ≤ 1e-12 on {ls['n_prunner_underflow']} (both → θ′_required = ∞)"]

        mono = sm["monotonicity"]
        L += ["", "### Monotonicity of the two curves (DESCRIPTIVE — never a gate)", "",
              "| curve | metric | points | violations (suppression falls as θ rises) | max drop | "
              "at θ |", "|---|---|--:|--:|--:|---|"]
        for arm in ("raw", "fix"):
            for m in METRICS:
                mo = mono[arm][m]
                L.append(f"| {arm} | {m} | {mo['n_points']} | {mo['n_violations']} | "
                         f"{mo['max_violation']:.4f} | "
                         f"{', '.join(f'{t:g}' for t in mo['at_theta']) or '—'} |")
        nonmono = [f"{arm}/{m}" for arm in ("raw", "fix") for m in METRICS
                   if mono[arm][m]["n_violations"]]
        if nonmono:
            L.append("")
            L.append(f"Non-monotone: {', '.join(nonmono)}. The bisection assumes a monotone "
                     "metric-vs-θ′ curve; where the fix curve is non-monotone the bracket is still "
                     "valid at its endpoints (the UNREACHABLE test is an actual evaluation at θ′max, "
                     "not an extrapolation) but the interior θ′ is one match, not necessarily the "
                     "unique one. Reported, not gated (FIXCAL: a model-chaotic map is itself a "
                     "finding that weakens the migration story).")

        cons = sm["metric_consistency"]
        L += ["", "### Metric-consistency (E1, spread contrast)",
              f"- verdict classes agree at every anchor: "
              f"**{'yes' if cons['verdicts_agree_all_anchors'] else 'NO'}**",
              f"- max across-metric θ′ spread (max/min at a commonly-MATCHED anchor): "
              f"{f(cons['max_across_metric_ratio'],3)}",
              f"- across-anchor θ′ spread under the primary metric: "
              f"{f(cons['across_anchor_ratio_primary'],3)}",
              f"- **map is metric-{'independent' if cons['metric_independent'] else 'dependent'}**"
              if cons["metric_independent"] is not None else
              "- **metric-independence not computable** (no anchor MATCHED under all three metrics, "
              "and/or <2 MATCHED anchors under the primary metric) — the verdict-class agreement above "
              "carries E1 on its own"]

    # ---- merged cross-model table
    all_anchors = sorted({a for d in models for a in d["anchors"]})
    L += ["", "## Merged cross-model migration table (primary metric = rep_rate)", "",
          "| θ_raw | " + " | ".join(d["_short"] for d in models) + " |",
          "|--:|" + "---|" * len(models)]
    for a in all_anchors:
        L.append(f"| {a:g} | " + " | ".join(theta_cell(d["_bis"][PRIMARY].get(a)) for d in models)
                 + " |")
    L += ["", "### Deployed band, merged (rep_rate)", "",
          "| θ_raw | engine | " + " | ".join(d["_short"] for d in models) + " |",
          "|--:|---|" + "---|" * len(models)]
    for a, note in DEPLOYED:
        if any(a in d["_bis"][PRIMARY] for d in models):
            L.append(f"| {a:g} | {note or '—'} | "
                     + " | ".join(theta_cell(d["_bis"][PRIMARY].get(a)) for d in models) + " |")

    # ---- loop confidence merged + amendment A
    L += ["", "## Loop confidence / required θ′, merged (amendment A / E2)", "",
          "| model | in-loop p_top mean | in-loop p_top median | median θ′_required | "
          "p90 θ′_required | frac θ′_req > θ′max | deployed anchor UNREACHABLE? |",
          "|---|--:|--:|--:|--:|--:|:--:|"]
    for d in models:
        A = summ["models"][d["_short"]]["amendA"]
        L.append(f"| {d['_short']} | {f(A['in_loop_p_top_mean'],4)} | "
                 f"{f(A['in_loop_p_top_median'],4)} | "
                 f"{'∞' if A['theta_required_median'] is None else f(A['theta_required_median'],2)} | "
                 f"{'∞' if A['theta_required_p90'] is None else f(A['theta_required_p90'],2)} | "
                 f"{f(A['frac_gt_theta_max'],4)} | "
                 f"{'YES' if A['observed_deployed_unreachable'] else 'no'} |")
    aa = summ["amendA_summary"]
    L += ["", "**Amendment-A prediction (E2), adjudicated as a CROSS-MODEL CONTRAST** (never an "
              "absolute θ′ threshold — PROTOCOL §4): UNREACHABLE models should rank above MATCHED "
              "models on both in-loop p_top and median θ′_required."]
    if aa["computable"]:
        L += [f"- UNREACHABLE at the deployed anchor: {', '.join(aa['unreachable_models'])}  ·  "
              f"MATCHED: {', '.join(aa['matched_models'])}",
              f"- separates on in-loop p_top: **{'yes' if aa['separates_on_in_loop_p_top'] else 'no'}** "
              f"(group means {f(aa['mean_in_loop_p_top_UNREACHABLE_group'],4)} vs "
              f"{f(aa['mean_in_loop_p_top_MATCHED_group'],4)}; contrast "
              f"{f(aa['group_contrast'],4)})",
              f"- separates on median θ′_required: "
              f"**{'yes' if aa['separates_on_median_theta_required'] else 'no'}**",
              f"- **E2 {'HOLDS' if aa['holds'] else 'FAILS'}** (both required). A failure does not "
              f"change the map; it means the mechanism is unexplained."]
    else:
        L += [f"- **Not computable:** {aa['note']}. "
              f"(All models: {', '.join(aa['unreachable_models'] or aa['matched_models'])}.) "
              f"The per-model statistics above are reported as diagnostics; with every model on one "
              f"side there is no contrast to adjudicate."]

    # ---- bottom line
    L += ["", "## Bottom line (frozen rule 5/6)", ""]
    if invalid:
        L.append("**INVALID** — a mandatory control failed; the harness is wrong. Do not interpret.")
    elif any_dep_unreach:
        for d in models:
            A = summ["models"][d["_short"]]["amendA"]
            if not A["observed_deployed_unreachable"]:
                continue
            anchors_s = summ["models"][d["_short"]]["deployed_unreachable_primary"]
            need = []
            for a in anchors_s:
                rq = d["_req"][a]["spec"]
                need.append(f"θ_raw={a:g} → θ′ ≈ "
                            + ("∞" if rq["infinite"] else
                               (f"{rq['theta']:.0f}" if rq["theta"] else "n/a")))
            med = A["theta_required_median"]
            med_s = "∞" if med is None else f"{med:.1f}"
            L.append(f"- `{d['_short']}`: deployed anchors {anchors_s} are **UNREACHABLE** at "
                     f"θ′ ≤ {d['theta_max']:g}. Closed form (static, upper bound) says you would need "
                     f"{'; '.join(need)}. In-loop top-1 p = {f(A['in_loop_p_top_mean'],3)} (mean), "
                     f"median per-token θ′_required = {med_s}.")
        L += ["",
              "Per amendment B this is a **structural impossibility**, not a dial that was too small: "
              "the raw operator's suppression at the deployed θ is substantially a **gauge artifact** "
              "(it is strong *because* it reads the arbitrary zero-point), and the normalized form is "
              "**not a drop-in with a rescaled dial**.",
              "",
              "**Downstream engine downstream_proposals (SGLang, mistral.rs, llama.cpp, transformers PR) are "
              "PAUSED.** The recommendation narrows to: *normalized-multiplicative for "
              "gauge-correctness + subtractive presence/frequency penalties for loop-breaking* "
              "(subtractive is shift-invariant AND effective on confident tokens, since z − α hits all "
              "logits equally). MATCHED runs only on anchors that are MATCHED — no nearby pair is "
              "substituted."]
        reach = {d["_short"]: summ["models"][d["_short"]]["deployed_matched_primary"] for d in models}
        L.append("")
        L.append("Reachable deployed anchors per model (the only pairs MATCHED may use): "
                 + json.dumps(reach))
    else:
        L += ["- Every deployed anchor is MATCHED: the migration note is *\"your θ_raw ≈ our θ′\"*. "
              "θ′/θ_raw ratios are in the deployed-band tables above (expect θ′ ≫ θ_raw).",
              "- MATCHED proceeds on these matched pairs."]

    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, "REPORT.md"), "w").write("\n".join(L) + "\n")
    json.dump(summ, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    print(f"branch={branch} · downstream_proposals={summ['downstream_proposals']} · invalid={invalid} · "
          f"models={len(models)} -> {args.out_dir}/REPORT.md, summary.json")


if __name__ == "__main__":
    main()
