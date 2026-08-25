#!/usr/bin/env python3
"""MATCHED (MATCHED) analyzer — applies the FROZEN decision rule in PREREG.md §6/§6a verbatim.

Reads any subset of runs/MATCHED/raw_<model>_<stage>.json (partial stages are handled and reported),
forms PAIRED bootstrap CIs on the contrast (fix - raw), oriented so positive = the fix is better, over
the unit of replication (prompt / json schema instance / HumanEval problem), and writes REPORT.md +
summary.json.

  python analyze_matched.py --raws 'results/matched_strength/raw_*.json' --out-dir results/matched_strength
"""
import os, json, glob, random, argparse
from collections import defaultdict

BOOT = 10000
SEED = 0
REP_RATE_TOL = 0.01
DEPLOYED_ANCHOR = 1.1
SAT_EPS = 0.02                       # a metric mean within this of 0 or 1 is flagged saturated

STRUCTURED = ["json_valid", "humaneval_pass1"]
OPEN_ENDED = ["quality", "distinct1", "distinct2"]
REPORTED_ONLY = ["rep_rate", "longest_run"]          # rep_rate is the MATCHING CHECK, never quality
ALL_STAGES = ["pairs", "gauge", "open", "json", "humaneval", "humaneval_scored"]

FROZEN_RULE = """\
> **Premise/validity (per pair):** the pair must actually be matched — |rep_rate(raw) − rep_rate(fix)| within
> the FIXCAL tolerance (0.01). A pair failing this is reported `MISMATCHED` and EXCLUDED from the quality
> verdict (it is a calibration failure, never evidence about the fix). If no pair survives on a model →
> INVALID for that model.
> **Gauge gate (c):** flip-rate under fix must be 0 (exact) at every surviving pair, and flip-rate under raw
> > 0. Failure here is a **hard REFUTED** regardless of quality metrics — gauge-invariance is the fix's
> entire reason to exist.
> **CORE (quality at equal suppression), per model, gated on CONTRASTS with bootstrap CIs (unit =
> prompt/problem/schema):**
> **DOMINANT** iff at every surviving pair the fix is ≥ raw on the structured metrics (JSON validity, pass@1)
> with the paired CI excluding 0 on at least one, AND not significantly worse on any open-ended metric.
> **TIES** iff all paired CIs include 0 (no reliable difference either way) — the honest "gauge invariance at
> no measurable cost" outcome, which is still a viable recommendation.
> **LOSES** iff the fix is significantly worse (paired CI excluding 0 in raw's favour) on any metric at any
> surviving pair — downstream_proposals pause on that basis, and the losing metric/pair is named.
> Report per pair and per metric; never conjoin metrics into one number. The deployed anchor (θ_raw=1.10) is
> reported separately and called out, since it carries the downstream proposal decision."""


# ---------------------------------------------------------------------------- stats
def paired_boot(fix_v, raw_v, boot=None, seed=None):
    """Paired bootstrap CI of mean(fix - raw) over the unit of replication."""
    boot = BOOT if boot is None else boot            # resolved at CALL time (--boot/--seed override)
    seed = SEED if seed is None else seed
    d = [a - b for a, b in zip(fix_v, raw_v)]
    n = len(d)
    if n == 0:
        return {"n": 0, "diff": None, "ci": [None, None], "excludes0": None, "sign": None}
    r = random.Random(seed)
    means = sorted(sum(d[r.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    lo, hi = means[int(0.025 * boot)], means[int(0.975 * boot)]
    return {"n": n, "diff": sum(d) / n, "ci": [lo, hi],
            "excludes0": (lo > 0 or hi < 0), "sign": (1 if lo > 0 else (-1 if hi < 0 else 0))}


def boot_from_arrays(fix_boot, raw_boot):
    """MAUVE case: set-level bootstrap replicates already share prompt resamples across conditions."""
    d = sorted(a - b for a, b in zip(fix_boot, raw_boot))
    n = len(d)
    lo, hi = d[int(0.025 * n)], d[int(0.975 * n)]
    return {"n": n, "diff": sum(d) / n, "ci": [lo, hi],
            "excludes0": (lo > 0 or hi < 0), "sign": (1 if lo > 0 else (-1 if hi < 0 else 0)),
            "unit": "set_bootstrap"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def saturated(m_raw, m_fix):
    for m in (m_raw, m_fix):
        if m is None:
            return False
    return all(min(abs(m), abs(1 - m)) <= SAT_EPS for m in (m_raw, m_fix))


# ---------------------------------------------------------------------------- loading
def load_raws(pattern):
    by_model = defaultdict(dict)
    files = sorted(glob.glob(pattern))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  skip unreadable {f}: {e}")
            continue
        if d.get("experiment") != "MATCHED" or d.get("stage") not in ALL_STAGES \
                or not d.get("model_slug"):
            print(f"  skip (not a MATCHED stage raw): {f}")
            continue
        slug, stage = d["model_slug"], d["stage"]
        d["_path"] = f
        by_model[slug][stage] = d
    return by_model, files


def cond_index(d):
    """label -> condition dict, plus (pair_idx, op) -> condition."""
    by_label, by_pair = {}, {}
    for c in d.get("conditions", []):
        by_label[c["label"]] = c
        if c.get("pair_idx") is not None:
            by_pair[(c["pair_idx"], c["op"])] = c
    return by_label, by_pair


# ---------------------------------------------------------------------------- per-model analysis
def analyze_model(slug, stages):
    out = {"model": slug, "stages_present": sorted(stages), "stages_missing":
           [s for s in ("pairs", "gauge", "open", "json", "humaneval_scored") if s not in stages]}
    ref = stages.get("pairs") or stages.get("open") or stages.get("gauge") or \
        stages.get("json") or stages.get("humaneval_scored") or stages.get("humaneval")
    out["pairs"] = ref.get("pairs", []) if ref else []
    out["pairs_meta"] = ref.get("pairs_meta", {}) if ref else {}
    out["model_id"] = ref.get("model") if ref else None
    out["revision"] = ref.get("revision") if ref else None
    out["dtype"] = ref.get("dtype") if ref else None

    # ---- controls -------------------------------------------------------
    ctrl = {}
    for st in ("pairs", "open", "json"):
        if st in stages:
            ctrl[f"{st}.noop_raw1_eq_fix1"] = stages[st].get("controls", {}).get("noop_raw1_eq_fix1")
    if "gauge" in stages:
        g = stages["gauge"].get("controls", {})
        ctrl["gauge.noop_flip_raw"] = g.get("noop_flip_raw")
        ctrl["gauge.noop_flip_fix"] = g.get("noop_flip_fix")
        ctrl["gauge.noop_gate_exact_zero"] = g.get("noop_gate_exact_zero")
        ctrl["gauge.instrumented_eq_loopcheck"] = g.get("instrumented_eq_loopcheck")
    out["controls"] = ctrl
    bad_ctrl = [k for k, v in ctrl.items() if isinstance(v, bool) and v is False]
    out["controls_pass"] = not bad_ctrl
    out["controls_failed"] = bad_ctrl

    # ---- Stage 0a: premise (rep_rate matching) ---------------------------
    premise = []
    if "pairs" in stages:
        d = stages["pairs"]
        _, by_pair = cond_index(d)
        for i, p in enumerate(d.get("pairs", [])):
            cr, cf = by_pair.get((i, "raw")), by_pair.get((i, "fix"))
            if not cr or not cf:
                continue
            rr = mean([r["rep_rate"] for r in cr["per_prompt"]])
            rf = mean([r["rep_rate"] for r in cf["per_prompt"]])
            delta = abs(rr - rf)
            premise.append({
                "pair_idx": i, "theta_raw": p["theta_raw"], "theta_fix": p["theta_fix"],
                "is_deployed_anchor": p.get("is_deployed_anchor", False),
                "rep_rate_raw": rr, "rep_rate_fix": rf, "abs_delta": delta,
                "tolerance": REP_RATE_TOL, "status": ("MATCHED" if delta <= REP_RATE_TOL else "MISMATCHED"),
                "fixcal_target_rep_rate": p.get("fixcal_target_rep_rate"),
                "fixcal_raw_drift": (None if p.get("fixcal_target_rep_rate") is None
                                     else rr - p["fixcal_target_rep_rate"]),
            })
    out["premise"] = premise
    out["premise_available"] = "pairs" in stages
    surviving = [e["pair_idx"] for e in premise if e["status"] == "MATCHED"]
    out["surviving_pairs"] = surviving

    # ---- determinism cross-check (pairs vs open) -------------------------
    if "pairs" in stages and "open" in stages:
        a, b = cond_index(stages["pairs"])[0], cond_index(stages["open"])[0]
        diffs = []
        for lab in sorted(set(a) & set(b)):
            for x, y in zip(a[lab]["per_prompt"], b[lab]["per_prompt"]):
                if x.get("gen_sha") and y.get("gen_sha") and x["gen_sha"] != y["gen_sha"]:
                    diffs.append({"label": lab, "prompt_idx": x["prompt_idx"]})
        out["determinism_cross_check"] = {
            "n_mismatched_generations": len(diffs), "examples": diffs[:5],
            "pairs_max_new": stages["pairs"].get("max_new"), "open_max_new": stages["open"].get("max_new"),
            "note": ("greedy decoding is deterministic, so >0 mismatches means the two stages did not run "
                     "the same configuration (check max_new/limit/dtype) or a harness artifact")}

    # ---- Stage 0b: gauge gate -------------------------------------------
    gauge = []
    if "gauge" in stages:
        rows = stages["gauge"].get("gauge", [])
        byk = {}
        for r in rows:
            if r.get("kind") == "pair":
                byk[(round(float(r["theta_raw"]), 4), r["op"])] = r
        for i, p in enumerate(stages["gauge"].get("pairs", [])):
            k = round(float(p["theta_raw"]), 4)
            rr, rf = byk.get((k, "raw")), byk.get((k, "fix"))
            if not rr or not rf:
                continue
            fr, ff = rr["pooled_flip_rate"], rf["pooled_flip_rate"]
            # sub-diagnosis (reported, does NOT change the frozen verdict): the gate conflates the real
            # failure (the fix is not gauge-invariant) with a positive-control failure (raw did not move,
            # so the gauge test has no power at this anchor).  Name which one fired.
            mode = None
            if ff != 0.0 and fr > 0.0:
                mode = "FIX_NOT_INVARIANT (the real refutation: flip(fix) != 0)"
            elif ff != 0.0 and fr <= 0.0:
                mode = "FIX_NOT_INVARIANT + raw control flat (both broken -- suspect the harness)"
            elif ff == 0.0 and fr <= 0.0:
                mode = ("RAW_CONTROL_FLAT (flip(raw)=0: the c=+-5 test has no power at this anchor; "
                        "this is NOT evidence against the fix -- see A1b, flip(raw)@1.02 ~ 0.6 at "
                        "16 prompts x 200 tokens, so a 0 here means the run was too short/narrow)")
            gauge.append({"pair_idx": i, "theta_raw": k, "theta_fix": p["theta_fix"],
                          "is_deployed_anchor": p.get("is_deployed_anchor", False),
                          "flip_raw": fr, "flip_fix": ff,
                          "fix_exact_zero": (ff == 0.0), "raw_positive": (fr > 0.0),
                          "failure_mode": mode,
                          "status": ("PASS" if (ff == 0.0 and fr > 0.0) else "FAIL")})
    out["gauge"] = gauge
    out["gauge_available"] = "gauge" in stages
    gauge_checked = [g for g in gauge if (not surviving or g["pair_idx"] in surviving)]
    out["gauge_gate"] = ("MISSING" if not gauge else
                         ("PASS" if all(g["status"] == "PASS" for g in gauge_checked) else "FAIL"))
    out["gauge_gate_failures"] = [g for g in gauge_checked if g["status"] == "FAIL"]

    # ---- CORE contrasts --------------------------------------------------
    contrasts = []                                   # one row per (pair, metric)
    pair_meta = {i: p for i, p in enumerate(out["pairs"])}

    def add(pair_idx, metric, fixv, rawv, cls, extra=None):
        if not fixv or not rawv:
            return
        st = paired_boot(fixv, rawv)
        row = {"pair_idx": pair_idx, "theta_raw": pair_meta.get(pair_idx, {}).get("theta_raw"),
               "theta_fix": pair_meta.get(pair_idx, {}).get("theta_fix"),
               "is_deployed_anchor": pair_meta.get(pair_idx, {}).get("is_deployed_anchor", False),
               "metric": metric, "class": cls, "mean_raw": mean(rawv), "mean_fix": mean(fixv), **st}
        row["saturated"] = saturated(row["mean_raw"], row["mean_fix"])
        row["surviving"] = pair_idx in surviving if premise else None
        if extra:
            row.update(extra)
        contrasts.append(row)

    def paired_vectors(cr, cf, key, unit_key, item_key):
        """align raw/fix per-unit values, dropping units where either side is missing/None."""
        rm = {r[unit_key]: r.get(key) for r in cr[item_key]}
        fm = {r[unit_key]: r.get(key) for r in cf[item_key]}
        units = sorted(set(rm) & set(fm), key=lambda u: (str(type(u)), u))
        rv, fv, drop = [], [], 0
        for u in units:
            if rm[u] is None or fm[u] is None:
                drop += 1
                continue
            rv.append(float(rm[u]))
            fv.append(float(fm[u]))
        return fv, rv, drop

    # arm (b) open-ended
    if "open" in stages:
        d = stages["open"]
        _, by_pair = cond_index(d)
        qunit = d.get("quality_unit")
        for i in range(len(d.get("pairs", []))):
            cr, cf = by_pair.get((i, "raw")), by_pair.get((i, "fix"))
            if not cr or not cf:
                continue
            for m in ("distinct1", "distinct2"):
                fv, rv, drop = paired_vectors(cr, cf, m, "prompt_idx", "per_prompt")
                add(i, m, fv, rv, "open_ended", {"n_dropped": drop})
            for m in ("rep_rate", "longest_run"):
                fv, rv, drop = paired_vectors(cr, cf, m, "prompt_idx", "per_prompt")
                add(i, m, fv, rv, "reported_only", {"n_dropped": drop})
            if qunit == "prompt":
                fv, rv, drop = paired_vectors(cr, cf, "quality", "prompt_idx", "per_prompt")
                add(i, "quality", fv, rv, "open_ended",
                    {"n_dropped": drop, "measure": d.get("quality_measure")})
            elif qunit == "set_bootstrap" and cr.get("quality_boot") and cf.get("quality_boot"):
                st = boot_from_arrays(cf["quality_boot"], cr["quality_boot"])
                contrasts.append({"pair_idx": i, "theta_raw": pair_meta.get(i, {}).get("theta_raw"),
                                  "theta_fix": pair_meta.get(i, {}).get("theta_fix"),
                                  "is_deployed_anchor": pair_meta.get(i, {}).get("is_deployed_anchor", False),
                                  "metric": "quality", "class": "open_ended",
                                  "mean_raw": cr.get("quality_point"), "mean_fix": cf.get("quality_point"),
                                  "measure": d.get("quality_measure"), "saturated": False,
                                  "surviving": i in surviving if premise else None, **st})
        out["quality_measure"] = d.get("quality_measure")
        out["quality_unit"] = qunit
        out["quality_detail"] = {k: v for k, v in (d.get("quality_detail") or {}).items() if k != "rubric"}
        out["quality_fallback_reason"] = d.get("quality_fallback_reason")

    # arm (a1) json
    if "json" in stages:
        d = stages["json"]
        _, by_pair = cond_index(d)
        for i in range(len(d.get("pairs", []))):
            cr, cf = by_pair.get((i, "raw")), by_pair.get((i, "fix"))
            if not cr or not cf:
                continue
            fv, rv, drop = paired_vectors(cr, cf, "valid", "item_idx", "per_item")
            add(i, "json_valid", fv, rv, "structured", {"n_dropped": drop})

    # arm (a2) humaneval (only the SCORED raw carries pass/fail)
    if "humaneval_scored" in stages:
        d = stages["humaneval_scored"]
        _, by_pair = cond_index(d)
        for i in range(len(d.get("pairs", []))):
            cr, cf = by_pair.get((i, "raw")), by_pair.get((i, "fix"))
            if not cr or not cf:
                continue
            fv, rv, drop = paired_vectors(cr, cf, "passed", "task_id", "per_problem")
            add(i, "humaneval_pass1", fv, rv, "structured",
                {"n_dropped": drop, "n_timeout_raw": cr.get("n_timeout"),
                 "n_timeout_fix": cf.get("n_timeout")})
        out["exec_sandbox"] = d.get("exec_sandbox")
    elif "humaneval" in stages:
        out["humaneval_generated_not_scored"] = True

    out["contrasts"] = contrasts

    # ---- baselines (theta=1.0 reference), reported ------------------------
    base = {}
    for st, key, unit in (("open", "per_prompt", None), ("json", "per_item", None),
                          ("humaneval_scored", "per_problem", None)):
        if st not in stages:
            continue
        bl, _ = cond_index(stages[st])
        c = bl.get("baseline_raw")
        if not c:
            continue
        rows = c.get(key, [])
        if st == "open":
            base["open_theta1"] = {m: mean([r.get(m) for r in rows])
                                   for m in ("rep_rate", "distinct1", "distinct2", "quality")}
            if base["open_theta1"].get("quality") is None and c.get("quality_point") is not None:
                base["open_theta1"]["quality"] = c["quality_point"]      # MAUVE: set-level scalar
        elif st == "json":
            base["json_theta1_valid"] = mean([r.get("valid") for r in rows])
        else:
            base["humaneval_theta1_pass1"] = mean([r.get("passed") for r in rows])
    out["baseline_theta1"] = base

    # ---- FROZEN VERDICT (PREREG §6/§6a) ----------------------------------
    out["verdict"], out["verdict_reason"] = frozen_verdict(out)
    anchor = [c for c in contrasts if c.get("is_deployed_anchor")]
    out["deployed_anchor"] = {
        "available": any(p.get("is_deployed_anchor") for p in out["pairs"]),
        "premise": [e for e in premise if e.get("is_deployed_anchor")],
        "gauge": [g for g in gauge if g.get("is_deployed_anchor")],
        "contrasts": [c for c in anchor if c["class"] != "reported_only"],
    }
    return out


def frozen_verdict(out):
    """Evaluation order (PREREG §6a): INVALID -> REFUTED -> LOSES -> DOMINANT -> TIES -> FAVOURS_FIX."""
    if not out["controls_pass"]:
        return "INVALID", f"mandatory control(s) failed: {out['controls_failed']} (debug the harness, do not interpret)"
    if not out["premise_available"]:
        return "INCOMPLETE", "no `pairs` stage: the matched-suppression premise is unverified (never granted silently)"
    if not out["surviving_pairs"]:
        return "INVALID", "no pair survives the |rep_rate(raw)-rep_rate(fix)| <= 0.01 premise gate"
    if out["gauge_gate"] == "MISSING":
        return "INCOMPLETE", "no `gauge` stage: the c=+-5 no-regression gate is unverified (never granted silently)"
    if out["gauge_gate"] == "FAIL":
        f = out["gauge_gate_failures"]
        return "REFUTED", ("gauge gate FAILED at " +
                           "; ".join(f"theta_raw={g['theta_raw']:g} (flip_fix={g['flip_fix']:.6g}, "
                                     f"flip_raw={g['flip_raw']:.6g}) -> {g['failure_mode']}" for g in f) +
                           " -- hard REFUTED regardless of quality metrics")

    scored = [c for c in out["contrasts"]
              if c["class"] in ("structured", "open_ended") and c["pair_idx"] in out["surviving_pairs"]]
    if not scored:
        return "INCOMPLETE", "premise + gauge gate pass, but no quality stage (open/json/humaneval_scored) is present"

    losses = [c for c in scored if c["sign"] == -1]
    if losses:
        names = ", ".join(f"{c['metric']}@theta_raw={c['theta_raw']:g} "
                          f"(diff {c['diff']:+.4f}, CI [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}])" for c in losses)
        return "LOSES", f"the fix is significantly worse on: {names}"

    struct = [c for c in scored if c["class"] == "structured"]
    if struct:
        pairs_with = sorted({c["pair_idx"] for c in struct})
        all_ge = all((c["diff"] is not None and c["diff"] >= 0) for c in struct)
        wins = [c for c in struct if c["sign"] == 1 and not c["saturated"]]
        opens_bad = [c for c in scored if c["class"] == "open_ended" and c["sign"] == -1]
        if all_ge and wins and not opens_bad and set(pairs_with) == set(out["surviving_pairs"]):
            return "DOMINANT", ("fix >= raw on every structured metric at every surviving pair; CI excludes 0 "
                                "in the fix's favour on " +
                                ", ".join(f"{c['metric']}@theta_raw={c['theta_raw']:g}" for c in wins) +
                                "; no open-ended metric significantly worse")

    if all(c["sign"] == 0 for c in scored):
        return "TIES", ("every paired CI includes 0 -- gauge invariance at no measurable cost "
                        "(still a viable recommendation)")

    favs = [c for c in scored if c["sign"] == 1]
    return "FAVOURS_FIX", ("not LOSES and not TIES, but the DOMINANT conditions are unmet "
                           "(no structured arm, or structured CIs include 0). Fix significantly better on: " +
                           ", ".join(f"{c['metric']}@theta_raw={c['theta_raw']:g}" for c in favs))


# ---------------------------------------------------------------------------- report
def fmt(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def esc(s):
    """escape markdown table cell separators (verdict reasons contain |...| absolute-value bars)"""
    return str(s).replace("|", "\\|").replace("\n", " ")


def fmt_ci(c):
    if c["ci"][0] is None:
        return "—"
    star = "**" if c["excludes0"] else ""
    return f"{star}[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]{star}"


def write_report(models, files, out_dir):
    L = ["# MATCHED — matched-SUPPRESSION head-to-head (quality at equal anti-repetition strength)\n"]
    L.append(f"Raws analysed: {len(files)} · models: {', '.join(sorted(models)) or '(none)'}\n")
    L.append("Pairs are FIXCAL `MATCHED` pairs on the primary metric `rep_rate`; the θ′ values were "
             "read from the FIXCAL summary at run time and asserted against the frozen FIXCAL table.\n")

    L.append("\n## Stage coverage (partial runs are reported, never silently completed)\n")
    L.append("| model | pairs | gauge | open | json | humaneval (gen) | humaneval (scored) |")
    L.append("|---|:--:|:--:|:--:|:--:|:--:|:--:|")
    for s in sorted(models):
        p = models[s]["stages_present"]
        L.append("| `" + s + "` | " + " | ".join("✓" if st in p else "·" for st in
                 ("pairs", "gauge", "open", "json", "humaneval", "humaneval_scored")) + " |")
    for s in sorted(models):
        if models[s].get("humaneval_generated_not_scored"):
            L.append(f"\n> `{s}`: HumanEval completions were **generated but not executed** — pass@1 is "
                     f"absent from the verdict until the opt-in `--exec` scoring step is run "
                     f"(PREREG §8).")

    L.append("\n## Frozen decision rule (PREREG.md §6, verbatim)\n")
    L.append(FROZEN_RULE)
    L.append("\n\n*(Operationalisation, frozen in PREREG §6a: contrast = mean over units of (fix − raw) "
             "oriented so positive = fix better; paired bootstrap B=10 000, seed 0, 95% percentile CI; "
             "higher-is-better = json_valid, humaneval_pass1, quality, distinct1, distinct2; rep_rate is the "
             "MATCHING CHECK and longest_run is reported only — neither is scored as quality; evaluation "
             "order INVALID → REFUTED → LOSES → DOMINANT → TIES → FAVOURS_FIX.)*\n")

    L.append("\n## Verdicts\n")
    L.append("| model | verdict | why |")
    L.append("|---|---|---|")
    for s in sorted(models):
        m = models[s]
        L.append(f"| `{s}` | **{m['verdict']}** | {esc(m['verdict_reason'])} |")

    L.append("\n## 0. Mandatory controls (PROTOCOL §3)\n")
    L.append("| model | check | value |")
    L.append("|---|---|---|")
    for s in sorted(models):
        for k, v in models[s]["controls"].items():
            L.append(f"| `{s}` | {k} | {v} |")
    for s in sorted(models):
        dc = models[s].get("determinism_cross_check")
        if dc:
            L.append(f"| `{s}` | determinism (pairs vs open generations) | "
                     f"{dc['n_mismatched_generations']} mismatched |")

    L.append("\n## Stage 0a — premise: are the pairs actually matched? (|Δrep_rate| ≤ 0.01)\n")
    L.append("| model | θ_raw | θ′_fix | rep_rate raw | rep_rate fix | \\|Δ\\| | status | FIXCAL raw drift |")
    L.append("|---|--:|--:|--:|--:|--:|:--:|--:|")
    for s in sorted(models):
        for e in models[s]["premise"]:
            tag = " **(deployed anchor)**" if e["is_deployed_anchor"] else ""
            L.append(f"| `{s}` | {e['theta_raw']:g}{tag} | {e['theta_fix']:.4f} | {fmt(e['rep_rate_raw'])} | "
                     f"{fmt(e['rep_rate_fix'])} | {fmt(e['abs_delta'])} | {e['status']} | "
                     f"{fmt(e['fixcal_raw_drift'])} |")
        if not models[s]["premise"]:
            L.append(f"| `{s}` | — | — | — | — | — | (pairs stage absent) | — |")

    L.append("\n## Stage 0b — gauge no-regression gate (A1 c=±5 flip-rate at every pair)\n")
    L.append("flip(fix) must be **exactly 0** and flip(raw) **> 0**. Failure = hard REFUTED.\n")
    L.append("| model | θ_raw | θ′_fix | flip raw | flip fix | status | sub-diagnosis (reported, not a re-gate) |")
    L.append("|---|--:|--:|--:|--:|:--:|---|")
    for s in sorted(models):
        for g in models[s]["gauge"]:
            tag = " **(deployed anchor)**" if g["is_deployed_anchor"] else ""
            L.append(f"| `{s}` | {g['theta_raw']:g}{tag} | {g['theta_fix']:.4f} | {g['flip_raw']:.6f} | "
                     f"{g['flip_fix']:.6f} | {g['status']} | {esc(g.get('failure_mode') or '—')} |")
        if not models[s]["gauge"]:
            L.append(f"| `{s}` | — | — | — | — | (gauge stage absent) | — |")

    L.append("\n## CORE — quality contrasts at equal suppression (fix − raw, positive = fix better)\n")
    L.append("**bold CI** = excludes 0. Rows on MISMATCHED pairs are excluded from the verdict and marked.\n")
    for s in sorted(models):
        m = models[s]
        if not m["contrasts"]:
            L.append(f"\n### `{s}` — no quality stage present\n")
            continue
        L.append(f"\n### `{s}`  ({m.get('model_id')}, {m.get('dtype')})\n")
        if m.get("quality_measure"):
            L.append(f"open-ended quality measure: **{m['quality_measure']}** "
                     f"({m.get('quality_detail')})\n")
        if m.get("quality_fallback_reason"):
            L.append(f"quality fallback chain: {m['quality_fallback_reason']}\n")
        if m.get("baseline_theta1"):
            L.append(f"θ=1.0 unpenalized reference: `{json.dumps(m['baseline_theta1'])}`\n")
        L.append("| θ_raw | metric | class | raw | fix | diff (fix−raw) | 95% CI | n | note |")
        L.append("|--:|---|---|--:|--:|--:|---|--:|---|")
        for c in m["contrasts"]:
            notes = []
            if c.get("saturated"):
                notes.append("SATURATED")
            if c["surviving"] is False:
                notes.append("EXCLUDED (MISMATCHED pair)")
            if c["class"] == "reported_only":
                notes.append("reported only, not a quality metric")
            if c.get("is_deployed_anchor"):
                notes.append("deployed anchor")
            if c.get("n_dropped"):
                notes.append(f"{c['n_dropped']} units dropped")
            L.append(f"| {c['theta_raw']:g} | {c['metric']} | {c['class']} | {fmt(c['mean_raw'])} | "
                     f"{fmt(c['mean_fix'])} | {fmt(c['diff'])} | {fmt_ci(c)} | {c['n']} | "
                     f"{'; '.join(notes)} |")

    L.append("\n## Deployed anchor (θ_raw = 1.10) — reported separately; it carries the downstream proposal decision\n")
    for s in sorted(models):
        a = models[s]["deployed_anchor"]
        if not a["available"]:
            L.append(f"- `{s}`: **no 1.10 pair exists** (UNREACHABLE in FIXCAL) — this model cannot speak "
                     f"to the deployed anchor. Documented, never substituted.")
            continue
        prem = a["premise"][0]["status"] if a["premise"] else "—"
        gau = a["gauge"][0]["status"] if a["gauge"] else "—"
        L.append(f"- `{s}`: premise **{prem}**, gauge **{gau}**" +
                 ("; " + ", ".join(f"{c['metric']} {fmt(c['diff'])} {fmt_ci(c)}" for c in a["contrasts"])
                  if a["contrasts"] else "; no quality contrast yet"))

    L.append("\n## Framing (FIXCAL AUTHOR RULING rider 1 — not a gate)\n")
    L.append("The 2.5×–6.6× per-model spread of θ′ at the same θ_raw *is A1 measured in suppression space*: "
             "raw θ=1.1 was never one intervention to begin with, so \"calibrate per model\" is not a defect "
             "of the fix relative to raw — it makes explicit what raw was already doing implicitly. "
             "Per rider 2, no gauge-artifact *mechanism* claim is made here (that is ZPREACH).\n")
    return "\n".join(L) + "\n"


def main():
    global BOOT, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--raws", default="results/matched_strength/raw_*.json")
    ap.add_argument("--out-dir", default="results/matched_strength")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    BOOT, SEED = args.boot, args.seed

    by_model, files = load_raws(args.raws)
    if not files:
        print(f"no raws matched {args.raws}")
    models = {s: analyze_model(s, st) for s, st in by_model.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    rep = write_report(models, files, args.out_dir)
    open(os.path.join(args.out_dir, "REPORT.md"), "w").write(rep)
    summ = {"experiment": "MATCHED", "request": "MATCHED", "n_raws": len(files), "raws": files,
            "boot": BOOT, "seed": SEED, "rep_rate_tol": REP_RATE_TOL,
            "frozen_rule": FROZEN_RULE, "models": models,
            "verdicts": {s: models[s]["verdict"] for s in models}}
    json.dump(summ, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    print(f"wrote {args.out_dir}/REPORT.md + summary.json")
    for s in sorted(models):
        print(f"  {s:>18s}: {models[s]['verdict']:<12s} stages={models[s]['stages_present']}")


if __name__ == "__main__":
    main()
