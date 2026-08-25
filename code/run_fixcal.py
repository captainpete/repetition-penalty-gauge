#!/usr/bin/env python3
"""FIXCAL (FIXCAL) — the migration map theta_raw -> theta'_fix at MATCHED repetition suppression.

Frozen protocol in PREREG.md. Reuses the fix_loopcheck harness UNMODIFIED (code/run_a1_loopcheck.py:
greedy + degen, on run_a1.PROMPTS). Three things happen per model:

  (a) RAW dense grid  theta in {1.02,1.05,1.08,1.1,1.15,1.2,1.3} (+ theta=1.0 baseline), all three
      suppression metrics (rep_rate primary, longest_run, distinct2).
  (b) BISECTION (not a fixed theta' grid): per raw anchor x per metric, bisect theta' in [1, theta'_max]
      (default 10) for the theta' whose FIX metric matches that anchor's. First-class UNREACHABLE
      verdict if the fix at theta'_max still falls short. Every theta' evaluation is cached.
  (c) CLOSED-FORM required theta': under the fix all logits are log-probs (negative) so HF's sign branch
      always MULTIPLIES, theta*log p = log(p^theta); the greedy argmax leaves a penalized top token only
      when p_top^theta < p_comp, i.e. theta'_required = ln(p_runner)/ln(p_top). Instrumented at
      theta=1.0 over the loop tokens (amendment A / protocol refinement refinement).

GPU (real run — 4 models, then the analyzer):
        --model gpt2 --dtype float32 --out results/fix_calibration/raw_gpt2.json
        --model EleutherAI/pythia-2.8b --dtype float32 --out results/fix_calibration/raw_pythia-2.8b.json
        --model Qwen/Qwen2.5-7B --dtype bfloat16 --out results/fix_calibration/raw_Qwen2.5-7B.json
        --model Qwen/Qwen2.5-Coder-7B --dtype bfloat16 \
      --out results/fix_calibration/raw_Qwen2.5-Coder-7B.json
  python code/analyze_fixcal.py \
      --raws 'results/fix_calibration/raw_*.json' --out-dir results/fix_calibration
CPU smoke:
  python run_fixcal.py --device cpu --model gpt2 --max-new 48 --limit 3 \
      --thetas 1.05,1.1 --theta-max 10 --out /tmp/FC/raw_gpt2.json
"""
import os, sys, json, math, time, argparse

os.environ.setdefault("HF_HUB_CACHE", "/hf/hub")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                              # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from run_a1 import PROMPTS                                # noqa: E402
from run_a1_loopcheck import greedy, degen                # noqa: E402  (fix_loopcheck harness, unmodified)

METRICS = ("rep_rate", "longest_run", "distinct2")
# suppression = SIGN * metric, so suppression INCREASES with theta for every metric
SIGN = {"rep_rate": -1.0, "longest_run": -1.0, "distinct2": +1.0}
# frozen per-metric bisection tolerances (PREREG §3b)
TOL = {"rep_rate": 0.01, "distinct2": 0.01, "longest_run": 0.25}
LOG_P_FLOOR = math.log(1e-12)      # p_runner <= 1e-12  -> theta_required := +inf
LOG_P_TOP_CEIL = -1e-6             # p_top >= 1 - 1e-6   -> theta_required := +inf
INF = float("inf")


# ----------------------------------------------------------------------------- sweeps + cache
def sweep(model, pids, theta, max_new, device, fix):
    """One full generation sweep over all prompts -> mean metrics (+ per-prompt, for the controls)."""
    per = []
    for p in pids:
        g = greedy(model, p, theta, max_new, device, fix)
        a, b, c = degen(p, g)
        per.append({"rep_rate": a, "distinct2": b, "longest_run": c})
    n = len(per)
    out = {m: sum(r[m] for r in per) / n for m in METRICS}
    out["per_prompt"] = per
    return out


class FixEvaluator:
    """theta' -> fix-metrics, cached (each miss is a full sweep; bisection midpoints are dyadic and
    coincide across anchors/metrics, so the cache carries most of the search)."""

    def __init__(self, model, pids, max_new, device):
        self.model, self.pids, self.max_new, self.device = model, pids, max_new, device
        self.cache, self.n_evals = {}, 0

    @staticmethod
    def key(theta):
        return f"{theta:.9f}"

    def __call__(self, theta):
        k = self.key(theta)
        if k not in self.cache:
            t0 = time.time()
            self.cache[k] = sweep(self.model, self.pids, theta, self.max_new, self.device, True)
            self.n_evals += 1
            r = self.cache[k]
            print(f"    [fix eval {self.n_evals}] theta'={theta:.6f}  rep_rate={r['rep_rate']:.4f} "
                  f"longest_run={r['longest_run']:.3f} distinct2={r['distinct2']:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        return self.cache[k]


# ----------------------------------------------------------------------------- bisection
def bisect_match(evalfix, metric, target, theta_max, iters):
    """Find theta' whose FIX `metric` matches `target` (that anchor's RAW value).

    Returns a dict with verdict in {MATCHED, UNREACHABLE, TRIVIAL} (PREREG §3b). Never substitutes."""
    sgn, tol = SIGN[metric], TOL[metric]
    s_t = sgn * target
    m_lo = evalfix(1.0)[metric]
    m_hi = evalfix(theta_max)[metric]
    n0 = evalfix.n_evals

    if sgn * m_lo >= s_t - tol:            # unpenalized fix already matches within tol
        return {"verdict": "TRIVIAL", "reachable": True, "theta_fix": 1.0, "achieved": m_lo,
                "delta": m_lo - target, "converged": True, "bracket": [1.0, 1.0],
                "iters": 0, "fresh_evals": evalfix.n_evals - n0}

    if sgn * m_hi < s_t - tol:             # ceiling short of the anchor by MORE than tol
        return {"verdict": "UNREACHABLE", "reachable": False, "theta_fix": None, "achieved": m_hi,
                "delta": m_hi - target, "shortfall": abs(m_hi - target), "converged": False,
                "bracket": [theta_max, None], "iters": 0, "fresh_evals": evalfix.n_evals - n0}

    # bracketable: sgn*m(1.0) < s_t - tol <= sgn*m(theta_max). Bisect toward the exact target so the
    # reported theta' is the SMALLEST one that matches, not the ceiling.
    lo, hi = 1.0, theta_max
    hi_val, it = m_hi, 0
    for it in range(1, iters + 1):
        mid = 0.5 * (lo + hi)
        mv = evalfix(mid)[metric]
        if abs(mv - target) <= tol:
            return {"verdict": "MATCHED", "reachable": True, "theta_fix": mid, "achieved": mv,
                    "delta": mv - target, "converged": True, "bracket": [lo, hi], "iters": it,
                    "fresh_evals": evalfix.n_evals - n0}
        if sgn * mv < s_t:
            lo = mid
        else:
            hi, hi_val = mid, mv
    # budget exhausted: report the smallest evaluated theta' that reaches at least the anchor's
    # suppression (the upper bracket), with the bracket carried.
    return {"verdict": "MATCHED", "reachable": True, "theta_fix": hi, "achieved": hi_val,
            "delta": hi_val - target, "converged": False, "bracket": [lo, hi], "iters": it,
            "fresh_evals": evalfix.n_evals - n0}


# ----------------------------------------------------------------------------- loop instrumentation
@torch.no_grad()
def greedy_instrumented(model, prompt_ids, max_new, device):
    """theta=1.0 (unpenalized) greedy, identical trajectory to greedy(..., theta=1.0, ...).
    Logs per loop token: p_top, p_runner (best non-argmax), p_runner_unseen (best token outside the
    penalized set -- the only competitor power-scaling can let through), and both theta'_required."""
    seen = set(prompt_ids)
    seen_idx = torch.tensor(sorted(seen), device=device, dtype=torch.long)
    cur = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    past, gen, recs, p_top_all = None, [], [], []
    gen_set = set()
    for pos in range(max_new):
        feed = cur if past is None else cur[:, -1:]
        out = model(feed, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        top = int(logp.argmax())
        lp_top = float(logp[top])
        p_top_all.append(math.exp(lp_top))

        is_loop = top in seen                              # == the repeat events rep_rate counts
        if is_loop:
            alt = logp.clone(); alt[top] = -INF
            runner = int(alt.argmax()); lp_run = float(alt[runner])
            uns = logp.clone(); uns[seen_idx] = -INF       # top is in seen -> masked out too
            u = int(uns.argmax()); lp_uns = float(uns[u])

            def theta_req(lp_c):
                if lp_top >= LOG_P_TOP_CEIL:               # p_top >= 1-1e-6 -> ln p_top ~ 0
                    return INF, "ptop_saturated"
                if lp_c <= LOG_P_FLOOR or lp_c == -INF:    # p_comp <= 1e-12
                    return INF, "prunner_underflow"
                return lp_c / lp_top, "ok"

            tr, tr_flag = theta_req(lp_run)
            tu, tu_flag = theta_req(lp_uns)
            recs.append({
                "pos": pos, "token": top,
                "p_top": math.exp(lp_top),
                "p_runner": math.exp(lp_run) if lp_run != -INF else 0.0,
                "p_runner_unseen": math.exp(lp_uns) if lp_uns != -INF else 0.0,
                "theta_required": None if tr == INF else tr, "theta_required_inf": tr == INF,
                "theta_required_flag": tr_flag,
                "theta_required_unseen": None if tu == INF else tu,
                "theta_required_unseen_inf": tu == INF, "theta_required_unseen_flag": tu_flag,
                "strict": top in gen_set,                  # loop token excluding prompt tokens
            })
        gen.append(top); gen_set.add(top)
        if top not in seen:
            seen.add(top)
            seen_idx = torch.cat([seen_idx, torch.tensor([top], device=device)])
        cur = torch.cat([cur, torch.tensor([[top]], device=device)], dim=1)
    return gen, recs, p_top_all


def quant(vals_finite, n_inf, q):
    """Quantile over the full population (finite values sorted first, infinities last).
    Returns None when the quantile lands in the infinite mass."""
    n = len(vals_finite) + n_inf
    if n == 0:
        return None
    i = min(n - 1, max(0, int(math.ceil(q * n)) - 1))
    return vals_finite[i] if i < len(vals_finite) else None


def dist_stats(vals, n_inf, theta_max):
    """Summary of a theta'_required population that may contain +inf."""
    v = sorted(vals)
    n = len(v) + n_inf
    gt_max = sum(1 for x in v if x > theta_max) + n_inf
    return {
        "n": n, "n_finite": len(v), "n_infinite": n_inf,
        "frac_infinite": (n_inf / n) if n else None,
        "median": quant(v, n_inf, 0.5), "p10": quant(v, n_inf, 0.10), "p90": quant(v, n_inf, 0.90),
        "median_is_infinite": quant(v, n_inf, 0.5) is None and n > 0,
        "mean_finite": (sum(v) / len(v)) if v else None,
        "min_finite": v[0] if v else None, "max_finite": v[-1] if v else None,
        "frac_gt_theta_max": (gt_max / n) if n else None,
    }


def conf_stats(vals):
    v = sorted(vals)
    if not v:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": len(v), "mean": sum(v) / len(v),
            "median": v[min(len(v) - 1, int(math.ceil(0.5 * len(v))) - 1)],
            "p10": v[min(len(v) - 1, max(0, int(math.ceil(0.10 * len(v))) - 1))],
            "p90": v[min(len(v) - 1, int(math.ceil(0.90 * len(v))) - 1)]}


def loop_summary(all_recs, all_ptop, n_gen, theta_max):
    def pick(strict):
        return [r for r in all_recs if (r["strict"] if strict else True)]

    out = {"n_gen_tokens": n_gen, "n_loop_tokens": len(all_recs),
           "loop_frac": len(all_recs) / n_gen if n_gen else None,
           "p_top_all_mean": sum(all_ptop) / len(all_ptop) if all_ptop else None,
           "n_ptop_saturated": sum(1 for r in all_recs if r["theta_required_flag"] == "ptop_saturated"),
           "n_prunner_underflow": sum(1 for r in all_recs
                                      if r["theta_required_flag"] == "prunner_underflow")}
    for tag, strict in (("", False), ("_strict", True)):
        rs = pick(strict)
        out[f"n_loop_tokens{tag}"] = len(rs)
        out[f"loop_frac{tag}"] = len(rs) / n_gen if n_gen else None
        out[f"p_top_loop{tag}"] = conf_stats([r["p_top"] for r in rs])
        for fld in ("theta_required", "theta_required_unseen"):
            fin = [r[fld] for r in rs if not r[f"{fld}_inf"]]
            ninf = sum(1 for r in rs if r[f"{fld}_inf"])
            out[f"{fld}{tag}"] = dist_stats(fin, ninf, theta_max)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--max-new", type=int, default=256, help="fix_loopcheck's 128 is a floor, not the setting")
    ap.add_argument("--thetas", default="1.02,1.05,1.08,1.1,1.15,1.2,1.3", help="RAW dense grid anchors")
    ap.add_argument("--theta-max", type=float, default=10.0, help="pre-registered bisection ceiling")
    ap.add_argument("--bisect-iters", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="use only first N prompts (smoke)")
    ap.add_argument("--out", default="results/fix_calibration/raw.json")
    args = ap.parse_args()

    anchors = sorted({float(x) for x in args.thetas.split(",") if float(x) != 1.0})
    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"
    pids = [tok(p)["input_ids"] for p in prompts]
    print(f"FIXCAL {args.model} (rev {rev}) device={args.device} dtype={args.dtype} "
          f"max_new={args.max_new} n_prompts={len(pids)} anchors={anchors} "
          f"theta_max={args.theta_max}", flush=True)

    evalfix = FixEvaluator(model, pids, args.max_new, args.device)

    # --- (0) controls: raw theta=1.0 baseline, fix theta'=1.0 no-op, instrumented theta=1.0
    print("  [controls]", flush=True)
    base_raw = sweep(model, pids, 1.0, args.max_new, args.device, False)
    base_fix = evalfix(1.0)
    noop_ok = all(
        abs(a[m] - b[m]) < 1e-12
        for a, b in zip(base_raw["per_prompt"], base_fix["per_prompt"]) for m in METRICS)
    print(f"    no-op control (raw th=1.0 == fix th'=1.0, per prompt, exactly): "
          f"{'PASS' if noop_ok else 'FAIL'}", flush=True)

    all_recs, all_ptop, n_gen, instr_per = [], [], 0, []
    for i, p in enumerate(pids):
        gen, recs, ptop = greedy_instrumented(model, p, args.max_new, args.device)
        a, b, c = degen(p, gen)
        instr_per.append({"rep_rate": a, "distinct2": b, "longest_run": c})
        for r in recs:
            r["prompt_idx"] = i
        all_recs += recs; all_ptop += ptop; n_gen += len(gen)
    instr_ok = all(abs(instr_per[i][m] - base_raw["per_prompt"][i][m]) < 1e-12
                   for i in range(len(pids)) for m in METRICS)
    print(f"    instrumentation control (instrumented th=1.0 == raw th=1.0 trajectory): "
          f"{'PASS' if instr_ok else 'FAIL'}", flush=True)
    loops = loop_summary(all_recs, all_ptop, n_gen, args.theta_max)
    tr = loops["theta_required"]
    ff = lambda x, p=3: "inf/na" if x is None else f"{x:.{p}f}"     # noqa: E731
    print(f"    loops: n={loops['n_loop_tokens']} frac={ff(loops['loop_frac'])}  in-loop p_top mean="
          f"{ff(loops['p_top_loop']['mean'], 4)} median={ff(loops['p_top_loop']['median'], 4)}  "
          f"theta_required median={ff(tr['median'], 2)} p10={ff(tr['p10'], 2)} "
          f"p90={ff(tr['p90'], 2)}  frac_inf={ff(tr['frac_infinite'])}  "
          f"frac>{args.theta_max:g}={ff(tr['frac_gt_theta_max'])}", flush=True)

    # --- (a) RAW dense grid
    print("  [raw dense grid]", flush=True)
    raw_grid = {f"{1.0:.9f}": base_raw}
    for th in anchors:
        t0 = time.time()
        raw_grid[f"{th:.9f}"] = sweep(model, pids, th, args.max_new, args.device, False)
        r = raw_grid[f"{th:.9f}"]
        print(f"    theta={th:g}  rep_rate={r['rep_rate']:.4f} longest_run={r['longest_run']:.3f} "
              f"distinct2={r['distinct2']:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    # --- (b) bisection: per anchor x per metric, independently
    print(f"  [bisection] theta' in [1, {args.theta_max}], <= {args.bisect_iters} iters, "
          f"tol={TOL}", flush=True)
    evalfix(args.theta_max)                                # ceiling, evaluated once, cached
    bisection = {m: [] for m in METRICS}
    for m in METRICS:
        for th in anchors:
            target = raw_grid[f"{th:.9f}"][m]
            row = bisect_match(evalfix, m, target, args.theta_max, args.bisect_iters)
            row.update({"theta_raw": th, "metric": m, "target": target,
                        "baseline_metric": base_raw[m],
                        "raw_suppresses": bool(SIGN[m] * target > SIGN[m] * base_raw[m])})
            bisection[m].append(row)
            tf = "n/a" if row["theta_fix"] is None else f"{row['theta_fix']:.4f}"
            print(f"    match on {m:<12} theta_raw={th:<5g} target={target:.4f} -> "
                  f"{row['verdict']:<11} theta'={tf} achieved={row['achieved']:.4f} "
                  f"(iters={row['iters']}, fresh={row['fresh_evals']})", flush=True)

    out = {
        "experiment": "FIXCAL", "req": "FIXCAL", "model": args.model, "revision": rev,
        "device": args.device, "dtype": args.dtype, "max_new": args.max_new,
        "n_prompts": len(pids), "anchors": anchors, "theta_max": args.theta_max,
        "bisect_iters": args.bisect_iters, "tol": TOL, "metrics": list(METRICS),
        "primary_metric": "rep_rate",
        "controls": {"noop_raw1_eq_fix1": bool(noop_ok),
                     "instrumented_eq_raw1": bool(instr_ok),
                     "raw_theta1": {m: base_raw[m] for m in METRICS},
                     "fix_theta1": {m: base_fix[m] for m in METRICS},
                     "instrumented_theta1": {m: sum(r[m] for r in instr_per) / len(instr_per)
                                             for m in METRICS}},
        "raw_grid": raw_grid,
        "fix_cache": evalfix.cache,
        "n_fix_evals": evalfix.n_evals,
        "bisection": bisection,
        "loop_stats": loops,
        "loop_tokens": all_recs,
        "elapsed_s": time.time() - t_start,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"  controls: noop={noop_ok} instr={instr_ok}; {evalfix.n_evals} fix sweeps, "
          f"{len(anchors) + 1} raw sweeps, {time.time() - t_start:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
