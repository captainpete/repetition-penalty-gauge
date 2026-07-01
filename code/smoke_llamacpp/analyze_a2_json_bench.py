#!/usr/bin/env python3
"""A2 benchmark analysis for the llama.cpp replication on JSONSchemaBench schemas.

Scores llama.cpp-generated text per (theta, penalty_last_n) against the ACTUAL
benchmark schemas (results/bench_a2/schemas.json). first_json and schema_valid are
copied VERBATIM from code/bench_a2/run_json_bench.py so rates are directly comparable
to the HF-side benchmark numbers and to the vLLM bench run.

Usage:
  python3 analyze_a2_json_bench.py '<glob of a2 raw chunk jsons>' <out summary.json> [manifest]
"""
import glob, json, sys
from collections import defaultdict


# ---- VERBATIM from code/bench_a2/run_json_bench.py ----
def first_json(text):
    """first complete brace-matched JSON object -> (obj, substring) or (None, None)."""
    i = text.find("{")
    if i < 0:
        return None, None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                sub = text[i:j + 1]
                try:
                    return json.loads(sub), sub
                except Exception:
                    return None, sub
    return None, None


def schema_valid(obj, schema):
    """structural validation against the actual JSON Schema (draft auto-detected)."""
    if obj is None:
        return False, "no_json"
    from jsonschema.validators import validator_for
    try:
        cls = validator_for(schema)
        v = cls(schema)
        errs = list(v.iter_errors(obj))
        return (len(errs) == 0), (None if not errs else errs[0].message[:120])
    except Exception as e:
        return False, "validator_error:" + repr(e)[:100]
# ---- end verbatim ----


def main():
    raw_glob = sys.argv[1] if len(sys.argv) > 1 else \
        "results/smoke_llamacpp_bench/a2_chunks/*.json"
    out = sys.argv[2] if len(sys.argv) > 2 else \
        "results/smoke_llamacpp_bench/a2_summary.json"
    manifest = sys.argv[3] if len(sys.argv) > 3 else \
        "results/bench_a2/schemas.json"

    schemas = json.load(open(manifest))["schemas"]
    paths = sorted(glob.glob(raw_glob))
    if not paths:
        sys.exit(f"no raw files match {raw_glob}")

    model = max_new = None
    buckets = defaultdict(lambda: [0, 0])   # (theta, last_n) -> [valid, total]
    fails = defaultdict(lambda: defaultdict(int))
    seen = set()
    for p in paths:
        d = json.load(open(p))
        model, max_new = d["model"], d["max_new"]
        for r in d["records"]:
            key = (round(r["theta"], 4), r["penalty_last_n"], r["schema_idx"], r.get("rep_idx", 0))
            if key in seen:
                continue    # tolerate overlapping/re-run chunks
            seen.add(key)
            obj, sub = first_json(r["text"])
            ok, reason = schema_valid(obj, schemas[r["schema_idx"]]["schema"])
            b = buckets[(key[0], key[1])]
            b[1] += 1
            b[0] += int(ok)
            if not ok:
                fails[(key[0], key[1])][reason or "invalid"] += 1

    thetas = sorted({k[0] for k in buckets})
    lastns = sorted({k[1] for k in buckets}, reverse=True)
    cond_label = {max(lastns): "whole_context", min(lastns): "default_64"} if len(lastns) > 1 else {}

    table = {}
    for ln in lastns:
        row = {}
        for th in thetas:
            v, t = buckets[(th, ln)]
            row[str(th)] = {"valid_rate": v / t if t else 0.0, "valid": v, "total": t}
        table[str(ln)] = row

    summary = {"model": model, "max_new": max_new, "manifest": manifest,
               "n_raw_files": len(paths), "n_records": len(seen),
               "condition_label": {str(k): v for k, v in cond_label.items()},
               "valid_rate": table,
               "top_fail_reasons": {f"{th}|{ln}": sorted(fails[(th, ln)].items(),
                                                         key=lambda kv: -kv[1])[:5]
                                    for th in thetas for ln in lastns},
               "hf_reference_valid_by_theta_raw": {"1.0": 0.970, "1.1": 0.955, "1.3": 0.225}}
    json.dump(summary, open(out, "w"), indent=2)

    print(f"model: {model}  (max_new={max_new}, n_records={len(seen)})")
    hdr = "penalty_last_n       | " + " | ".join(f"theta={th}" for th in thetas)
    print(hdr)
    for ln in lastns:
        lbl = cond_label.get(ln, "")
        cells = " | ".join(f"{table[str(ln)][str(th)]['valid_rate']:.3f}   " for th in thetas)
        print(f"{ln:>6} ({lbl:<13}) | {cells}")
    print("HF-side benchmark reference (raw): theta 1.0/1.1/1.3 -> 0.970 / 0.955 / 0.225")


if __name__ == "__main__":
    main()
