#!/usr/bin/env python3
"""A2 JSON schema-validity analysis for the llama.cpp smoke replication.

first_json, json_valid, and the JSON_TASKS schema type-checks are copied
VERBATIM from code/run_a2_downstream.py so the numbers are directly comparable
to the HF measurement. Scores llama.cpp-generated text per (theta, penalty_last_n)."""
import json, sys
from collections import defaultdict

# ---- VERBATIM from run_a2_downstream.py: JSON_TASKS schemas (type checks) ----
JSON_TASKS = [
    ("a user with fields name (string), age (integer), email (string)",
     {"name": str, "age": int, "email": str}),
    ("a product with fields title (string), price (number), in_stock (boolean)",
     {"title": str, "price": (int, float), "in_stock": bool}),
    ("a book with fields title (string), author (string), year (integer), tags (array of strings)",
     {"title": str, "author": str, "year": int, "tags": list}),
    ("a city with fields name (string), population (integer), country (string), capital (boolean)",
     {"name": str, "population": int, "country": str, "capital": bool}),
    ("an event with fields name (string), date (string), attendees (integer), virtual (boolean)",
     {"name": str, "date": str, "attendees": int, "virtual": bool}),
    ("a car with fields make (string), model (string), year (integer), electric (boolean)",
     {"make": str, "model": str, "year": int, "electric": bool}),
]


# ---- VERBATIM from run_a2_downstream.py ----
def first_json(text):
    i = text.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(text[i:j + 1])
                except Exception: return None
    return None


def json_valid(obj, schema):
    if not isinstance(obj, dict):
        return False
    for k, t in schema.items():
        if k not in obj or not isinstance(obj[k], t) or (t is int and isinstance(obj[k], bool)):
            return False
    return True
# ---- end verbatim ----


raw = sys.argv[1] if len(sys.argv) > 1 else \
    "../../results/smoke_llamacpp/a2_raw.json"
out = sys.argv[2] if len(sys.argv) > 2 else \
    "../../results/smoke_llamacpp/a2_summary.json"

d = json.load(open(raw))
buckets = defaultdict(lambda: [0, 0])    # (theta, last_n) -> [valid, total]
for r in d["records"]:
    schema = JSON_TASKS[r["schema_idx"]][1]
    ok = json_valid(first_json(r["text"]), schema)
    key = (round(r["theta"], 4), r["penalty_last_n"])
    buckets[key][1] += 1
    buckets[key][0] += int(ok)

thetas = sorted({k[0] for k in buckets})
lastns = sorted({k[1] for k in buckets}, reverse=True)   # whole-context first
cond_label = {max(lastns): "whole_context", min(lastns): "default_64"} if len(lastns) > 1 else {}

table = {}
for ln in lastns:
    row = {}
    for th in thetas:
        v, t = buckets[(th, ln)]
        row[str(th)] = {"valid_rate": v / t if t else 0.0, "valid": v, "total": t}
    table[str(ln)] = row

summary = {"model": d["model"], "max_new": d["max_new"], "json_reps": d.get("json_reps"),
           "condition_label": {str(k): v for k, v in cond_label.items()},
           "valid_rate": table,
           "hf_reference_valid_by_theta": {"1.0": 1.0, "1.1": 1.0, "1.3": 0.0}}
json.dump(summary, open(out, "w"), indent=2)

print(f"model: {d['model']}  (max_new={d['max_new']}, json_reps={d.get('json_reps')})")
print("JSON schema-valid rate  (n per cell = "
      f"{buckets[(thetas[0], lastns[0])][1]})")
hdr = "penalty_last_n     | " + " | ".join(f"theta={th}" for th in thetas)
print(hdr)
for ln in lastns:
    lbl = cond_label.get(ln, "")
    cells = " | ".join(f"{table[str(ln)][str(th)]['valid_rate']:.3f}   " for th in thetas)
    print(f"{ln:>6} ({lbl:<13})| {cells}")
print("HF reference (all-seen): theta 1.0/1.1/1.3 -> 1.0 / 1.0 / 0.0")
