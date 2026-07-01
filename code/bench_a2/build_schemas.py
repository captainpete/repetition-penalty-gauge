#!/usr/bin/env python3
"""Part A step 1 — build the JSONSchemaBench manifest for the A2 JSON-validity rerun.

Canonical source: HF dataset `epfl-dlab/JSONSchemaBench` (guidance-ai / EPFL-DLAB;
github.com/guidance-ai/jsonschemabench). ~10K real-world JSON schemas across splits.

Procedure (deterministic, seed 0):
  pool = Github_easy(train+val+test) then Github_medium(train+val+test)  [in that order]
  shuffle(pool) with random.Random(0)
  keep a schema iff:
    (a) json.loads parses it and it is a JSON object at the top level,
    (b) it declares an object instance: "object" in (type field),
    (c) jsonschema can *compile* it: validator_for(schema).check_schema(schema) passes,
    (d) no remote ($ref to http[s]) reference (unresolvable at validate() time -> would
        spuriously invalidate every instance); internal #/... refs are allowed,
    (e) the schema string serializes to <= 600 GPT-2 tokens (must fit the prompt).
  take the first 200 that pass.

Writes results/bench_a2/schemas.json (manifest: list of {split, id, schema, gpt2_tokens}).
"""
import os, json, random, argparse
os.environ.setdefault("HF_HOME", "$HF_HUB_CACHE")

MAX_GPT2_TOKENS = 600
N_TARGET = 200
POOL_SPLITS = ["train", "val", "test"]
POOL_CONFIGS = ["Github_easy", "Github_medium"]


def has_remote_ref(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$ref" and isinstance(v, str) and v.lower().startswith(("http://", "https://")):
                return True
            if has_remote_ref(v):
                return True
    elif isinstance(obj, list):
        return any(has_remote_ref(x) for x in obj)
    return False


def is_object_schema(schema):
    t = schema.get("type")
    if t == "object":
        return True
    if isinstance(t, list) and "object" in t:
        return True
    # heuristic: has properties/required and no conflicting scalar type -> object
    if t is None and ("properties" in schema or "required" in schema):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench_a2/schemas.json")
    ap.add_argument("--n", type=int, default=N_TARGET)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from jsonschema.validators import validator_for
    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("gpt2")

    pool = []  # (config, split, unique_id, schema_str)
    for cfg in POOL_CONFIGS:
        ds = load_dataset("epfl-dlab/JSONSchemaBench", cfg)
        for sp in POOL_SPLITS:
            if sp not in ds:
                continue
            for r in ds[sp]:
                pool.append((cfg, sp, r["unique_id"], r["json_schema"]))
    print(f"pool size (Github_easy+medium, all splits): {len(pool)}")

    random.Random(args.seed).shuffle(pool)

    manifest = []
    stats = {"parse_fail": 0, "not_object": 0, "compile_fail": 0, "remote_ref": 0, "too_long": 0, "kept": 0}
    for cfg, sp, uid, sstr in pool:
        if len(manifest) >= args.n:
            break
        try:
            schema = json.loads(sstr)
        except Exception:
            stats["parse_fail"] += 1
            continue
        if not isinstance(schema, dict):
            stats["parse_fail"] += 1
            continue
        if not is_object_schema(schema):
            stats["not_object"] += 1
            continue
        try:
            cls = validator_for(schema)
            cls.check_schema(schema)
        except Exception:
            stats["compile_fail"] += 1
            continue
        if has_remote_ref(schema):
            stats["remote_ref"] += 1
            continue
        ntok = len(gpt2(sstr)["input_ids"])
        if ntok > MAX_GPT2_TOKENS:
            stats["too_long"] += 1
            continue
        draft = schema.get("$schema", cls.META_SCHEMA.get("$id", "unknown"))
        manifest.append({"config": cfg, "split": sp, "id": uid,
                         "validator": cls.__name__, "draft": draft,
                         "gpt2_tokens": ntok, "schema": schema})
        stats["kept"] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"source": "epfl-dlab/JSONSchemaBench",
               "configs": POOL_CONFIGS, "splits": POOL_SPLITS,
               "seed": args.seed, "n": len(manifest),
               "max_gpt2_tokens": MAX_GPT2_TOKENS,
               "filter_stats": stats,
               "schemas": manifest}, open(args.out, "w"), indent=1)
    print("filter stats:", stats)
    from collections import Counter
    print("by config:", Counter(m["config"] for m in manifest))
    print("by draft:", Counter(m["draft"] for m in manifest))
    print(f"wrote {len(manifest)} schemas -> {args.out}")


if __name__ == "__main__":
    main()
