#!/usr/bin/env python3
"""A2 JSON-validity on the 200 JSONSchemaBench schemas (results/bench_a2/schemas.json),
measured INSIDE vLLM using the STOCK SamplingParams.repetition_penalty knob (default V1
engine) — raw operator only; the "fix" column comes from the HF-side benchmark run.

Protocol MIRRORS code/bench_a2/run_json_bench.py exactly: build_prompt / first_json /
schema_valid are copied VERBATIM (greedy, max_new=512, stop ["\n\n", "```"], jsonschema
validation against the actual schema) so rates are comparable to the HF-side numbers
(raw 0.970/0.955/0.225 at theta=1.0/1.1/1.3).

Checkpointed per theta: each theta writes <out>.theta<t>.part.json and completed thetas
are skipped on resume; the final merged file is written when all thetas are done.

  .venv/bin/python run_a2_vllm_bench.py
"""
import os, json, argparse
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def build_prompt(schema):
    s = json.dumps(schema, indent=2)
    return ("Output ONLY a single JSON object that conforms to this JSON Schema. "
            "Do not include any explanation.\n"
            f"JSON Schema:\n{s}\nJSON object:\n")
# ---- end verbatim ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--manifest", default="results/bench_a2/schemas.json")
    ap.add_argument("--thetas", default="1.0,1.1,1.3")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="first N schemas (smoke)")
    ap.add_argument("--out", default="results/smoke_vllm_bench/a2_raw.json")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.82)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()
    thetas = [float(x) for x in args.thetas.split(",")]

    man = json.load(open(args.manifest))
    schemas = man["schemas"]
    if args.limit:
        schemas = schemas[:args.limit]
    prompts = [build_prompt(m["schema"]) for m in schemas]
    stop = ["\n\n", "```"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def part_path(theta):
        return f"{args.out}.theta{theta}.part.json"

    todo = [t for t in thetas if not os.path.exists(part_path(t))]
    llm = dtype = None
    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
                  gpu_memory_utilization=args.gpu_mem_frac, seed=0,
                  max_model_len=args.max_model_len)
        dtype = str(llm.llm_engine.model_config.dtype)

    for theta in thetas:
        if os.path.exists(part_path(theta)):
            print(f"  theta={theta}: checkpoint exists, skipping", flush=True)
            continue
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new,
                            repetition_penalty=theta, stop=stop)
        outs = llm.generate(prompts, sp)
        recs, nvalid = [], 0
        for m, o in zip(schemas, outs):
            text = o.outputs[0].text
            obj, sub = first_json(text)
            ok, reason = schema_valid(obj, m["schema"])
            nvalid += ok
            recs.append({"id": m["id"], "valid": bool(ok), "reason": reason,
                         "extracted": sub, "output": text[:2000]})
        json.dump({"theta": theta, "dtype": dtype, "n": len(recs), "n_valid": nvalid,
                   "valid_rate": nvalid / len(recs), "records": recs},
                  open(part_path(theta), "w"), indent=1)
        print(f"  theta={theta}: schema-valid {nvalid}/{len(recs)} = {nvalid/len(recs):.3f}", flush=True)

    # merge
    runs, summary = {}, {}
    dtypes = set()
    for theta in thetas:
        d = json.load(open(part_path(theta)))
        key = f"raw_theta{theta}"
        runs[key] = d["records"]
        summary[key] = {"valid_rate": d["valid_rate"], "n": d["n"], "n_valid": d["n_valid"]}
        if d.get("dtype"):
            dtypes.add(d["dtype"])
    json.dump({"model": args.model, "dtype": sorted(dtypes), "engine": "V1 (default)",
               "knob": "stock SamplingParams.repetition_penalty",
               "seen_set": "vLLM repetition_penalty penalizes prompt+output tokens",
               "manifest": args.manifest, "thetas": thetas, "n_schemas": len(schemas),
               "max_new": args.max_new, "prompt_template": "conforms-to-schema",
               "summary": summary, "runs": runs}, open(args.out, "w"), indent=1)
    print("summary:", json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
