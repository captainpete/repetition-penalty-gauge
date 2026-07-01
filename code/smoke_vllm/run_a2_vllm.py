#!/usr/bin/env python3
"""A2 JSON-validity probe, measured INSIDE vLLM using the STOCK repetition_penalty knob.

Reuses ../run_a2_downstream.py's JSON_TASKS (6 schemas), prompt construction, and
first_json + json_valid validators VERBATIM so numbers are comparable. No custom
logits processors here -- this exercises vLLM's native SamplingParams.repetition_penalty
exactly as a user would hit it (default engine, whatever V vLLM picks). Greedy decode,
Qwen2.5-Coder-7B bf16, theta in {1.0, 1.1, 1.3}. Reports schema-valid rate per theta.
HF reference (../run_a2_downstream.py): 1.0 / 1.0 / 0.0.

  .venv/bin/python run_a2_vllm.py
"""
import os, json, argparse
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams

# ---- VERBATIM from ../run_a2_downstream.py ----
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--thetas", default="1.0,1.1,1.3")
    ap.add_argument("--json-reps", type=int, default=8)  # matches original default
    ap.add_argument("--max-tokens", type=int, default=160)  # matches original JSON gen length
    ap.add_argument("--out", default="../../results/smoke_vllm/a2_raw.json")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.90)
    args = ap.parse_args()
    thetas = [float(x) for x in args.thetas.split(",")]

    # Prompt construction VERBATIM from original.
    json_prompts, json_schemas = [], []
    for desc, sch in JSON_TASKS:
        for _ in range(args.json_reps):
            json_prompts.append(f"Output ONLY a single JSON object describing {desc}. JSON:\n")
            json_schemas.append(sch)

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_frac, seed=0)
    mc = llm.llm_engine.model_config
    dtype = str(mc.dtype)

    res, raw = {}, {}
    for theta in thetas:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            repetition_penalty=theta, stop=["\n\n", "```"])
        outs = llm.generate(json_prompts, sp)
        texts = [o.outputs[0].text for o in outs]
        valids = [json_valid(first_json(t), json_schemas[k]) for k, t in enumerate(texts)]
        rate = sum(valids) / len(valids)
        res[f"theta{theta}"] = {"json_valid_rate": rate, "n": len(valids)}
        raw[f"theta{theta}"] = [{"prompt": json_prompts[k], "text": texts[k],
                                 "valid": bool(valids[k])} for k in range(len(texts))]
        print(f"  theta={theta}: JSON valid {rate:.3f} ({sum(valids)}/{len(valids)})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"model": args.model, "dtype": dtype, "thetas": thetas,
               "n_json": len(json_prompts), "max_tokens": args.max_tokens,
               "seen_set": "vLLM repetition_penalty penalizes prompt+output tokens",
               "metrics": res, "raw": raw}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
