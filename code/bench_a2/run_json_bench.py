#!/usr/bin/env python3
"""Part A — JSON schema-valid rate on JSONSchemaBench schemas, under the raw vs fix
repetition penalty. Mirrors code/run_a2_downstream.py (RepPen logits-processor with exact
HF semantics + --fix log_softmax-before-penalize; greedy; batched left-pad decode;
first_json brace-matching extraction) but embeds a *benchmark* schema in the prompt and
validates the emitted object against that *actual* schema with jsonschema.

  .venv/bin/python run_json_bench.py
"""
import os, json, argparse
os.environ.setdefault("HF_HOME", "$HF_HUB_CACHE")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


class RepPen(LogitsProcessor):
    """exact A1/A2 HF rep-penalty semantics during generation; --fix = log_softmax before penalizing."""
    def __init__(self, theta, fix):
        self.theta, self.fix = theta, fix

    def __call__(self, input_ids, scores):
        if self.fix:
            scores = torch.log_softmax(scores, dim=-1)
        if self.theta != 1.0:
            for b in range(scores.size(0)):
                seen = torch.unique(input_ids[b])
                s = scores[b, seen]
                scores[b, seen] = torch.where(s < 0, s * self.theta, s / self.theta)
        return scores


def gen_batch(model, tok, prompts, theta, fix, device, max_new, stop):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                         logits_processor=[RepPen(theta, fix)], stop_strings=stop, tokenizer=tok,
                         pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--manifest", default="results/bench_a2/schemas.json")
    ap.add_argument("--thetas", default="1.0,1.1,1.3")
    ap.add_argument("--ops", default="raw,fix")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="first N schemas (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/bench_a2/json_raw.json")
    args = ap.parse_args()
    thetas = [float(x) for x in args.thetas.split(",")]
    ops = [(o, o == "fix") for o in args.ops.split(",")]

    man = json.load(open(args.manifest))
    schemas = man["schemas"]
    if args.limit:
        schemas = schemas[:args.limit]
    prompts = [build_prompt(m["schema"]) for m in schemas]
    stop = ["\n\n", "```"]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()

    runs = {}       # "op_thetaX" -> list of per-schema records
    summary = {}
    for theta in thetas:
        for op, fix in ops:
            outs = []
            for i in range(0, len(prompts), args.bs):
                outs += gen_batch(model, tok, prompts[i:i + args.bs], theta, fix, args.device, args.max_new, stop)
            recs = []
            nvalid = 0
            for m, o in zip(schemas, outs):
                obj, sub = first_json(o)
                ok, reason = schema_valid(obj, m["schema"])
                nvalid += ok
                recs.append({"id": m["id"], "valid": bool(ok), "reason": reason,
                             "extracted": sub, "output": o[:2000]})
            key = f"{op}_theta{theta}"
            runs[key] = recs
            summary[key] = {"valid_rate": nvalid / len(recs), "n": len(recs), "n_valid": nvalid}
            print(f"  theta={theta} {op}: schema-valid {nvalid}/{len(recs)} = {nvalid/len(recs):.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "thetas": thetas, "ops": [o for o, _ in ops],
               "n_schemas": len(schemas), "max_new": args.max_new, "bs": args.bs,
               "prompt_template": "conforms-to-schema", "summary": summary, "runs": runs},
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
