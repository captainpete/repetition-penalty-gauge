#!/usr/bin/env python3
"""A2 (rep-penalty) — does the per-token corruption matter END-TO-END? Measures real
task quality on COMPLETE generations under raw vs --fix (normalize-before-penalize) penalty, at
theta in {1.0,1.1,1.3}: (1) HumanEval pass@1 (greedy, code EXECUTED in a sandboxed subprocess), (2)
JSON-schema conformance (emit a complete JSON object, parse + validate). theta=1.0 is the control.
A measurable raw degradation recovered by the fix converts the structured-output implication to evidence.

  python code/run_a2_downstream.py
"""
import os, re, sys, json, argparse, subprocess, tempfile
os.environ.setdefault("HF_HUB_CACHE", "/hf/hub")
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


def exec_ok(prog, timeout=8):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=timeout).returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


# ---- JSON conformance task: (instruction, required {field: type-check}) ----
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--thetas", default="1.0,1.1,1.3")
    ap.add_argument("--n-humaneval", type=int, default=164)
    ap.add_argument("--json-reps", type=int, default=8)   # prompt variants per schema
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/A2_downstream/raw.json")
    args = ap.parse_args()
    thetas = [float(x) for x in args.thetas.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()

    from datasets import load_dataset
    he = load_dataset("openai/openai_humaneval", split="test").select(range(min(args.n_humaneval, 164)))
    he_prompts = [r["prompt"] for r in he]
    he_stop = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@", "\n```"]
    json_prompts, json_schemas = [], []
    for desc, sch in JSON_TASKS:
        for _ in range(args.json_reps):
            json_prompts.append(f"Output ONLY a single JSON object describing {desc}. JSON:\n")
            json_schemas.append(sch)

    res = {}
    for theta in thetas:
        for op, fix in [("raw", False), ("fix", True)]:
            # HumanEval pass@1
            comps = []
            for i in range(0, len(he_prompts), args.bs):
                comps += gen_batch(model, tok, he_prompts[i:i + args.bs], theta, fix, args.device, 512, he_stop)
            passed = 0
            for r, c in zip(he, comps):
                prog = r["prompt"] + c + "\n" + r["test"] + f"\ncheck({r['entry_point']})\n"
                passed += exec_ok(prog)
            pass1 = passed / len(he)
            # JSON conformance
            jouts = []
            for i in range(0, len(json_prompts), args.bs):
                jouts += gen_batch(model, tok, json_prompts[i:i + args.bs], theta, fix, args.device, 160, ["\n\n", "```"])
            valid = sum(json_valid(first_json(o), json_schemas[k]) for k, o in enumerate(jouts)) / len(jouts)
            res[f"{op}_theta{theta}"] = {"humaneval_pass1": pass1, "json_valid_rate": valid}
            print(f"  theta={theta} {op}: HumanEval pass@1 {pass1:.3f} | JSON valid {valid:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "thetas": thetas, "n_humaneval": len(he),
               "n_json": len(json_prompts), "metrics": res}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
