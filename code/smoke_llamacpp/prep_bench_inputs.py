#!/usr/bin/env python3
"""Materialize the benchmark manifests as plain-text inputs for the C++ drivers.

- A1: results/bench_a1/prefixes.json -> bench_inputs/a1_prefixes.txt (one prefix per
  line, manifest order; prefixes are single-line WikiText segments, asserted here).
- A2: results/bench_a2/schemas.json  -> bench_inputs/a2_prompts/prompt_NNN.txt, one
  file per schema (prompts contain newlines). Prompt construction is copied VERBATIM
  from code/bench_a2/run_json_bench.py:build_prompt so cross-stack rates are
  comparable to the HF-side benchmark numbers.
"""
import json, os, sys

ROOT = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
OUT = os.path.join(ROOT, "code/smoke_llamacpp/bench_inputs")


# ---- VERBATIM from code/bench_a2/run_json_bench.py ----
def build_prompt(schema):
    s = json.dumps(schema, indent=2)
    return ("Output ONLY a single JSON object that conforms to this JSON Schema. "
            "Do not include any explanation.\n"
            f"JSON Schema:\n{s}\nJSON object:\n")
# ---- end verbatim ----


def main():
    os.makedirs(os.path.join(OUT, "a2_prompts"), exist_ok=True)

    a1 = json.load(open(os.path.join(ROOT, "results/bench_a1/prefixes.json")))
    prefixes = [p["prefix_text"] for p in a1["prefixes"]]
    assert len(prefixes) == 200
    for i, p in enumerate(prefixes):
        assert "\n" not in p and "\r" not in p, f"prefix {i} contains newline"
    with open(os.path.join(OUT, "a1_prefixes.txt"), "w") as f:
        f.write("\n".join(prefixes) + "\n")
    print(f"wrote {len(prefixes)} A1 prefixes")

    a2 = json.load(open(os.path.join(ROOT, "results/bench_a2/schemas.json")))
    schemas = a2["schemas"]
    assert len(schemas) == 200
    for i, m in enumerate(schemas):
        with open(os.path.join(OUT, "a2_prompts", f"prompt_{i:03d}.txt"), "w") as f:
            f.write(build_prompt(m["schema"]))
    print(f"wrote {len(schemas)} A2 prompt files")


if __name__ == "__main__":
    main()
