#!/usr/bin/env python3
"""Build results/smoke_{llamacpp,vllm}_bench/summary.json from the per-leg outputs.

Run after all four legs finish:
  python3 summarize_bench.py
Reads:
  results/smoke_llamacpp_bench/a1_summary.json, a2_summary.json  (from analyze_* scripts)
  results/smoke_vllm_bench/a1_raw.json, a2_raw.json
Writes:
  results/smoke_llamacpp_bench/summary.json, results/smoke_vllm_bench/summary.json
"""
import json, os

R = "results"
HF_REF = {
    "a1_gpt2_large_flip": {"1.0": 0.0, "1.3": 0.9639},
    "a2_json_valid_raw": {"1.0": 0.970, "1.1": 0.955, "1.3": 0.225},
    "a2_json_valid_fix": {"1.0": 0.970, "1.1": 0.975, "1.3": 0.970},
}
OLD_SMOKE = {   # previous cross-stack round (16 hand-written prompts / 6 schemas)
    "llamacpp": {"a1_flip_1.3": 0.9413, "a2_valid": {"1.0": 1.0, "1.1": 1.0, "1.3": 0.167}},
    "vllm":     {"a1_flip_1.3": 0.9387, "a2_valid": {"1.0": 1.0, "1.1": 1.0, "1.3": 0.167}},
}


def llamacpp():
    a1 = json.load(open(f"{R}/smoke_llamacpp_bench/a1_summary.json"))
    a2 = json.load(open(f"{R}/smoke_llamacpp_bench/a2_summary.json"))
    # a1_driver emits theta with %.4g, so 1.0 round-trips as int 1 -> key "1"
    if "1" in a1["flip_rate_by_theta"] and "1.0" not in a1["flip_rate_by_theta"]:
        a1["flip_rate_by_theta"]["1.0"] = a1["flip_rate_by_theta"].pop("1")
    s = {
        "stack": "llama.cpp (commit 4fc4ec5541b243957ae5099edb67372f8f3b550e, own sampler chain)",
        "manifests": {"a1": f"{R}/bench_a1/prefixes.json", "a2": f"{R}/bench_a2/schemas.json"},
        "a1": {
            "model": "gpt2-large f16 GGUF, CPU backend, penalty_last_n=1024, 200 prefixes x 200 tokens",
            "validity_gate_theta1.0_flip": a1["flip_rate_by_theta"]["1.0"],
            "validity_gate_pass": a1["validity_gate_pass"],
            "flip_rate_by_theta": a1["flip_rate_by_theta"],
        },
        "a2": {
            "model": "Qwen2.5-Coder-7B bf16 GGUF, CUDA -ngl 26 (see REPORT caveat), stock repeat_penalty, 200 schemas, max_new=512",
            "valid_rate_by_lastn_theta": a2["valid_rate"],
            "condition_label": a2["condition_label"],
        },
        "hf_benchmark_reference": HF_REF,
        "previous_smoke_round": OLD_SMOKE["llamacpp"],
    }
    json.dump(s, open(f"{R}/smoke_llamacpp_bench/summary.json", "w"), indent=2)
    print("wrote smoke_llamacpp_bench/summary.json")


def vllm():
    a1 = json.load(open(f"{R}/smoke_vllm_bench/a1_raw.json"))
    a2 = json.load(open(f"{R}/smoke_vllm_bench/a2_raw.json"))
    a1sum = {th: {k: v for k, v in d.items() if k != "per_prompt"}
             for th, d in a1["summary"].items()}
    gate = a1sum.get("1.0", {}).get("flip_rate", None)
    s = {
        "stack": "vLLM 0.8.5.post1 (A1: V0 + pre-penalty logits_processor shift; A2: stock repetition_penalty on V1)",
        "manifests": {"a1": f"{R}/bench_a1/prefixes.json", "a2": f"{R}/bench_a2/schemas.json"},
        "a1": {
            "model": f"{a1['model']} dtype={a1['dtype']}, {a1['n_prompts']} prefixes x {a1['max_tokens']} tokens, ignore_eos",
            "validity_gate_theta1.0_flip": gate,
            "validity_gate_pass": gate == 0.0,
            "flip_rate_by_theta": a1sum,
        },
        "a2": {
            "model": f"{a2['model']} dtype={a2['dtype']}, {a2['n_schemas']} schemas, max_new={a2['max_new']}",
            "valid_rate_by_theta": a2["summary"],
        },
        "hf_benchmark_reference": HF_REF,
        "previous_smoke_round": OLD_SMOKE["vllm"],
    }
    json.dump(s, open(f"{R}/smoke_vllm_bench/summary.json", "w"), indent=2)
    print("wrote smoke_vllm_bench/summary.json")


if __name__ == "__main__":
    for fn in (llamacpp, vllm):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skipped {fn.__name__}: missing {e.filename}")
