#!/usr/bin/env python3
"""Build the WikiText-103 prefix manifest for the benchmark-sourced A1 re-run.

Protocol (SimCTG / contrastive-search open-ended generation):
  - Load WikiText-103 test split ('Salesforce/wikitext', 'wikitext-103-raw-v1', test).
  - Keep only prose segments: drop empty lines and headings (" = Title = " style), and
    require >= 40 GPT-2 BPE tokens BEFORE truncation (so a 32-token prefix is a real cut).
  - With seed 0, sample 200 eligible segments.
  - Prefix = first 32 GPT-2 BPE tokens, decoded back to TEXT. Prefixes are defined as TEXT
    (not token ids) so every model shares the same prefixes despite different tokenizers.

Writes results/bench_a1/prefixes.json — WRITE ONCE, never regenerate with different contents
(other agents reuse this exact file).

CPU only, no GPU. Run:
  python make_prefixes.py --out ../../results/bench_a1/prefixes.json
"""
import os, re, json, argparse, random


N_PREFIXES = 200
PREFIX_TOKENS = 32
MIN_TOKENS = 40
SEED = 0
HEADING_RE = re.compile(r"^\s*=+\s.*\s=+\s*$")  # " = Title = ", " = = Sub = = ", ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench_a1/prefixes.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")

    eligible = []  # (row_id, text, ids[:32])
    for i, row in enumerate(ds):
        text = row["text"]
        if not text or not text.strip():
            continue
        if HEADING_RE.match(text):
            continue
        ids = tok(text)["input_ids"]
        if len(ids) < MIN_TOKENS:
            continue
        eligible.append((i, text, ids[:PREFIX_TOKENS]))

    rng = random.Random(SEED)
    chosen = rng.sample(eligible, N_PREFIXES)

    manifest = {
        "protocol": ("200 32-token prefixes sampled from the WikiText-103 test set, seed 0. "
                     "Prefixes are the first 32 GPT-2 BPE tokens of each sampled prose segment, "
                     "decoded back to TEXT so all models share the same prefixes. Filter: drop "
                     "empty lines and headings; require >= 40 GPT-2 tokens before truncation."),
        "dataset": "Salesforce/wikitext / wikitext-103-raw-v1 / test",
        "n_prefixes": N_PREFIXES,
        "prefix_tokens": PREFIX_TOKENS,
        "min_tokens_before_truncation": MIN_TOKENS,
        "seed": SEED,
        "tokenizer": "gpt2",
        "n_eligible": len(eligible),
        "prefixes": [
            {"index": k, "source_row_id": rid, "prefix_text": tok.decode(ids)}
            for k, (rid, _text, ids) in enumerate(chosen)
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"eligible segments: {len(eligible)}")
    print(f"wrote {N_PREFIXES} prefixes -> {args.out}")
    # sanity: token length under gpt2 after decode (should be ~32)
    lens = [len(tok(p["prefix_text"])["input_ids"]) for p in manifest["prefixes"][:5]]
    print(f"first-5 re-tokenized gpt2 lengths: {lens}")


if __name__ == "__main__":
    main()
