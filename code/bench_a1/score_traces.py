#!/usr/bin/env python3
"""Score generated traces under the base model's unpenalized distribution (teacher-forced).
For each condition, per-position log p of the generated token given prefix + preceding tokens."""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai-community/gpt2-large")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--batch-size", type=int, default=20)
ap.add_argument("--out", required=True)
ap.add_argument("--natural", action="store_true")
args = ap.parse_args()

R = "results/bench_a1"
prefixes = [p["prefix_text"] for p in json.load(open(f"{R}/prefixes.json"))["prefixes"]]

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).cuda().eval()

def recs_of(path, cond_key):
    d = json.load(open(path))
    out = {}
    for r in d["records"]:
        out[(float(r[cond_key]), float(r["c"]), int(r["prompt_idx"]))] = r["gen_ids"]
    return out

import os.path
short = args.model.split("/")[-1]
if getattr(args, "natural", False):
    nat = recs_of(f"{R}/raw_flip_{short}_natural.json", "theta")
    cs = sorted({c for (t, c, i) in nat}, reverse=True)  # [0.0, -13.34]
    conds = {
      "ctrl_shipped": [nat[(1.3, cs[0], i)] for i in range(200)],
      "ctrl_median":  [nat[(1.3, cs[1], i)] for i in range(200)],
    }
else:
    ctrl = recs_of(f"{R}/raw_flip_{short}.json", "theta")
    sub  = recs_of(f"{R}/raw_flip_{short}_subtractive.json", "alpha")
    conds = {
      "unpenalized": [ctrl[(1.0, 5.0, i)] for i in range(200)],
      "ctrl_p5":     [ctrl[(1.3, 5.0, i)] for i in range(200)],
      "ctrl_m5":     [ctrl[(1.3, -5.0, i)] for i in range(200)],
      "subtractive": [sub[(1.0, 5.0, i)] for i in range(200)],
    }
    fixpath = f"{R}/raw_flip_{short}_fix.json"
    if os.path.exists(fixpath):
        fix = recs_of(fixpath, "theta")
        conds["normalized"] = [fix[(1.3, 5.0, i)] for i in range(200)]

@torch.no_grad()
def score(gen_lists):
    out = []
    for b0 in range(0, 200, args.batch_size):
        batch_idx = range(b0, min(b0+args.batch_size, 200))
        seqs, plens = [], []
        for i in batch_idx:
            pid = tok(prefixes[i], return_tensors="pt").input_ids[0]
            gid = torch.tensor(gen_lists[i], dtype=torch.long)
            seqs.append(torch.cat([pid, gid])); plens.append(len(pid))
        maxlen = max(len(s) for s in seqs)
        pad = tok.eos_token_id or 0
        inp = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for j, s in enumerate(seqs):
            inp[j, :len(s)] = s; att[j, :len(s)] = 1
        inp, att = inp.cuda(), att.cuda()
        logits = model(input_ids=inp, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        for j, s in enumerate(seqs):
            pl, n = plens[j], len(s)
            tokenlp = logp[j, pl-1:n-1, :].gather(1, s[pl:n].unsqueeze(1).cuda()).squeeze(1)
            out.append(tokenlp.cpu().tolist())
    return out

res = {name: score(gl) for name, gl in conds.items()}
json.dump(res, open(args.out, "w"))
print("wrote", args.out, {k: len(v) for k, v in res.items()})
