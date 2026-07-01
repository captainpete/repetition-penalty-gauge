#!/usr/bin/env python3
"""Shared semantics for the benchmark-sourced A1 re-run.

Penalty semantics are copied verbatim (in intent) from code/run_a1.py + run_a1_zeropoint.py:
  1. gauge shift c is ADDED to the raw logits BEFORE the penalty (== lm_head.bias += c),
  2. everything after the model forward is done in fp32 (logits.float()), even for bf16 models,
  3. optional FIX operator: log_softmax(logits) BEFORE the penalty (normalize-before-penalize),
  4. HF RepetitionPenaltyLogitsProcessor sign-branch over ALL previously-seen ids
     (prompt ids + generated ids): v<0 -> v*theta, v>=0 -> v/theta; no-op at theta==1,
  4b. optional SUBTRACTIVE operator (presence-style penalty): for every previously-seen id,
     z_i -> z_i - alpha (exact standard presence_penalty semantics), applied at the same point
     as the sign-branch penalty and AFTER the gauge shift c. This is a gauge-invariant control:
     a scalar shift c added to every logit before subtracting the same alpha from the same seen
     set can never change the argmax, so the c=+5 and c=-5 runs are token-identical (flip == 0),
     exactly like the theta==1 gate. When subtractive is set it REPLACES the sign-branch penalty.
  5. greedy = argmax(logits) after shift+(fix)+penalty.

This module batches generation across prefixes for throughput. Batching cannot introduce a
spurious flip at theta=1.0: adding a constant c to every logit never changes the argmax, so the
c=+5 and c=-5 runs produce token-identical streams at theta=1.0 regardless of batch size (the
validity gate is exact by construction). At theta>1 the per-row penalty makes them diverge.
Left-padding + explicit attention_mask/position_ids keep padded rows from affecting real rows.
"""
import torch


@torch.no_grad()
def batched_greedy(model, prompt_ids_list, c, theta, max_new, device,
                   fix=False, batch_size=32, collect_zp=False, subtractive=None):
    """Greedy decode with gauge shift c + HF rep-penalty, batched across prefixes.

    Returns (gens, zp) where gens is a list (len == len(prompt_ids_list)) of token-id lists,
    and zp is None unless collect_zp, in which case zp = dict with running zero-point stats:
      {'zp_pos': int, 'zp_tot': int, 'ztops': [float, ...]}  (collected at the given c/theta;
      caller uses c=0, theta=1 to match run_a1_zeropoint.py).
    """
    n = len(prompt_ids_list)
    gens = [None] * n
    zp_pos, zp_tot, ztops = 0, 0, []

    for start in range(0, n, batch_size):
        chunk = prompt_ids_list[start:start + batch_size]
        B = len(chunk)
        maxlen = max(len(p) for p in chunk)
        input_ids = torch.zeros((B, maxlen), dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for i, p in enumerate(chunk):
            input_ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, maxlen - len(p):] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)

        out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        V = logits.shape[-1]

        seen = torch.zeros((B, V), dtype=torch.bool, device=device)
        for i, p in enumerate(chunk):
            seen[i, torch.tensor(p, device=device)] = True

        gen = [[] for _ in range(B)]
        cur_attn = attn
        for step in range(max_new):
            if collect_zp:
                ztops.extend(logits.max(dim=-1).values.tolist())
                pos_seen = (logits > 0) & seen
                zp_pos += int(pos_seen.sum())
                zp_tot += int(seen.sum())
            lg = logits + c
            if fix:
                lg = torch.log_softmax(lg, dim=-1)
            if subtractive is not None:
                # presence-style: z_i -> z_i - alpha for every previously-seen id (same seen-set
                # as the sign-branch penalty). Gauge-invariant, so a control (see module docstring).
                lg = torch.where(seen, lg - subtractive, lg)
            elif theta != 1.0:
                pen = torch.where(lg < 0, lg * theta, lg / theta)
                lg = torch.where(seen, pen, lg)
            nxt = lg.argmax(dim=-1)  # [B]
            row = torch.arange(B, device=device)
            seen[row, nxt] = True
            for i in range(B):
                gen[i].append(int(nxt[i]))
            if step == max_new - 1:
                break
            cur_attn = torch.cat([cur_attn, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
            pos_next = cur_attn.sum(-1, keepdim=True) - 1
            out = model(input_ids=nxt.unsqueeze(1), attention_mask=cur_attn,
                        position_ids=pos_next, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

        for i in range(B):
            gens[start + i] = gen[i]

    zp = {"zp_pos": zp_pos, "zp_tot": zp_tot, "ztops": ztops} if collect_zp else None
    return gens, zp


def load_prefix_ids(tokenizer, prefixes_json):
    """Tokenize the manifest prefix TEXTs with the given model tokenizer."""
    import json
    d = json.load(open(prefixes_json))
    texts = [p["prefix_text"] for p in d["prefixes"]]
    # Match run_a1.py exactly: tok(text)["input_ids"] with tokenizer defaults.
    ids = [tokenizer(t)["input_ids"] for t in texts]
    return texts, ids
