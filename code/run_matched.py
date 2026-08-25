#!/usr/bin/env python3
"""MATCHED (MATCHED) — matched-SUPPRESSION head-to-head: quality at equal anti-repetition strength.

Frozen protocol in PREREG.md.  Every existing raw-vs-fix comparison is at matched THETA, which is
confounded (the normalized operator does less work per unit theta).  This runs the fair comparison:
raw(theta_raw) vs fix(theta'_fix) at the FIXCAL pairs that produce EQUAL measured repetition
suppression, and asks whether the fix costs quality.

Harnesses reused UNMODIFIED (imported, never edited):
  code/run_a1.py            -- PROMPTS, generate() (gauge shift c + HF penalty semantics), distinct_n
  code/run_a1_loopcheck.py  -- greedy() + degen() (rep_rate / distinct2 / longest_run)
  code/run_a2_downstream.py -- RepPen LogitsProcessor, gen_batch, JSON_TASKS, first_json, json_valid
  code/judge_lib.py            -- Qwen2.5-7B-Instruct judge (fallback quality measure)

Stages (each writes its own raw, so a later failure never loses earlier work):
  pairs      Stage 0a -- verify the frozen pairs + RE-MEASURE rep_rate(raw) vs rep_rate(fix)  [premise gate]
  gauge      Stage 0b -- A1 c=+-5 flip-rate under raw and under fix at every pair             [hard gate]
  open       Stage 1  -- arm (b) open-ended quality: distinct-1/2 + MAUVE|judge|NLL
  json       Stage 2  -- arm (a) part 1: the a2_downstream run JSON-schema validity                           [leads arm a]
  humaneval  Stage 3  -- arm (a) part 2: canonical HumanEval pass@1 (generation; --exec scores)

REAL RUN (GPU, in the lab container as user lab; models cached under /hf/hub) -- see PREREG.md §7.
CPU SMOKE:
  python run_matched.py --device cpu --model gpt2 --stage pairs --limit 2 --max-new 24 --out /tmp/M/raw_gpt2_pairs.json
"""
import os, sys, json, time, random, shutil, signal, resource, argparse, hashlib, tempfile, subprocess

os.environ.setdefault("HF_HUB_CACHE", "/hf/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/hf/datasets")

HERE = os.path.dirname(os.path.abspath(__file__))
EXPDIR = os.path.abspath(os.path.join(HERE, os.pardir))
for _p in (os.path.join(EXPDIR, "A1"), os.path.join(EXPDIR, "A2"), EXPDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------- frozen inputs (PREREG §3)
# FIXCAL status block: the MATCHED-pair allowlist MATCHED may use.  theta' is READ from the FIXCAL
# summary at run time; this table is the ASSERTION it must satisfy (2-dp rounded, tol 0.006).
FROZEN_THETA_FIX = {
    "gpt2":             {1.02: 2.86},
    "pythia-2.8b":      {1.02: 1.21, 1.05: 1.70, 1.1: 2.51, 1.15: 3.81},
    "Qwen2.5-7B":       {1.02: 1.28, 1.05: 1.84, 1.1: 5.50},
    "Qwen2.5-Coder-7B": {1.02: 1.28, 1.05: 3.25, 1.1: 6.63},
}
ALLOWLIST = {k: sorted(v) for k, v in FROZEN_THETA_FIX.items()}
# 2-3 levels per model per the request; the deployed anchor 1.10 wherever it exists.
# gpt2 has ONLY 1.02 (1.10 is UNREACHABLE at theta'<=10) -- documented deviation, PREREG §3.
DEFAULT_LEVELS = {
    "gpt2":             [1.02],
    "pythia-2.8b":      [1.02, 1.05, 1.1],
    "Qwen2.5-7B":       [1.02, 1.05, 1.1],
    "Qwen2.5-Coder-7B": [1.02, 1.05, 1.1],
}
DEPLOYED_ANCHOR = 1.1
THETA_FIX_TOL = 0.006          # frozen table is 2-dp rounded
REP_RATE_TOL = 0.01            # FIXCAL matching tolerance (FIXCAL TOL["rep_rate"])
VENDORED_PAIRS = os.path.join(HERE, "fix_calibration_summary.json")
PAPER_PAIRS = "results/fix_calibration/summary.json"

# canonical HumanEval stop sequences (official human-eval / Codex protocol)
HE_STOP = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]

JUDGE_RUBRIC = (
    "You are grading the fluency and coherence of a text continuation produced by a language model.\n"
    "Rate ONLY the continuation, on this scale:\n"
    "5 = fluent, grammatical, on-topic and internally consistent throughout\n"
    "4 = mostly fluent and coherent, minor lapses\n"
    "3 = understandable but noticeably disfluent, repetitive, or drifting\n"
    "2 = largely incoherent, heavy repetition or broken syntax\n"
    "1 = degenerate: near-total repetition, gibberish, or no meaningful content\n\n"
    "Prompt:\n{prompt}\n\nContinuation:\n{text}\n\n"
    "Answer with a single integer from 1 to 5 and nothing else."
)


def slug_of(model):
    return model.rsplit("/", 1)[-1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_ids(ids):
    return hashlib.sha1(",".join(str(i) for i in ids).encode()).hexdigest()[:16]


def die(msg):
    print(f"ABORT: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


# ---------------------------------------------------------------------------- pairs (PREREG §3)
def resolve_pairs_json(arg):
    if arg:
        return arg
    if os.path.exists(PAPER_PAIRS):
        return PAPER_PAIRS
    return VENDORED_PAIRS


def load_pairs(pairs_json, slug, levels):
    """Read theta' from the FIXCAL summary at run time and ASSERT against the frozen table."""
    if slug not in FROZEN_THETA_FIX:
        die(f"model slug {slug!r} is not in the FIXCAL allowlist {sorted(FROZEN_THETA_FIX)}")
    if not os.path.exists(pairs_json):
        die(f"pairs json not found: {pairs_json}")
    d = json.load(open(pairs_json))
    if slug not in d.get("models", {}):
        die(f"{slug!r} absent from {pairs_json}")
    md = d["models"][slug]
    by_anchor = {}
    for e in md["migration"]["rep_rate"]:
        by_anchor[round(float(e["theta_raw"]), 4)] = e

    bad = [t for t in levels if round(t, 4) not in ALLOWLIST[slug]]
    if bad:
        die(f"--levels {bad} outside the FROZEN FIXCAL allowlist for {slug}: {ALLOWLIST[slug]} "
            f"(no substitution of nearby pairs -- MATCHED pre-registration 1)")

    pairs = []
    for t in levels:
        k = round(float(t), 4)
        e = by_anchor.get(k)
        if e is None:
            die(f"anchor theta_raw={k} absent from the rep_rate migration for {slug}")
        if e.get("verdict") != "MATCHED" or e.get("theta_fix") is None:
            die(f"anchor theta_raw={k} on {slug} is {e.get('verdict')} in FIXCAL -- not usable")
        if e.get("metric") != "rep_rate":
            die(f"anchor theta_raw={k} on {slug} is not the rep_rate map")
        exp = FROZEN_THETA_FIX[slug][k]
        got = float(e["theta_fix"])
        if abs(got - exp) > THETA_FIX_TOL:
            die(f"FROZEN PAIR MISMATCH {slug} theta_raw={k}: summary.json theta'={got!r} "
                f"but the frozen FIXCAL table says {exp!r} (tol {THETA_FIX_TOL})")
        pairs.append({
            "theta_raw": k, "theta_fix": got, "frozen_theta_fix_2dp": exp,
            "fixcal_target_rep_rate": e.get("target"), "fixcal_achieved_rep_rate": e.get("achieved"),
            "fixcal_delta": e.get("delta"), "is_deployed_anchor": (k == DEPLOYED_ANCHOR),
        })
    meta = {
        "pairs_json": pairs_json, "pairs_json_sha256": sha256_file(pairs_json),
        "fixcal_model": md.get("model"), "fixcal_revision": md.get("revision"),
        "fixcal_dtype": md.get("dtype"), "fixcal_max_new": md.get("max_new"),
        "fixcal_n_prompts": md.get("n_prompts"), "allowlist": ALLOWLIST[slug],
        "levels_used": [p["theta_raw"] for p in pairs],
        "deployed_anchor_available": DEPLOYED_ANCHOR in ALLOWLIST[slug],
    }
    return pairs, meta


# ---------------------------------------------------------------------------- model io
def load_lm(name, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device).eval()
    rev = getattr(m.config, "_commit_hash", None) or "unknown"
    return m, tok, rev


def pick_dtype(arg, device, fixcal_dtype):
    if arg != "auto":
        return arg
    if device == "cpu":
        return "float32"
    return fixcal_dtype if fixcal_dtype in ("float32", "bfloat16", "float16") else "float32"


def base_meta(args, slug, pairs, pmeta, rev, dtype, stage):
    return {
        "experiment": "MATCHED", "request": "MATCHED", "stage": stage,
        "model": args.model, "model_slug": slug, "revision": rev,
        "device": args.device, "dtype": dtype, "max_new": args.max_new,
        "seed": args.seed, "limit": args.limit, "pairs": pairs, "pairs_meta": pmeta,
        "rep_rate_tol": REP_RATE_TOL, "deployed_anchor": DEPLOYED_ANCHOR,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_out(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)


def conditions_for(pairs):
    """Frozen condition list: the no-op control first, then raw/fix at each pair."""
    conds = [("baseline_raw", 1.0, False, None), ("baseline_fix", 1.0, True, None)]
    for i, p in enumerate(pairs):
        conds.append((f"raw@{p['theta_raw']:g}", p["theta_raw"], False, i))
        conds.append((f"fix@{p['theta_raw']:g}", p["theta_fix"], True, i))
    return conds


# ---------------------------------------------------------------------------- stage: pairs / open
def run_openlike(args, stage):
    import torch
    from run_a1 import PROMPTS, distinct_n
    from run_a1_loopcheck import greedy, degen

    slug = slug_of(args.model)
    levels = [float(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS[slug]
    pairs, pmeta = load_pairs(resolve_pairs_json(args.pairs_json), slug, levels)
    dtype = pick_dtype(args.dtype, args.device, pmeta.get("fixcal_dtype"))
    model, tok, rev = load_lm(args.model, args.device, dtype)

    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS
    pids = [tok(p)["input_ids"] for p in prompts]

    out_conds = []
    for label, theta, fix, pi in conditions_for(pairs):
        t0 = time.time()
        per = []
        for j, ids in enumerate(pids):
            with torch.no_grad():
                g = greedy(model, ids, theta, args.max_new, args.device, fix)
            rr, d2, lr = degen(ids, g)
            rec = {"prompt_idx": j, "rep_rate": rr, "distinct1": distinct_n(g, 1),
                   "distinct2": d2, "longest_run": lr, "gen_sha": sha1_ids(g)}
            if stage == "open":
                rec["text"] = tok.decode(g)
            per.append(rec)
        out_conds.append({"label": label, "op": ("fix" if fix else "raw"), "theta": theta,
                          "pair_idx": pi, "per_prompt": per})
        m = sum(r["rep_rate"] for r in per) / len(per)
        print(f"  [{stage}] {label:>12s} theta={theta:.6g} rep_rate={m:.4f} ({time.time()-t0:.1f}s)", flush=True)

    meta = base_meta(args, slug, pairs, pmeta, rev, dtype, stage)
    meta["n_prompts"] = len(prompts)
    meta["conditions"] = out_conds
    # no-op control (PROTOCOL §3): raw theta=1.0 must equal fix theta'=1.0 per prompt, exactly
    b_raw = next(c for c in out_conds if c["label"] == "baseline_raw")["per_prompt"]
    b_fix = next(c for c in out_conds if c["label"] == "baseline_fix")["per_prompt"]
    meta["controls"] = {"noop_raw1_eq_fix1": all(a["gen_sha"] == b["gen_sha"] for a, b in zip(b_raw, b_fix))}
    print(f"  control noop_raw1_eq_fix1: {meta['controls']['noop_raw1_eq_fix1']}", flush=True)

    if stage == "open":
        add_quality(args, meta, model, tok)
    return meta


# ---------------------------------------------------------------------------- quality (arm b)
def add_quality(args, meta, gen_model, gen_tok):
    """Frozen preference order (PREREG §4): MAUVE -> judge -> negative mean NLL."""
    pref = args.quality
    conds = meta["conditions"]
    texts = {c["label"]: [r["text"] for r in c["per_prompt"]] for c in conds}
    prompts_used = None
    from run_a1 import PROMPTS
    prompts_used = PROMPTS[:args.limit] if args.limit else PROMPTS

    if pref == "none":
        meta["quality_measure"] = "none"
        meta["quality_unit"] = None
        return

    # generation is complete, so free the generator before loading a scorer (a 7B judge + a 7B
    # generator do not co-reside on a 24GB card).  --keep-gen-model-on-gpu opts out.
    if gen_model is not None and args.device.startswith("cuda") and not args.keep_gen_model_on_gpu:
        import torch
        gen_model.to("cpu")
        torch.cuda.empty_cache()

    order = [pref] if pref != "auto" else ["mauve", "judge", "nll"]
    reasons = []
    for cand in order:
        try:
            if cand == "mauve":
                ok = quality_mauve(args, meta, texts)
            elif cand == "judge":
                ok = quality_judge(args, meta, texts, prompts_used, gen_model)
            else:
                ok = quality_nll(args, meta, texts)
            if ok:
                meta["quality_measure"] = cand
                meta["quality_fallback_reason"] = reasons or None
                print(f"  quality measure = {cand}", flush=True)
                return
            reasons.append(f"{cand}: unavailable")
        except Exception as e:                                  # never let an untested path sink the run
            reasons.append(f"{cand}: {type(e).__name__}: {e}")
            print(f"  quality '{cand}' unavailable -> {type(e).__name__}: {e}", flush=True)
    meta["quality_measure"] = "none"
    meta["quality_unit"] = None
    meta["quality_fallback_reason"] = reasons


def quality_mauve(args, meta, texts):
    """MAUVE vs the theta=1.0 reference.  Set-level scalar => featurize once, bootstrap prompt indices."""
    import mauve                                                # noqa: F401  (availability probe)
    ref = texts["baseline_raw"]
    dev = 0 if args.device.startswith("cuda") else -1
    feats, point = {}, {}
    pf = None
    for lab, txt in texts.items():
        out = mauve.compute_mauve(p_text=ref, q_text=txt, device_id=dev,
                                  max_text_length=args.mauve_max_len, verbose=False,
                                  featurize_model_name=args.mauve_featurizer)
        pf = out.p_features if pf is None else pf
        feats[lab] = out.q_features
        point[lab] = float(out.mauve)
    rng = random.Random(args.seed)
    n = len(ref)
    idxs = [[rng.randrange(n) for _ in range(n)] for _ in range(args.mauve_boot)]
    boots = {lab: [] for lab in texts}
    for idx in idxs:
        for lab in texts:
            o = mauve.compute_mauve(p_features=pf[idx], q_features=feats[lab][idx],
                                    device_id=dev, verbose=False)
            boots[lab].append(float(o.mauve))
    for c in meta["conditions"]:
        c["quality_point"] = point[c["label"]]
        c["quality_boot"] = boots[c["label"]]
    meta["quality_unit"] = "set_bootstrap"
    meta["quality_detail"] = {"featurizer": args.mauve_featurizer, "boot": args.mauve_boot,
                              "reference": "baseline_raw (theta=1.0)"}
    return True


def quality_judge(args, meta, texts, prompts_used, gen_model):
    """Blind, randomized-order 1-5 coherence with the cached Qwen2.5-7B-Instruct (judge_lib)."""
    import torch
    import judge_lib
    judge_lib.JUDGE = args.judge_model
    jtok, jm = judge_lib.load(args.device)
    pool = []                                                   # (cond_label, prompt_idx, judge_prompt)
    for c in meta["conditions"]:
        for r in c["per_prompt"]:
            pool.append((c["label"], r["prompt_idx"],
                         JUDGE_RUBRIC.format(prompt=prompts_used[r["prompt_idx"]],
                                             text=r["text"][:args.judge_max_chars])))
    rng = random.Random(args.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)                                          # blind + randomized presentation order
    scored = judge_lib.score(jtok, jm, [pool[i][2] for i in order], 1, 5,
                             device=args.device, bs=args.judge_bs)
    got = {}
    for pos, i in enumerate(order):
        got[(pool[i][0], pool[i][1])] = scored[pos]
    n_bad = 0
    for c in meta["conditions"]:
        for r in c["per_prompt"]:
            v = got.get((c["label"], r["prompt_idx"]))
            if v is None:
                n_bad += 1
            r["quality"] = None if v is None else float(v)
    meta["quality_unit"] = "prompt"
    meta["quality_detail"] = {"judge": args.judge_model, "scale": [1, 5], "blind": True,
                              "randomized_order": True, "seed": args.seed,
                              "n_unparseable": n_bad, "rubric": JUDGE_RUBRIC}
    del jm
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return n_bad < len(pool)                                    # all-unparseable => fall through


def quality_nll(args, meta, texts):
    """Mean NLL of each generation under a fixed held-out reference model; stored NEGATED (higher=better)."""
    import torch
    rm, rt, rrev = load_lm(args.nll_ref, args.device, "float32" if args.device == "cpu" else "bfloat16")
    with torch.no_grad():
        for c in meta["conditions"]:
            for r in c["per_prompt"]:
                ids = rt(r["text"], return_tensors="pt").input_ids[:, :args.nll_max_tokens].to(args.device)
                if ids.shape[1] < 2:
                    r["quality"] = None
                    continue
                r["quality"] = -float(rm(ids, labels=ids).loss)
    del rm
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    meta["quality_unit"] = "prompt"
    meta["quality_detail"] = {"nll_ref": args.nll_ref, "nll_ref_revision": rrev,
                              "sign": "negated mean NLL (higher = better)",
                              "max_tokens": args.nll_max_tokens,
                              "caveat": "paired within-model contrast, so reference-family overlap "
                                        "affects both arms equally"}
    return True


# ---------------------------------------------------------------------------- stage: gauge (arm c)
def run_gauge(args):
    """A1 flip-rate test with c=+-5 at every matched pair, under raw and under fix (PREREG §4c)."""
    import torch
    import run_a1
    from run_a1 import PROMPTS
    from run_a1_loopcheck import greedy

    slug = slug_of(args.model)
    levels = [float(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS[slug]
    pairs, pmeta = load_pairs(resolve_pairs_json(args.pairs_json), slug, levels)
    dtype = pick_dtype(args.dtype, args.device, pmeta.get("fixcal_dtype"))
    model, tok, rev = load_lm(args.model, args.device, dtype)

    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS
    pids = [tok(p)["input_ids"] for p in prompts]
    cs = [float(x) for x in args.gauge_cs.split(",")]
    if len(cs) != 2:
        die("--gauge-cs must be exactly two values (the A1 c=-5,+5 contrast)")

    def gen(ids, c, theta, fix):
        run_a1.FIX = fix                                        # run_a1.generate() reads this module global
        with torch.no_grad():
            g, _ = run_a1.generate(model, ids, c, theta, args.max_new, "greedy", args.seed, args.device)
        return g

    def flip_rows(theta, fix):
        per = []
        for j, ids in enumerate(pids):
            a = gen(ids, cs[0], theta, fix)
            b = gen(ids, cs[1], theta, fix)
            n = min(len(a), len(b))
            d = sum(1 for i in range(n) if a[i] != b[i])
            per.append({"prompt_idx": j, "n_tokens": n, "n_flips": d, "flip_rate": (d / n if n else 0.0)})
        tot = sum(r["n_tokens"] for r in per)
        return per, (sum(r["n_flips"] for r in per) / tot if tot else 0.0)

    rows = []
    # mandatory no-op control first: theta=1.0 is a provable gauge no-op under BOTH operators
    for op, fix in (("raw", False), ("fix", True)):
        per, pooled = flip_rows(1.0, fix)
        rows.append({"kind": "noop_control", "theta_raw": None, "op": op, "theta": 1.0,
                     "per_prompt": per, "pooled_flip_rate": pooled})
        print(f"  [gauge] noop {op} theta=1.0 flip={pooled:.6f}", flush=True)
    for p in pairs:
        for op, fix, th in (("raw", False, p["theta_raw"]), ("fix", True, p["theta_fix"])):
            t0 = time.time()
            per, pooled = flip_rows(th, fix)
            rows.append({"kind": "pair", "theta_raw": p["theta_raw"], "theta_fix": p["theta_fix"],
                         "op": op, "theta": th, "is_deployed_anchor": p["is_deployed_anchor"],
                         "per_prompt": per, "pooled_flip_rate": pooled})
            print(f"  [gauge] pair theta_raw={p['theta_raw']:g} {op} theta={th:.6g} "
                  f"flip={pooled:.6f} ({time.time()-t0:.1f}s)", flush=True)

    # instrumentation control: run_a1.generate(c=0) must reproduce run_a1_loopcheck.greedy()
    run_a1.FIX = False
    with torch.no_grad():
        ga, _ = run_a1.generate(model, pids[0], 0.0, pairs[0]["theta_raw"], args.max_new,
                                "greedy", args.seed, args.device)
        gb = greedy(model, pids[0], pairs[0]["theta_raw"], args.max_new, args.device, False)
    noop = {r["op"]: r["pooled_flip_rate"] for r in rows if r["kind"] == "noop_control"}

    meta = base_meta(args, slug, pairs, pmeta, rev, dtype, "gauge")
    meta["n_prompts"] = len(prompts)
    meta["cs"] = cs
    meta["gauge"] = rows
    meta["controls"] = {"noop_flip_raw": noop.get("raw"), "noop_flip_fix": noop.get("fix"),
                        "noop_gate_exact_zero": (noop.get("raw") == 0.0 and noop.get("fix") == 0.0),
                        "instrumented_eq_loopcheck": (ga == gb)}
    print(f"  controls: {meta['controls']}", flush=True)
    return meta


# ---------------------------------------------------------------------------- stage: json (arm a1)
def run_json(args):
    import torch
    from run_a2_downstream import JSON_TASKS, gen_batch, first_json, json_valid

    slug = slug_of(args.model)
    levels = [float(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS[slug]
    pairs, pmeta = load_pairs(resolve_pairs_json(args.pairs_json), slug, levels)
    dtype = pick_dtype(args.dtype, args.device, pmeta.get("fixcal_dtype"))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, dtype)).to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"

    prompts, schemas, sidx = [], [], []
    for si, (desc, sch) in enumerate(JSON_TASKS):
        for _ in range(args.json_reps):
            prompts.append(f"Output ONLY a single JSON object describing {desc}. JSON:\n")
            schemas.append(sch)
            sidx.append(si)
    if args.limit:
        prompts, schemas, sidx = prompts[:args.limit], schemas[:args.limit], sidx[:args.limit]

    out_conds = []
    for label, theta, fix, pi in conditions_for(pairs):
        t0 = time.time()
        outs = []
        for i in range(0, len(prompts), args.bs):
            with torch.no_grad():
                outs += gen_batch(model, tok, prompts[i:i + args.bs], theta, fix,
                                  args.device, args.json_max_new, ["\n\n", "```"])
        per = [{"item_idx": k, "schema_idx": sidx[k],
                "valid": int(bool(json_valid(first_json(o), schemas[k]))),
                "text": o[:args.keep_chars]} for k, o in enumerate(outs)]
        rate = sum(r["valid"] for r in per) / len(per)
        out_conds.append({"label": label, "op": ("fix" if fix else "raw"), "theta": theta,
                          "pair_idx": pi, "per_item": per, "json_valid_rate": rate})
        print(f"  [json] {label:>12s} theta={theta:.6g} valid={rate:.4f} ({time.time()-t0:.1f}s)", flush=True)

    meta = base_meta(args, slug, pairs, pmeta, rev, dtype, "json")
    meta["n_json"] = len(prompts)
    meta["json_reps"] = args.json_reps
    meta["conditions"] = out_conds
    b_raw = next(c for c in out_conds if c["label"] == "baseline_raw")["per_item"]
    b_fix = next(c for c in out_conds if c["label"] == "baseline_fix")["per_item"]
    meta["controls"] = {"noop_raw1_eq_fix1": all(a["text"] == b["text"] for a, b in zip(b_raw, b_fix))}
    print(f"  control noop_raw1_eq_fix1: {meta['controls']['noop_raw1_eq_fix1']}", flush=True)
    return meta


# ---------------------------------------------------------------------------- stage: humaneval (arm a2)
def load_humaneval(args):
    """Canonical problems.  Order: human-eval package -> datasets openai_humaneval -> vendored JSONL."""
    if args.humaneval_jsonl:
        probs = [json.loads(l) for l in open(args.humaneval_jsonl) if l.strip()]
        return {p["task_id"]: p for p in probs}, f"jsonl:{args.humaneval_jsonl}"
    try:
        from human_eval.data import read_problems
        return dict(read_problems()), "human-eval package"
    except Exception:
        pass
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    return ({r["task_id"]: {"task_id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
                            "entry_point": r["entry_point"]} for r in ds},
            "datasets:openai/openai_humaneval")


def truncate_completion(text):
    """Standard human-eval completion extraction: cut at the earliest stop sequence."""
    cut = len(text)
    for s in HE_STOP:
        i = text.find(s)
        if 0 <= i < cut:
            cut = i
    return text[:cut]


def run_humaneval_gen(args):
    import torch
    from run_a2_downstream import gen_batch

    slug = slug_of(args.model)
    levels = [float(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS[slug]
    pairs, pmeta = load_pairs(resolve_pairs_json(args.pairs_json), slug, levels)
    dtype = pick_dtype(args.dtype, args.device, pmeta.get("fixcal_dtype"))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, dtype)).to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"

    problems, source = load_humaneval(args)
    tids = sorted(problems, key=lambda t: int(t.split("/")[1]))
    if args.limit:
        tids = tids[:args.limit]
    he_prompts = [problems[t]["prompt"] for t in tids]

    out_conds = []
    for label, theta, fix, pi in conditions_for(pairs):
        t0 = time.time()
        comps = []
        for i in range(0, len(he_prompts), args.bs):
            with torch.no_grad():
                comps += gen_batch(model, tok, he_prompts[i:i + args.bs], theta, fix,
                                   args.device, args.he_max_new, HE_STOP)
        out_conds.append({"label": label, "op": ("fix" if fix else "raw"), "theta": theta, "pair_idx": pi,
                          "completions": {t: truncate_completion(c) for t, c in zip(tids, comps)},
                          "completions_untruncated": {t: c for t, c in zip(tids, comps)}})
        print(f"  [humaneval-gen] {label:>12s} theta={theta:.6g} ({time.time()-t0:.1f}s)", flush=True)

    meta = base_meta(args, slug, pairs, pmeta, rev, dtype, "humaneval")
    meta["humaneval_source"] = source
    meta["humaneval_stop"] = HE_STOP
    meta["task_ids"] = tids
    meta["problems"] = {t: problems[t] for t in tids}
    meta["conditions"] = out_conds
    meta["executed"] = False
    print("  NOTE: generation only -- no code was executed.  Score with: "
          "--stage humaneval --exec --score-from <this file>", flush=True)
    return meta


# ---- the sandbox (PREREG §8: mandatory, non-negotiable) --------------------------------------------
def exec_sample(program, timeout=10.0, nproc=64, mem_bytes=2 * 1024 ** 3,
                fsize=8 * 1024 ** 2, pyflags=("-I", "-S")):
    """Execute ONE model-generated program in a hardened throwaway subprocess.  Never in-process,
    never with cwd inside /work.  Returns (status, returncode, seconds, stderr_head)."""
    tmp = tempfile.mkdtemp(prefix="matched_he_")
    try:
        with open(os.path.join(tmp, "prog.py"), "w") as f:
            f.write(program)
        opath, epath = os.path.join(tmp, "stdout.txt"), os.path.join(tmp, "stderr.txt")
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}   # scrubbed: no proxy/HF/PYTHONPATH

        def _limits():                                          # child-only, applied before exec
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        t0 = time.time()
        with open(opath, "wb") as so, open(epath, "wb") as se:
            proc = subprocess.Popen([sys.executable, *pyflags, "prog.py"], cwd=tmp, env=env,
                                    stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                    start_new_session=True, preexec_fn=_limits)
            try:
                rc = proc.wait(timeout=timeout)
                status = "pass" if rc == 0 else "fail"
            except subprocess.TimeoutExpired:
                try:                                            # kill the WHOLE group, not just the child
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                rc, status = None, "timeout"
        dt = time.time() - t0
        try:
            err = open(epath, "rb").read(2000).decode("utf-8", "replace")
        except Exception:
            err = ""
        return status, rc, dt, err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)                  # tempdir removed, always


def run_humaneval_score(args):
    """Separate opt-in scoring step.  Loads NO model; reads completions written by the generation step."""
    src = args.score_from
    if not src or not os.path.exists(src):
        die("--stage humaneval --exec requires --score-from <completions json> "
            "(or run the generation stage first)")
    d = json.load(open(src))
    problems, tids = d["problems"], d["task_ids"]
    print(f"  [humaneval-exec] scoring {src}: {len(tids)} problems x {len(d['conditions'])} conditions "
          f"(timeout {args.exec_timeout}s, AS {args.exec_mem_mb}MB, NPROC {args.exec_nproc})", flush=True)
    out_conds = []
    for c in d["conditions"]:
        t0, per = time.time(), []
        for t in tids:
            p = problems[t]
            prog = p["prompt"] + c["completions"].get(t, "") + "\n" + p["test"] + \
                f"\ncheck({p['entry_point']})\n"
            st, rc, dt, err = exec_sample(prog, timeout=args.exec_timeout, nproc=args.exec_nproc,
                                          mem_bytes=args.exec_mem_mb * 1024 ** 2)
            per.append({"task_id": t, "status": st, "returncode": rc, "seconds": round(dt, 3),
                        "passed": int(st == "pass"), "stderr_head": err[:400]})
        rate = sum(r["passed"] for r in per) / max(1, len(per))
        out_conds.append({"label": c["label"], "op": c["op"], "theta": c["theta"],
                          "pair_idx": c["pair_idx"], "per_problem": per, "humaneval_pass1": rate,
                          "n_timeout": sum(1 for r in per if r["status"] == "timeout")})
        print(f"  [humaneval-exec] {c['label']:>12s} pass@1={rate:.4f} "
              f"timeouts={out_conds[-1]['n_timeout']} ({time.time()-t0:.1f}s)", flush=True)

    meta = {k: v for k, v in d.items() if k not in ("conditions", "problems")}
    meta["stage"] = "humaneval_scored"
    meta["scored_from"] = src
    meta["executed"] = True
    meta["exec_sandbox"] = {"per_sample_subprocess": True, "cwd": "fresh tempfile.mkdtemp() (removed after)",
                            "timeout_s": args.exec_timeout, "kill": "SIGKILL to the whole process group",
                            "rlimit_as_mb": args.exec_mem_mb, "rlimit_nproc": args.exec_nproc,
                            "rlimit_fsize_mb": 8, "rlimit_core": 0,
                            "python_flags": ["-I", "-S"], "env": "scrubbed (no proxy/HF/PYTHONPATH)",
                            "stdin": "DEVNULL", "stdout_stderr": "files inside the tempdir"}
    meta["conditions"] = out_conds
    return meta


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="MATCHED (MATCHED) matched-suppression head-to-head")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "bfloat16", "float16"],
                    help="auto = the dtype FIXCAL calibrated the pairs under (fp32 on cpu)")
    ap.add_argument("--stage", required=True, choices=["pairs", "gauge", "open", "json", "humaneval"])
    ap.add_argument("--exec", dest="do_exec", action="store_true",
                    help="OPT-IN: execute model-generated code (humaneval scoring). Default OFF.")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N units (prompts / json items / humaneval problems); 0 = all")
    ap.add_argument("--pairs-json", default=None,
                    help=f"FIXCAL summary.json (default: {PAPER_PAIRS} if present, else the vendored copy)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--levels", default=None,
                    help="comma theta_raw anchors; must be a subset of the frozen FIXCAL allowlist")
    ap.add_argument("--max-new", type=int, default=256, help="FIXCAL calibrated at 256 -- keep for matching")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gauge-cs", default="-5,5")
    # arm (b) quality
    ap.add_argument("--quality", default="auto", choices=["auto", "mauve", "judge", "nll", "none"])
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--judge-bs", type=int, default=8)
    ap.add_argument("--judge-max-chars", type=int, default=2000)
    ap.add_argument("--keep-gen-model-on-gpu", action="store_true",
                    help="do NOT offload the generator before loading the quality scorer (default: offload, "
                         "since a 7B judge and a 7B generator do not co-reside on 24GB)")
    ap.add_argument("--nll-ref", default="gpt2-large")
    ap.add_argument("--nll-max-tokens", type=int, default=512)
    ap.add_argument("--mauve-featurizer", default="gpt2-large")
    ap.add_argument("--mauve-max-len", type=int, default=256)
    ap.add_argument("--mauve-boot", type=int, default=200)
    # arm (a)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--json-reps", type=int, default=8)
    ap.add_argument("--json-max-new", type=int, default=160)
    ap.add_argument("--keep-chars", type=int, default=600)
    ap.add_argument("--he-max-new", type=int, default=512)
    ap.add_argument("--humaneval-jsonl", default=None)
    ap.add_argument("--score-from", default=None, help="completions json to score (with --exec)")
    ap.add_argument("--exec-timeout", type=float, default=10.0)
    ap.add_argument("--exec-nproc", type=int, default=64)
    ap.add_argument("--exec-mem-mb", type=int, default=2048)
    args = ap.parse_args()

    t0 = time.time()
    if args.stage == "humaneval":
        if args.do_exec and args.score_from:                    # scoring only -- no model is loaded
            meta = run_humaneval_score(args)
            meta["elapsed_s"] = time.time() - t0
            write_out(args.out or scored_path(args.score_from), meta)
            return
        gmeta = run_humaneval_gen(args)                         # generation (no execution)
        gmeta["elapsed_s"] = time.time() - t0
        gen_out = args.out or default_out("humaneval", slug_of(args.model))
        write_out(gen_out, gmeta)
        if args.do_exec:                                        # generate-then-score in one invocation
            args.score_from = gen_out
            meta = run_humaneval_score(args)
            meta["elapsed_s"] = time.time() - t0
            write_out(scored_path(gen_out), meta)
        return

    if args.stage == "gauge":
        meta, stage_name = run_gauge(args), "gauge"
    elif args.stage == "json":
        meta, stage_name = run_json(args), "json"
    else:
        meta, stage_name = run_openlike(args, args.stage), args.stage
    meta["elapsed_s"] = time.time() - t0
    write_out(args.out or default_out(stage_name, slug_of(args.model)), meta)


def scored_path(p):
    return p[:-5] + "_scored.json" if p.endswith(".json") else p + "_scored.json"


def default_out(stage, slug):
    root = os.path.abspath(os.path.join(HERE, ".."))  # code/ -> repo root
    return os.path.join(root, "runs", "MATCHED", f"raw_{slug}_{stage}.json")


if __name__ == "__main__":
    main()
