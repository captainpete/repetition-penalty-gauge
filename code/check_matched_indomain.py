"""C1 in-domain suppression check — rule FROZEN before computation:
metric = rep_rate via public/code/run_a1_loopcheck.degen(prompt_ids, gen) on untruncated
HumanEval completions, prompt = the task's HumanEval prompt, tokenizer at pinned revision.
MATCHED iff |rep_rate(raw) - rep_rate(fix)| <= 0.01 (FIXCAL tolerance), per pair.
Control: theta=1.0 raw and fix conditions must agree. Direction reported either way."""
import json, sys
sys.path.insert(0, 'code')  # run from the repository root
from run_a1_loopcheck import degen
from transformers import AutoTokenizer

d = json.load(open('results/matched_strength/raw_Qwen2.5-Coder-7B_humaneval.json'))
tok = AutoTokenizer.from_pretrained(d['model'], revision=d['revision'])
prompts = {tid: tok(p['prompt'], add_special_tokens=False)['input_ids'] for tid, p in d['problems'].items()}
res = {}
for c in d['conditions']:
    rates, n_short = [], 0
    for tid, text in c['completions_untruncated'].items():
        gen = tok(text, add_special_tokens=False)['input_ids']
        if len(gen) < 8:
            n_short += 1; continue
        rates.append(degen(prompts[tid], gen)[0])
    res[c['label']] = {'op': c['op'], 'theta': c['theta'], 'n': len(rates), 'short': n_short,
                       'rep_rate': sum(rates)/len(rates)}
for k, v in res.items():
    print(f"{k:14s} op={v['op']:3s} theta={v['theta']:<8} n={v['n']:3d} (short-skip {v['short']:2d}) rep_rate={v['rep_rate']:.4f}")
print('\n== verdicts (frozen tol 0.01) ==')
b_raw = next(v for v in res.values() if v['op']=='raw' and v['theta']==1.0)
b_fix = next(v for v in res.values() if v['op']=='fix' and v['theta']==1.0)
print(f"control theta=1.0: raw {b_raw['rep_rate']:.4f} fix {b_fix['rep_rate']:.4f} agree={abs(b_raw['rep_rate']-b_fix['rep_rate'])<=0.005}")
for tr, tf in [(1.02, 1.28125), (1.05, 3.25), (1.1, 6.625)]:
    r = next(v for v in res.values() if v['op']=='raw' and abs(v['theta']-tr)<1e-9)
    f = next(v for v in res.values() if v['op']=='fix' and abs(v['theta']-tf)<1e-9)
    gap = f['rep_rate'] - r['rep_rate']
    verdict = 'MATCHED' if abs(gap) <= 0.01 else ('FIX SUPPRESSES LESS (in-domain dose lighter)' if gap>0 else 'FIX SUPPRESSES MORE')
    print(f"pair {tr}->{tf}: raw {r['rep_rate']:.4f} fix {f['rep_rate']:.4f} gap {gap:+.4f} -> {verdict}")
