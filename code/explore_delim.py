#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, not pre-registered) — re-slice A2 raw data with a
brackets-only delimiter definition (excludes whitespace, which inflated prose) to
see whether the domain-specificity and confidence-first clauses fail because the
effect is genuinely weak or because the pre-registered 'structural' tag was too loose.
"""
import json, random
TEST = (1.1, 1.2, 1.3, 1.4, 1.5)
DELIM = set('{}[]()",:;=')   # grammar-obligatory delimiters; NO whitespace, NO '.'/"'"-in-prose


def is_delim(s):
    t = s.strip()
    return bool(t) and all(c in DELIM for c in t)


def boot_diff(a, b, n=10000, seed=0):
    r = random.Random(seed)
    out = sorted((sum(a[r.randrange(len(a))] for _ in range(len(a))) / len(a))
                 - (sum(b[r.randrange(len(b))] for _ in range(len(b))) / len(b)) for _ in range(n))
    return out[250], out[9750]


d = json.load(open("runs/A2/raw.json"))
R = d["records"]

# domain: delimiter-flip rate per token, code vs prose
def rate(domains):
    rs = []
    for r in R:
        if r["domain"] in domains and r["theta"] in TEST:
            f = sum(1 for p in r["positions"] if p["flip"] and is_delim(p.get("top1_str", "")))
            rs.append(f / max(1, len(r["positions"])))
    return rs
code, prose = rate(("json", "python")), rate(("prose",))
cm, pm = sum(code)/len(code), sum(prose)/len(prose)
lo, hi = boot_diff(code, prose)
print(f"[delim-only] code {cm:.4f} vs prose {pm:.4f} -> ratio {cm/pm if pm else float('inf'):.1f}x  diffCI[{lo:.4f},{hi:.4f}]")

# confidence-first restricted to delimiter tokens (penalized, positive)
zf, zn = [], []
for r in R:
    if r["domain"] in ("json", "python") and r["theta"] in TEST:
        for p in r["positions"]:
            if p["top1_seen"] and p["top1_pos"] and is_delim(p.get("top1_str", "")):
                (zf if p["flip"] else zn).append(p["z_top"])
if zf and zn:
    lo2, hi2 = boot_diff(zf, zn)
    print(f"[delim-only] confidence-first: z_top flipped {sum(zf)/len(zf):.2f} (n={len(zf)}) "
          f"vs non-flip {sum(zn)/len(zn):.2f} (n={len(zn)})  diffCI[{lo2:.2f},{hi2:.2f}]")

# what fraction of all flips are delimiters, code vs prose?
for dom in (("json","python"), ("prose",)):
    tot = fl = dl = 0
    for r in R:
        if r["domain"] in dom and r["theta"] in TEST:
            for p in r["positions"]:
                tot += 1
                if p["flip"]:
                    fl += 1
                    dl += is_delim(p.get("top1_str",""))
    print(f"[{'/'.join(dom)}] flips={fl}/{tot} ({100*fl/tot:.1f}%), of which delimiters={dl} ({100*dl/max(1,fl):.0f}% of flips)")
