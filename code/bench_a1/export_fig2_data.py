import json, numpy as np
T = 200
def cum(traces):
    arr = np.full((len(traces), T), np.nan)
    for i, tr in enumerate(traces):
        n = min(len(tr), T); arr[i, :n] = np.cumsum(tr[:n])
    return arr
D = "paper/figdata"
m = {k: cum(v) for k, v in json.load(open("scores_gpt2-large.json")).items()}
nat = {k: cum(v) for k, v in json.load(open("scores_gpt2-large_natural.json")).items()}
base = m["unpenalized"]
cost = lambda a: -(np.nanmean(a - base, axis=0))
cols = {"shipped": cost(nat["ctrl_shipped"]), "median": cost(nat["ctrl_median"]),
        "sub": cost(m["subtractive"]), "norm": cost(m["normalized"])}
with open(f"{D}/fig2a.dat", "w") as f:
    f.write("pos shipped median sub norm\n")
    for t in range(0, T, 2):
        f.write(f"{t+1} " + " ".join(f"{cols[k][t]:.3f}" for k in ("shipped","median","sub","norm")) + "\n")
MODELS = [("gpt2","scores_gpt2.json"),("gpt2large","scores_gpt2-large.json"),
          ("pythia","scores_pythia-2.8b.json"),("qwen","scores_Qwen2.5-7B.json"),
          ("qwenI","scores_Qwen2.5-7B-Instruct.json")]
for short, f_ in MODELS:
    d = {k: cum(v) for k, v in json.load(open(f_)).items()}
    g = np.abs(d["ctrl_p5"] - d["ctrl_m5"])
    med = np.nanmedian(g, axis=0); q1 = np.nanpercentile(g,25,axis=0); q3 = np.nanpercentile(g,75,axis=0)
    with open(f"{D}/fig2b_{short}.dat", "w") as f:
        f.write("pos med q1 q3\n")
        for t in range(0, T, 2):
            f.write(f"{t+1} {med[t]:.3f} {q1[t]:.3f} {q3[t]:.3f}\n")
print("dat files written")
