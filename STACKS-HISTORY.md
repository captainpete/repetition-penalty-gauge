# How the sign-branched repetition penalty spread: a git genealogy

Companion to `STACKS-SURVEY.md`. That file establishes *which* stacks carry the CTRL
sign-branch (divide-if-positive / multiply-if-negative on the raw logit) at current HEAD; this
file traces, by `git blame`/`git log`/pickaxe on full-history clones, *how each one got it* — the
introducing commit, the PR/issue behind it, and whether the stack independently re-derived the
operator or copied it from a named prior implementation.

Method: blobless full clones under `stacks/` (gitignored), five lineage-grouped agents. Every
SHA, date, author, and PR number below came from a real command or a fetched GitHub page; items
that could not be confirmed are marked UNVERIFIED. Verified 2026-07-07.

---

## 1. The "fix" was independently invented at least twice

CTRL (Keskar et al. 2019) introduced the multiplicative penalty in its **naive** form: divide
every seen token's logit by θ. Dividing a *negative* logit by θ>1 moves it toward zero and
*raises* its probability — backwards. The **sign-branch** (multiply negatives, divide positives)
is the community's fix for that. The fix was arrived at independently more than once; the gauge
dependence described in the note is present in each derivation from its first commit, and none of
the derivations records it:

- **HuggingFace transformers (Dec 2019)** — the only stack with a true naive→fix cycle *in its own
  history*: naive divide added by thomwolf (`bbc0c86f`, 2019-12-17), fixed to the sign-branch by
  von Platen (issue #2302 → PR #2303, merged 2019-12-25). This is the fix most of the ecosystem
  inherited.
- **llama.cpp (Mar 2023)** — an **independent re-derivation**: beiller (PR #20, `129c7d1e`,
  2023-03-12) added the sign-branch directly from the CTRL paper, reasoning about the
  negative-logit problem himself ("if score < 0 then repetition penalty has to be multiplied"),
  with a `// repetition penalty from CTRL paper` citation. No prior naive form in llama.cpp; not
  copied from HF.
- **NVIDIA FasterTransformer (≤2023)** — a **third lineage**: the sign-branch was already present
  when FasterTransformer's kernel was checked into LMDeploy (`9efcac38`, 2023-06-20) and underlies
  the NVIDIA datacenter branch (TensorRT-LLM, LMDeploy TurboMind). Whether FasterTransformer
  itself derived it independently or copied HF is UNVERIFIED (its own repo history was not audited).

Each derivation stopped at "negative logits are now suppressed too"; none continued to the
question of what the branch point itself depends on.

---

## 2. Propagation graph (edges backed by commit messages / code comments)

```
CTRL 2019 (naive divide)
   │  bug: dividing negative logits raises their prob (HF #2302)
   ▼
┌──────────────────────── independent derivations of the sign-branch "fix" ────────────────────────┐
│                                                                                                    │
│  HF transformers                 llama.cpp (beiller #20)          NVIDIA FasterTransformer         │
│  (naive→fix, PR #2303, 2019)     (re-derived, CTRL cite, 2023)    (sign-branch pre-2023, UNVERIF.) │
└──────┬──────────────────────────────────┬─────────────────────────────────┬──────────────────────┘
       │ copied (class B)                  │ copied comment verbatim         │ imported kernel
       │                                   ▼                                 ▼
   ┌───┼───────────────┐              KoboldCpp (2023-10)          TensorRT-LLM (NVIDIA, 2023)
   │   │               │              (2 own copies)              LMDeploy TurboMind (import #7)
   ▼   ▼               ▼
  TGI TGW  LMDeploy-pytorch   mlx-lm (cites CTRL+HF)   ExLlamaV2 ("as in HF")   candle (Mazare, HF)
  #317 #2916  #1197           #399                     initial commit          #535
   │
   │  vLLM ── HF "parity" (#1424, 2023)
   ▼
  vLLM ──────────┬───────────────┬────────────────┐
                 │ "following     │ "sync kernels  │ "same impl
                 │  vllm"         │  to vllm"      │  as vLLM"
                 ▼                ▼                ▼
              SGLang #973     aphrodite #1403   mistral.rs #1638

  Ollama MLX (#15631, 2026): same idiom, NO comment, parallel adoption — not a textual descendant.
```

Two structural observations:
1. **HF is the root, but vLLM is the amplifier.** SGLang, aphrodite, and mistral.rs each name
   *vLLM* (not HF) as their source in the introducing commit. vLLM's HF-parity port (#1424)
   became the de-facto reference for the server/GPU-inference cluster.
2. **The datacenter-CUDA branch is a separate genealogy** (NVIDIA FasterTransformer →
   TensorRT-LLM, LMDeploy TurboMind), independent of the HF/PyTorch branch.

---

## 3. Per-stack genealogy

| Stack | Introducing commit (sha / date / author / PR) | Origin class | Named source | Default | Notable footgun (beyond by-design gauge dependence) |
|---|---|---|---|---|---|
| **HF transformers** | `bbc0c86f` naive 2019-12-17 thomwolf → `18e5bdbe` fix 2019-12-24 von Platen (**PR #2303**); vectorized `29bdb883` #8598 2020-11-20 | **A (naive→fix)** | — (origin) | off | 3 mutually-exclusive `where` copies (L390/402/411), not double-applied |
| **vLLM** | `69be658b` 2023-10-29 ljss (**#1424**); moved #10681, #18437 | B (copied) | **HF** ("parity") | off | latent: CUDA kernel has no dtype-equality guard (safe via fp32 upcast) |
| **llama.cpp** | `129c7d1e` 2023-03-12 beiller (**#20**); comment `dd7eff57` 2023-04-29 ivanstepanovftw (#1126) | **re-derived** | CTRL paper | off | `--repeat-last-n -1` clamps to 0 = disabled (docs say ctx-size); resolver commented out `common.cpp:1289-1291` |
| **KoboldCpp** | comment inlined `5db89b90` 2023-10-25 (master-merge); 2nd copy `c03302b6` #2167 2026-05-10 | B (copied) | **llama.cpp** (verbatim comment) | off | 2 own copies; two-tier slope is a discontinuous 50/50 step, not a ramp |
| **Ollama (MLX)** | `ff23dd34` 2026-04-18 D. Hiltgen (**#15631**) | B (parallel idiom, no comment) | — (independent) | **on (1.1)** | branches `<0` (vs llama.cpp `<=0`); ring buffer avoids the clamp bug |
| **TGI** | `62f91f78` 2023-05-26 OlivierDehaene (**#317**, a throughput feature) | B (copied) | **HF** | off | server + Gaudi duplicate copies can drift |
| **text-generation-webui** | `3443219c` 2023-06-29 oobabooga (**#2916**) | B (monkeypatch) | **HF** (+range from ExLlama) | off | `-self._range:` with range=0 slices whole context (relies on `-0==0`) |
| **LMDeploy (PyTorch)** | `5ea53acd` 2024-03-01 grimoire (**#1197**) | B (copied) | **HF** three-liner | off | — |
| **LMDeploy (TurboMind CUDA)** | import `9efcac38` 2023-06-20 Li Zhang (**#7**); unified `6aa9a4cd` #4223 2026-01-12 | B (copied) | **NVIDIA FasterTransformer** | off | — |
| **aphrodite-engine** | `b24ec345` 2025-08-11 AlpinDale (**#1403** "sync kernels to vllm") | B (fork copy) | **vLLM** (verbatim) | off | inherits vLLM's latent dtype footgun; own fork-era penalty now dead |
| **SGLang** | `ab787594` 2024-08-08 Juwan Yoo (**#973** "following vllm"); restored #21258 2026-04-01 | B (copied) | **vLLM** | off | seen-set = **output only, prompt exempt** (restoration dropped prompt cumulation) |
| **TensorRT-LLM** | `23bc5b7c` 2023-09-20 Kaiyu Xie ("Initial commit", born sign-branched); unified `d879430b` #846 | B (copied) | NVIDIA (FasterTransformer lineage, UNVERIFIED) | off | temperature applied before the sign test (`penaltyKernels.cu` L190-193 → L204); benign only while temp>0 |
| **mistral.rs** | `d0652804` 2025-09-17 Ryan Li (**#1638** "same implementation as vLLM") | B (copied) | **vLLM** | off (presets 1.1) | subtractive freq/presence applied **before** the sign test (L1233-34 → L1236); operators don't commute |
| **candle** | `4300864c` 2023-08-21 Laurent Mazare (**#535**); to lib #623 | B (from HF knowledge, no cite) | HF (implicit; Mazare is HF/candle author) | examples on (1.1) | uses `>= 0.`; HashSet dedups (presence-once) |
| **mlx-lm** | `b1cc6d0` 2024-02-16 vishal-14069 (**#399**); refactor #1094 2024-11-07 A. Hannun | B (copied, cited) | **CTRL arXiv + HF** (only dual-citation stack) | off | default window only **20 tokens** |
| **ExLlamaV2** | initial `bb834695` 2023-08-30 turboderp; comment `dc474c9` 2023-12-25 | B (likely from exllama v1, UNVERIFIED) | HF ("as in HF repetition penalty" comment) | **on (1.025)** | decay-tail linearly ramps θ→1 for older tokens; **no CUDA mirror exists** (survey said otherwise — see §5); ordering is *correct* (sign on raw logit, subtractive after) |

Every stack: sole/known live path confirmed reachable (no dead code); the sign-branch faithfully
implements CTRL, so the gauge dependence is **by design**, not an additional bug.

---

## 4. Notes

- **Convergent invention.** The sign-branch was derived independently at least twice (HF 2019,
  llama.cpp 2023; plausibly a third time in FasterTransformer), so the gauge dependence is not a
  single propagated mistake.
- **The llama.cpp/KoboldCpp comment** ("The academic publication that described this technique
  actually just only divided… This is common fix for this problem, which is to multiply by the
  penalty instead of dividing.") traces to llama.cpp: beiller's PR #20 wording, reworded in
  ivanstepanovftw's PR #1126 (2023-04-29); KoboldCpp copied it verbatim (2023-10-25).
- **Ordering issues, confirmed at current lines**: mistral.rs applies subtractive penalties before
  the sign test (`sampler.rs` L1233-34 → L1236); TensorRT-LLM applies temperature before it
  (`penaltyKernels.cu` L190-193 → L204). Both are latent (triggered only by nonzero freq/presence,
  or a non-positive temperature, respectively).
- **Defaults**: Ollama (`repeat_penalty` 1.1), ExLlamaV2 (1.025), candle's examples (1.1), and
  GPT4All (1.1) enable the penalty by default; the other surveyed stacks default it off.

---

## 5. Corrections this pass makes to STACKS-SURVEY.md

1. **ExLlamaV2 has no CUDA mirror.** The survey (§2.5) implied a mirrored CUDA kernel; full-history
   search (`git log --all -S'rep_p' -- exllamav2_ext/cuda/`) returns nothing — the rep penalty is
   CPU-only C++. The "mirrored CUDA kernel" expectation likely came from exllama v1 (a different
   repo, not audited). **Applied**: `STACKS-SURVEY.md §2.5` now states CPU-only, no CUDA copy.
2. **SGLang PR numbers.** An earlier issue-tracker sweep referenced June-2026 SGLang PRs
   (#28179/#28181/#28180/#28535) that are **not present** in the clone at HEAD 2026-07-07. The
   actual in-tree restoration of the penalizer is **#21258 (2026-04-01)**; the June numbers appear
   to originate from GitHub issue-listing fetches that returned fabricated future-dated entries, so
   they should not be quoted without re-checking each at its own URL.

(Correction 1 is applied to `STACKS-SURVEY.md §2.5` above; correction 2 concerns a working issue
index, not the survey text.)

---

## 6. Unverified items

- **NVIDIA FasterTransformer**: whether it derived the sign-branch independently or copied HF is
  unknown; its own repo history was not audited. It is treated as a distinct lineage only because
  the operator was already present when imported into LMDeploy (2023-06) with no HF reference.
- **TensorRT-LLM** pre-open-source history is hidden (squashed "Initial commit" 2023-09); its
  internal provenance (presumably FasterTransformer) is inferred, not blamed.
- **ExLlamaV2** pre-repo authorship (exllama v1) not cloned; the sign-branch was present at
  ExLlamaV2's first commit.
- **KoboldCpp** `concedo` branch merges upstream master, so naive date-bisection is unreliable;
  claims used `git log --first-parent` and merge-diffs.
- **mlx-lm** history is preserved (filtered) from `ml-explore/mlx-examples`, so the origin commit
  (#399) is genuine, not a split artifact.
- All classifications are source-reading at one HEAD per repo; no binaries were run.
