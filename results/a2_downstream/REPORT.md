# A2 — downstream task quality (raw vs fix) — RESULT

Qwen/Qwen2.5-Coder-7B, HumanEval n=164, JSON n=48, θ∈[1.0, 1.1, 1.3].

| metric | op | θ=1.0 | θ=1.1 | θ=1.3 | Δ(θ1.0−θ1.3) |
|---|---|---|---|---|---|
| HumanEval pass@1 | raw | 0.079 | 0.043 | 0.000 | +0.079 |
| HumanEval pass@1 | fix | 0.079 | 0.085 | 0.079 | +0.000 |
| JSON valid-rate | raw | 1.000 | 1.000 | 0.000 | +1.000 |
| JSON valid-rate | fix | 1.000 | 1.000 | 1.000 | +0.000 |

## Verdict
- **HumanEval pass@1: CORRUPTION MATTERS, FIX RECOVERS.** raw drops +0.079 from θ1.0→θ1.3; the fix drops only +0.000 (recovers most of it). End-to-end evidence.
- **JSON valid-rate: CORRUPTION MATTERS, FIX RECOVERS.** raw drops +1.000 from θ1.0→θ1.3; the fix drops only +0.000 (recovers most of it). End-to-end evidence.

**Overall: the structured-output corruption is real end-to-end** on at least one metric, and the normalize-before-penalize fix recovers it — converts the paper's implication to evidence.
