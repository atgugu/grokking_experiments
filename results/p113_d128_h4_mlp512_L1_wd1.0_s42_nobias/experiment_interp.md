# Analysis: Why Gini Goes Up as Test Loss Goes Down in the No-Bias Experiment

## Context

The no-bias run (`p113_d128_h4_mlp512_L1_wd1.0_s42_nobias`) trains a 1-layer transformer with `mlp_bias=false` on modular addition (a+b mod 113). The question: why does the Fourier Gini coefficient rise simultaneously with test loss falling?

## Answer: They are two views of the same phase transition (grokking)

The Gini coefficient and test loss are **mechanistically linked** — the Gini rise *causes* the test loss drop. Here's why:

### What Gini measures
- Gini = sparsity of the Fourier frequency energy distribution across the logit table
- **Low Gini (~0.36)**: energy spread uniformly across all 113 frequencies → the network uses a complex, non-generalizable lookup table (memorization)
- **High Gini (~0.93)**: energy concentrated in a handful of key frequencies → the network has learned a sparse trigonometric algorithm that computes `(a+b) mod 113` using `cos(2πk·a/113)` and `sin(2πk·b/113)` for a few key k values

### The causal chain
1. **Weight decay** continuously penalizes all parameters equally
2. The memorization solution requires large weights spread across many frequencies
3. Weight decay gradually erodes the memorization solution, forcing the network toward a **simpler representation**
4. The simplest correct solution is the sparse trig algorithm (a few frequencies)
5. As the network consolidates into fewer frequencies: **Gini goes up**
6. The sparse trig algorithm **generalizes perfectly** (it computes the true function): **test loss goes down**

### Timeline in the no-bias run

| Phase | Epochs | Gini | Test Loss | What's happening |
|-------|--------|------|-----------|------------------|
| Init | 0 | 0.92 | 4.73 | Random init (spurious sparsity) |
| Memorization | 200-4000 | 0.43→0.36 | 19→17 | Network memorizes training set; Gini drops as it uses many frequencies |
| Overfitting plateau | 4000-12000 | 0.36→0.41 | 17→19.6 | Memorization deepens; test loss keeps rising |
| **Grokking transition** | **12000-22000** | **0.41→0.99** | **19.6→0.0** | Weight decay erodes noisy frequencies; trig algorithm crystallizes |
| Post-grokking | 22000-40000 | 0.93 | ~0 | Stable sparse solution; continued slow refinement |

### Comparison with standard (bias=true) run
- **With bias**: grokking at epochs ~6000-8000 (fast, sharp transition)
- **No bias**: grokking at epochs ~12000-22000 (2-3x slower, more gradual)
- The no-bias model takes longer because without bias terms in the MLP, it has fewer parameters to express the trig algorithm, so weight decay needs more time to steer the network toward the sparse solution

### Key frequency evidence
The top-5 key frequencies at convergence are [68, 45, 31, 82, 24] — these are specific frequencies k where the network computes `cos(2πk/113)` and `sin(2πk/113)` components. The fact that only ~5 out of 113 frequencies carry most of the energy is exactly what Gini measures.

## Summary

**Gini up + test loss down = grokking.** Weight decay forces the network from a memorized solution (all frequencies, low Gini, bad generalization) to a sparse trigonometric algorithm (few frequencies, high Gini, perfect generalization). The two metrics are correlated because they measure the same underlying transition from different angles: Gini from the mechanistic/Fourier perspective, test loss from the performance perspective.
