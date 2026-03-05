# Depth Sweep Conclusions

Sweep: 5 operations × 3 layer depths (L=1, L=2, L=3), modular arithmetic mod 113, 40K epochs.

## Key Findings

### 1. Depth does not universally accelerate grokking

Deeper networks are not strictly better. L=3 addition reaches only 28% test accuracy at 40K epochs,
far worse than L=1 (grokked ~epoch 5,000) and L=2 (grokked ~epoch 3,700). Similarly, L=3 x²+y²
collapses to ~2% test accuracy—the worst result across the entire sweep.

### 2. x³+ab never groks at any depth

x³+ab (a³ + ab mod 113) fails to grok at L=1, L=2, or L=3. This is not a depth limitation but
reflects the algebraic structure of the task: x³+ab lacks the symmetric Fourier decomposition that
enables the trigonometric algorithm used for addition and subtraction. More depth does not fix a
fundamental representational mismatch.

### 3. Multiplication uniquely benefits from depth

Multiplication is the only operation that clearly improves with depth. It never groks at L=1 within
40K epochs (or groks very late), but groks at L=3 around epoch 1,600—the fastest grokking event
across the entire depth sweep. L=2 multiplication also groks, but later than L=3. The additional
layers appear to provide the compositional capacity needed for modular multiplication's Fourier
structure.

### 4. Weight norm decreases with depth (for addition)

Final weight norms for addition: L=1 ≈ 48.9, L=2 ≈ 25.4, L=3 ≈ 20.2. Deeper models converge to
lower-norm solutions, consistent with implicit regularization effects of additional parameters under
AdamW. However, lower weight norm at L=3 does not guarantee faster grokking.

### 5. Fourier sparsity (Gini) stays high for add/sub at all depths

Gini coefficients for addition and subtraction remain in the 0.94–0.97 range regardless of depth
(for runs that grok). The sparse trigonometric algorithm is robust to depth—when the model does
grok, it finds the same qualitative solution.

### 6. L=3 produces incomplete or unstable grokking for most operations

At 40K epochs, most L=3 runs (except multiplication) either never cross 95% test accuracy or show
very slow, incomplete grokking trajectories. The L=3 optimization landscape appears harder to
navigate under the full-batch AdamW setup used here, possibly due to gradient flow issues through
deeper residual paths without LayerNorm.

## Summary Table

| Operation | L=1 Grok | L=2 Grok | L=3 Grok |
|-----------|-----------|-----------|-----------|
| a + b     | ~5,000    | ~3,700    | Never (28% acc) |
| a - b     | ~5,000    | ~4,000    | Never |
| a × b     | Never     | Late      | ~1,600 (fastest) |
| a² + b²   | ~8,000    | ~6,000    | Never (2% acc) |
| a³ + ab   | Never     | Never     | Never |

## Takeaway

Depth is a double-edged sword for grokking. It can unlock operations that are hard for shallow
networks (multiplication), but it also destabilizes training for operations that shallow networks
handle reliably (addition, x²+y²). The most robust grokking behavior is observed at L=1 and L=2
for symmetric operations, and at L=3 only for multiplication.
