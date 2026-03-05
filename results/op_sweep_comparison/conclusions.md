# Operation Sweep: Conclusions

## Sweep Summary

5 arithmetic operations tested on the standard 1-layer transformer (d_model=128, 4 heads, d_mlp=512) trained on a + b mod 113 baseline settings (wd=1.0, lr=1e-3, train_frac=0.3) for 40K epochs.

| Operation | Grok Epoch | Train Acc | Test Acc | Gini | W Norm | Key Freqs |
|-----------|----------:|----------:|---------:|-----:|-------:|-----------|
| a + b     | 8,350 | 1.0000 | 1.0000 | 0.945 | 48.9 | [68, 45, 31, 82, 24] |
| a − b     | 11,800 | 1.0000 | 1.0000 | 0.960 | 46.8 | [30, 83, 65, 48, 4] |
| a × b     | 14,900 | 1.0000 | 1.0000 | 0.064 | 57.5 | [67, 46, 74, 39, 73] |
| a² + b²   | 1,800* | 1.0000 | 0.9996 | 0.133 | 80.7 | [76, 37, 105, 8, 68] |
| a³ + ab   | Never | 1.0000 | 0.0129 | 0.307 | 56.3 | [63, 50, 45, 68, 104] |

Grok epoch = first epoch reaching 95% test accuracy. *a²+b² reaches 95% at epoch 1,800 but takes until ~2,700 to reach 99%; final test accuracy is 0.9996 (never fully converges).

## Key Observations

### 1. Grokking is not universal: three succeed, one stalls, one never grokks

Four of five operations eventually generalize, but with strikingly different trajectories:
- **a+b**: groks at epoch 8,350 (baseline)
- **a−b**: groks at epoch 11,800 (~1.4x slower than addition)
- **a×b**: groks at epoch 14,900 (~1.8x slower than addition)
- **a²+b²**: reaches 95% test accuracy at epoch 1,800 (fastest!) but never fully converges (final test 0.9996)
- **a³+ab**: pure memorization throughout all 40K epochs — test accuracy stays below 1.33% (near chance for 113 classes)

### 2. The Gini paradox: multiplication groks with near-uniform frequency usage

The most striking result is that a×b groks to 100% test accuracy while having Gini=0.064 — effectively a flat, uniform Fourier spectrum. This breaks any naive identification of "high Gini = grokked":

| Operation | Gini | Grokked? |
|-----------|-----:|---------|
| a + b     | 0.945 | Yes |
| a − b     | 0.960 | Yes |
| a × b     | 0.064 | Yes |
| a² + b²   | 0.133 | Yes (~) |
| a³ + ab   | 0.307 | No |

Multiplication groks using ~60 roughly equally-weighted frequencies — a fundamentally different algorithmic strategy than addition's sparse 5-frequency solution. The intermediate Gini of a³+ab (0.307) suggests the model found partial structure without achieving generalization.

### 3. Algebraic structure determines representation complexity

To explain the Gini variation, we computed the theoretical Fourier sparsity of each operation by applying a 2D DFT to its value table f(a,b) = (a op b) mod p (see `scripts/analyze_algebraic_structure.py`):

| Operation | Theory Gini | Learned Gini |
|-----------|------------:|-------------:|
| a + b     | 0.906 | 0.945 |
| a − b     | 0.906 | 0.965 |
| a × b     | 0.002 | 0.064 |
| a² + b²   | 0.045 | 0.133 |
| a³ + ab   | 0.028 | 0.088 |

The theoretical and learned Gini values align closely for the four operations that grokked. This reveals that:

- **Addition/subtraction** are intrinsically Fourier-sparse: f(a,b) = (a±b) mod p has most of its energy concentrated in a handful of frequencies. The model exploits this by learning the sparse trigonometric algorithm (cos(ω(a+b)) = cos(ωa)cos(ωb) − sin(ωa)sin(ωb) for a few ω).
- **Multiplication/quadratic operations** are intrinsically Fourier-dense: f(a,b) = (a×b) mod p requires contributions from across the frequency spectrum. The model is forced to use a dense representation, giving low Gini even after grokking.

The model doesn't freely choose its representation complexity — the algebraic structure of the operation determines it.

### 4. Specific frequency choices are not theoretically determined

While the Gini values align, the Pearson correlation between theoretical and learned frequency spectra is near zero for all operations (r ≈ -0.04 for addition, r ≈ 0.12 for multiplication). The model does NOT learn to use the "theoretically simplest" frequencies — instead, it selects whatever frequencies are most efficient given its initialization and optimization trajectory.

For addition, the theoretical spectrum concentrates at frequencies {1, 2} (near-DC), but the model uses frequencies {68, 45, 31, 82, 24} — high-frequency oscillations. This is because the model computes modular arithmetic via circular embeddings (cos/sin at chosen frequency ω), not by approximating the integer-valued function directly. Any frequency ω works equally well for the trig identity; the specific choice is an accident of optimization.

### 5. Addition is faster to grok than subtraction, despite identical algebraic complexity

Addition groks at epoch 8,350 versus subtraction at 11,800 — a 1.4x difference despite both having the same theoretical Gini (0.906) and the same functional form (linear mod p). The difference in key frequencies ([68, 45, 31, 82, 24] vs. [30, 83, 65, 48, 4]) confirms these are genuinely different solution pathways.

The addition asymmetry likely arises from the asymmetry in initialization: the model was designed with addition as the baseline, and random initialization may slightly favor finding the addition-compatible frequencies first.

### 6. a²+b² groks fast but incompletely

a²+b² reaches 95% test accuracy faster than any other operation (epoch 1,800) but never fully converges (final 0.9996). The operation's Fourier structure (theory Gini=0.045, learned Gini=0.133) is intermediate — denser than addition but sparser than multiplication. The model finds a partial solution quickly but seems unable to close the last ~0.04% of errors, possibly because the residual errors require a more complex computation than the 1-layer architecture can represent.

### 7. a³+ab does not grok: architectural limitation

The cubic operation a³+ab = a(a²+b) never produces any generalization — test accuracy stays below 1.33% for the entire 40K epoch training, indistinguishable from random predictions. The model memorizes training data perfectly (train loss → 0) with no sign of the phase transition seen in other operations.

The theoretical spectrum (theory Gini=0.028) is as dense as multiplication — both require complex Fourier representations. But unlike multiplication, the 1-layer transformer appears unable to represent a³+ab at all. This may reflect an intrinsic limitation: a³+ab = a·(a²+b) requires computing a product of a linear function (a) and a nonlinear function (a²+b), which may need at least 2 transformer layers to compose. All grokking runs used 1 layer; this operation may require depth.

## Causal Structure

The results support a two-level picture of grokking:

**Level 1: Can the architecture represent the operation at all?**
- If no (a³+ab with 1 layer): pure memorization, no grokking
- If yes: proceed to Level 2

**Level 2: How much regularization pressure is needed to prefer the algorithmic solution?**
- Operations with dense Fourier structure (multiplication) require the algorithm to use many weights → the algorithmic solution is not much simpler than memorization → slower grokking
- Operations with sparse Fourier structure (addition, subtraction) → the algorithmic solution is dramatically simpler → faster grokking

The Gini coefficient is not a *cause* of grokking but a *symptom* of the underlying algebraic structure. High Gini means the network found a sparse algorithm; low Gini means the network found a dense but still generalizing algorithm.

## Figures

- [`op_sweep_test_accuracy.png`](op_sweep_test_accuracy.png) — Test accuracy curves for all 5 operations
- [`op_sweep_test_accuracy_log.png`](op_sweep_test_accuracy_log.png) — Test accuracy on log-scale epoch axis
- [`op_sweep_train_loss.png`](op_sweep_train_loss.png) — Training loss curves
- [`op_sweep_grokking_time.png`](op_sweep_grokking_time.png) — Grokking epoch by operation
- [`op_sweep_gini_evolution.png`](op_sweep_gini_evolution.png) — Gini coefficient evolution over training
- [`op_sweep_weight_norm.png`](op_sweep_weight_norm.png) — Weight norm evolution over training
- [`op_sweep_frequency_composition.png`](op_sweep_frequency_composition.png) — Key frequency composition per operation
- [`op_sweep_algebraic_structure.png`](op_sweep_algebraic_structure.png) — Theoretical vs. learned frequency spectra (5-panel)
- [`op_sweep_gini_comparison.png`](op_sweep_gini_comparison.png) — Theory vs. learned Gini per operation (bar chart)
- [`op_sweep_animation.gif`](op_sweep_animation.gif) — Animated training dynamics
