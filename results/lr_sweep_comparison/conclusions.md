# Learning Rate Sweep: Conclusions

## Sweep Summary

5 learning rate values tested on the standard 1-layer transformer (d_model=128, 4 heads, d_mlp=512) trained on modular addition (a + b mod 113) for 40K epochs with weight decay 1.0 and train fraction 0.3.

| LR | Grok Epoch | Train Acc | Test Acc | Gini | W Norm | Key Freqs |
|------:|----------:|---------:|---------:|-----:|------:|-----------|
| 1e-4 | Never | 1.0000 | 0.1799 | 0.916 | 54.7 | [89, 24, 68, 45, 65] |
| 3e-4 | 20,400 | 1.0000 | 1.0000 | 0.935 | 35.5 | [68, 45, 31, 82, 89] |
| 1e-3 | 8,350 | 1.0000 | 1.0000 | 0.945 | 48.9 | [68, 45, 31, 82, 24] |
| 3e-3 | 1,500 | 1.0000 | 1.0000 | 0.958 | 33.2 | [24, 89, 68, 45, 61] |
| 1e-2 | 300* | 0.0112 | 0.0078 | 0.991 | 2.3 | [23, 68, 5, 108, 86] |

Grok epoch = first epoch where test accuracy reaches 95% (Never = did not grok within 40K epochs).
*lr=1e-2 groks at epoch 300 but catastrophically collapses at epoch 18,400 and never recovers.

## Key Observations

### 1. Learning rate controls grokking speed across two orders of magnitude

Among stable grokking runs, higher LR dramatically accelerates the memorization-to-generalization transition:
- **lr=3e-4**: 20,400 epochs
- **lr=1e-3**: 8,350 epochs (2.4x faster)
- **lr=3e-3**: 1,500 epochs (13.6x faster than 3e-4)

Each 3x increase in LR produces roughly a 3-5x speedup in grokking. The relationship is approximately linear in log-log space.

### 2. Too-low LR prevents grokking entirely

At lr=1e-4 (10x below default), the model memorizes perfectly (100% train accuracy) but achieves only 18% test accuracy after 40K epochs. Despite having high Gini coefficient (0.916) and developing some Fourier structure, the optimization dynamics are too slow for the trigonometric algorithm to overtake memorization within the training budget. The weight norm reaches 54.7 -- the highest of all runs -- suggesting insufficient regularization pressure at this learning rate.

### 3. Too-high LR causes catastrophic late collapse

lr=1e-2 produces the most dramatic result: the model groks extremely fast (epoch 300) and maintains perfect test accuracy for 18,000 epochs. Then at epoch 18,400, it catastrophically collapses to near-chance performance (0.78% test accuracy) and never recovers. The final weight norm of only 2.3 suggests the solution was driven to an extremely sparse, fragile representation that eventually destabilized.

### 4. The instability window is narrow

The transition from "stable grokking" to "eventual collapse" happens between lr=3e-3 (stable, 1,500 epoch grok) and lr=1e-2 (collapse at 18,400). This is only a ~3x increase in learning rate. The model at lr=3e-3 shows no signs of instability, making the failure mode at 1e-2 sudden rather than gradual.

### 5. Gini coefficient is high across all runs

Unlike the weight decay and train fraction sweeps where non-grokking runs had clearly lower Gini, here all runs develop high Fourier sparsity (0.916-0.991). Even the non-grokking lr=1e-4 run has Gini 0.916, suggesting it is developing Fourier structure but simply hasn't completed the transition. This indicates the learning rate primarily affects the *speed* of the phase transition rather than whether the Fourier structure can form.

### 6. Key frequencies show partial overlap across grokking runs

Among the three stable grokking runs, frequencies {68, 45} appear in all three, with {31, 82} shared between lr=3e-4 and lr=1e-3. The consistency is stronger than in the train fraction sweep (where all runs used different splits) because all LR sweep runs use the same train/test split. The specific frequency selection still varies with learning rate, suggesting the optimization path influences which frequencies get selected.

## Causal Mechanism

The results reveal learning rate's role in the memorization-generalization tradeoff:

1. **Too slow** (lr=1e-4): The effective regularization from weight decay is proportional to `wd * lr` in AdamW. At lr=1e-4, the effective regularization is 10x weaker than at lr=1e-3, so memorization remains cheaper than the algorithmic solution even after 40K epochs.
2. **Optimal range** (lr=3e-4 to 3e-3): Higher LR increases both the optimization speed *and* the effective regularization pressure, accelerating the transition to the trigonometric algorithm.
3. **Too fast** (lr=1e-2): The solution converges rapidly to the algorithm but the large step size prevents the optimizer from settling into a stable minimum. After thousands of epochs, accumulated noise drives the parameters out of the grokked basin.

The key insight is that in AdamW, weight decay and learning rate interact multiplicatively: the actual weight decay applied per step is `wd * lr * param`. This means lr=1e-4 with wd=1.0 has the same effective regularization as lr=1e-3 with wd=0.1 -- and the weight decay sweep showed that wd=0.1 barely groks. This unifies the WD and LR sweep results.

## Figures

- [`lr_sweep_test_accuracy.png`](lr_sweep_test_accuracy.png) -- Test accuracy curves for all 5 learning rates
- [`lr_sweep_test_accuracy_log.png`](lr_sweep_test_accuracy_log.png) -- Test accuracy on log-scale epoch axis
- [`lr_sweep_train_loss.png`](lr_sweep_train_loss.png) -- Training loss curves
- [`lr_sweep_grokking_time.png`](lr_sweep_grokking_time.png) -- Grokking epoch vs. learning rate
- [`lr_sweep_gini_evolution.png`](lr_sweep_gini_evolution.png) -- Gini coefficient evolution over training
- [`lr_sweep_weight_norm.png`](lr_sweep_weight_norm.png) -- Weight norm evolution over training
- [`lr_sweep_animation.gif`](lr_sweep_animation.gif) -- Animated 2x2 panel showing training dynamics over time
