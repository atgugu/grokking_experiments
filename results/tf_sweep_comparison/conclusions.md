# Train Fraction Sweep: Conclusions

## Sweep Summary

7 train fraction values tested on the standard 1-layer transformer (d_model=128, 4 heads, d_mlp=512) trained on modular addition (a + b mod 113) for 40K epochs with weight decay 1.0.

| TF | # Train | # Test | Grok Epoch | Train Acc | Test Acc | Gini | Key Freqs |
|-----:|-------:|------:|----------:|----------:|---------:|-----:|-----------|
| 0.05 | 638 | 12131 | Never | 1.0000 | 0.0056 | 0.570 | [43, 70, 95, 18, 19] |
| 0.1 | 1277 | 11492 | Never | 1.0000 | 0.0036 | 0.459 | [79, 34, 85, 28, 104] |
| 0.2 | 2554 | 10215 | Never | 1.0000 | 0.0065 | 0.368 | [79, 34, 95, 18, 68] |
| 0.3 | 3831 | 8938 | 8350 | 1.0000 | 1.0000 | 0.945 | [68, 45, 31, 82, 24] |
| 0.4 | 5108 | 7661 | 1900 | 1.0000 | 1.0000 | 0.951 | [87, 26, 109, 4, 20] |
| 0.5 | 6384 | 6385 | 500 | 1.0000 | 1.0000 | 0.960 | [91, 22, 89, 24, 60] |
| 0.7 | 8938 | 3831 | 400 | 1.0000 | 1.0000 | 0.941 | [83, 30, 92, 21, 89] |

Grok epoch = first epoch where test accuracy reaches 95% (Never = did not grok within 40K epochs).

## Key Observations

### 1. Sharp phase transition at train fraction >= 0.3

Below tf=0.3, the model never groks within 40K epochs despite perfectly memorizing the training set:
- **tf=0.05** (638 train): Test accuracy 0.56% -- pure memorization with no generalization.
- **tf=0.1** (1277 train): Test accuracy 0.36% -- similarly stuck at chance.
- **tf=0.2** (2554 train): Test accuracy 0.65% -- still no meaningful generalization.

At tf=0.3 (3831 train), the model suddenly groks at epoch 8,350. The transition from "never groks" to "groks" is abrupt -- there is no partial generalization regime.

### 2. Grokking speed accelerates dramatically with more data

Among grokking runs, more training data leads to much faster generalization:
- **tf=0.3**: 8,350 epochs
- **tf=0.4**: 1,900 epochs (4.4x faster)
- **tf=0.5**: 500 epochs (16.7x faster)
- **tf=0.7**: 400 epochs (20.9x faster)

The relationship is strongly nonlinear: doubling from tf=0.3 to tf=0.5 produces a 16.7x speedup, while further increasing from tf=0.5 to tf=0.7 yields only marginal improvement.

### 3. Gini coefficient cleanly separates grokking from non-grokking

Non-grokking runs have final Gini coefficients of 0.37--0.57, indicating diffuse Fourier energy spread across many frequencies. Grokking runs reach 0.94--0.96, reflecting strong concentration into a sparse set of trigonometric components. The gap between non-grokking (max 0.57) and grokking (min 0.94) is large and unambiguous.

### 4. Non-grokking runs show *decreasing* Gini with more data

Counter-intuitively, among non-grokking runs, Gini decreases as train fraction increases (0.57 at tf=0.05, 0.46 at tf=0.1, 0.37 at tf=0.2). With more training data but insufficient data to trigger the algorithmic solution, the memorization solution becomes more complex (more diffuse frequency usage), not simpler.

### 5. Key frequencies vary across grokking runs

Unlike the weight decay sweep where grokking runs share a common pool of key frequencies, train fraction sweep runs each select different dominant frequencies:
- tf=0.3: {68, 45, 31, 82, 24}
- tf=0.4: {87, 26, 109, 4, 20}
- tf=0.5: {91, 22, 89, 24, 60}
- tf=0.7: {83, 30, 92, 21, 89}

This suggests that many valid frequency subsets can support the trigonometric algorithm, and the specific choice depends on the training data distribution (different random splits).

### 6. Diminishing returns above tf=0.5

The jump from tf=0.5 (500 epochs) to tf=0.7 (400 epochs) is only a 20% speedup despite 40% more training data. Once sufficient data is available, the constraint on grokking speed shifts from data quantity to the optimization dynamics of transitioning from memorization to the algorithmic solution.

## Causal Mechanism

The results support a data-threshold interpretation of grokking:

1. The trigonometric algorithm for modular addition requires learning to embed inputs on circles at key frequencies -- this is a structured, low-dimensional solution
2. With too little data (tf < 0.3), the training set does not sufficiently constrain the solution space -- memorization remains cheaper than learning the algorithm even under weight decay pressure
3. At the critical threshold (~30% of all pairs = 3,831 samples), the training set provides enough coverage of the cyclic group that the algorithmic solution becomes favored under regularization
4. Beyond the threshold, more data makes the memorization solution increasingly expensive relative to the algorithm, accelerating the transition
5. Weight decay (held at 1.0 across all runs) provides the regularization pressure, but data quantity determines whether that pressure can drive the network to the algorithmic solution

## Figures

- [`tf_sweep_test_accuracy.png`](tf_sweep_test_accuracy.png) -- Test accuracy curves for all 7 train fractions
- [`tf_sweep_test_accuracy_log.png`](tf_sweep_test_accuracy_log.png) -- Test accuracy on log-scale epoch axis
- [`tf_sweep_train_loss.png`](tf_sweep_train_loss.png) -- Training loss curves
- [`tf_sweep_grokking_time.png`](tf_sweep_grokking_time.png) -- Grokking epoch vs. train fraction
- [`tf_sweep_gini_evolution.png`](tf_sweep_gini_evolution.png) -- Gini coefficient evolution over training
- [`tf_sweep_weight_norm.png`](tf_sweep_weight_norm.png) -- Weight norm evolution over training
- [`tf_sweep_animation.gif`](tf_sweep_animation.gif) -- Animated 2x2 panel showing training dynamics over time
