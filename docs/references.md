# References

Annotated bibliography of key papers underlying this project's theory, methods, and analysis.

## Primary References

### Power et al. (2022) — Discovery of Grokking

**A. Power, Y. Burda, H. Edwards, I. Babuschkin, V. Misra.** "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." *ICML 2022 Workshop on Mathematical Reasoning.*
[arXiv:2201.02177](https://arxiv.org/abs/2201.02177)

First observation that small transformers trained on algorithmic tasks (modular arithmetic, permutation groups) exhibit *delayed generalization*: the network memorizes perfectly, then — thousands of epochs later — suddenly generalizes. This paper coined the term "grokking" and established the experimental setup (small datasets, weight decay, modular arithmetic) that subsequent work builds on.

### Nanda et al. (2023) — Progress Measures via Mechanistic Interpretability

**N. Nanda, L. Chan, T. Lieberum, J. Smith, J. Steinhardt.** "Progress Measures for Grokking via Mechanistic Interpretability." *ICLR 2023.*
[arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

The primary reference for this project. Shows that the network learns a *trigonometric algorithm*: it represents inputs as points on a circle and uses cosine/sine at a small number of key frequencies to compute modular addition. Introduces four progress measures (restricted loss, excluded loss, Gini coefficient, weight norm) that track the formation of this algorithm during training — revealing that the useful circuit forms *gradually*, long before the sharp jump in test accuracy. Demonstrates that grokking is not sudden but is preceded by steady, measurable progress in the Fourier domain.

### Zhong et al. (2023) — Clock and Pizza

**Z. Zhong, Z. Liu, M. Tegmark, J. Andreas.** "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks." *NeurIPS 2023.*
[arXiv:2306.17844](https://arxiv.org/abs/2306.17844)

Provides complementary geometric interpretations of modular arithmetic circuits. The "Clock" algorithm arranges inputs on a circle (matching Nanda et al.'s trig interpretation) while the "Pizza" algorithm uses a different slicing geometry. Shows that different training runs can converge to distinct but equivalent algorithmic solutions, and provides tools for distinguishing them.

## Supporting References

### Michaud et al. (2023) — Quantization Model of Grokking

**E.J. Michaud, Z. Liu, M. Tegmark.** "The Quantization Model of Neural Scaling." *arXiv preprint.*
[arXiv:2303.13506](https://arxiv.org/abs/2303.13506)

Proposes a theory where neural networks learn discrete "quanta" of capability. Applied to grokking, this framework explains why generalization happens abruptly: the trigonometric circuit must reach a critical strength threshold before it outperforms memorization. The transition is a phase change, not gradual improvement.

### Olsson et al. (2022) — Induction Heads

**C. Olsson, N. Elhage, N. Nanda, N. Joseph, et al.** "In-context Learning and Induction Heads." *Transformer Circuits Thread.*
[arXiv:2209.11895](https://arxiv.org/abs/2209.11895)

Establishes the methodology for mechanistic interpretability of attention patterns in transformers. While focused on in-context learning rather than grokking, the techniques for analyzing attention heads (per-head patterns, QK/OV circuits) are directly applicable to understanding how the grokking transformer routes information from the `a` and `b` positions to the `=` position.

### Elhage et al. (2022) — Toy Models of Superposition

**N. Elhage, T. Hume, C. Olsson, N. Schiefer, et al.** "Toy Models of Superposition." *Transformer Circuits Thread.*
[arXiv:2209.10652](https://arxiv.org/abs/2209.10652)

Foundational work on how neural networks represent more features than they have dimensions, through superposition. Relevant to understanding why the grokking transformer's 128-dimensional embedding space can represent multiple Fourier frequencies simultaneously, and why the MLP neurons specialize to individual frequencies rather than mixing them.

### Transformer Circuits Thread

**Anthropic.** "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread, 2021.*
[transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)

Introduces the conceptual framework for decomposing transformers into interpretable circuits: the residual stream as a communication channel, attention heads as information movers, and MLPs as feature computers. This framework motivates the analysis structure used throughout this project (embedding analysis, attention patterns, neuron-level decomposition, unembedding analysis).
