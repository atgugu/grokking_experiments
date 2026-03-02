"""Activation caching via forward hooks for mechanistic interpretability."""

import torch
import torch.nn as nn

from .transformer import GrokkingTransformer


class ActivationCache:
    """Context manager that hooks into a GrokkingTransformer to extract internal activations.

    Captures:
    - Attention patterns: (batch, n_heads, seq, seq) per block
    - MLP pre-activations: (batch, seq, d_mlp) per block
    - MLP post-activations: (batch, seq, d_mlp) per block
    - Residual stream: (batch, seq, d_model) after each block

    Usage:
        with ActivationCache(model) as cache:
            logits = model(x)
            attn = cache.attention_patterns  # list of (batch, n_heads, seq, seq)
            neurons = cache.neuron_activations  # list of (batch, seq, d_mlp)
            residuals = cache.residual_stream  # list of (batch, seq, d_model)
    """

    def __init__(self, model: GrokkingTransformer):
        self.model = model
        self._handles: list = []
        self.attention_patterns: list[torch.Tensor] = []
        self.neuron_activations: list[torch.Tensor] = []  # post-ReLU
        self.neuron_pre_activations: list[torch.Tensor] = []  # pre-ReLU
        self.residual_stream: list[torch.Tensor] = []

    def __enter__(self):
        self.clear()

        for i, block in enumerate(self.model.blocks):
            # Hook attention: capture per-head weights
            handle = block.attn.register_forward_hook(self._make_attn_hook(i))
            self._handles.append(handle)

            # Hook MLP: capture pre and post activation
            # The MLP is Sequential(Linear, ReLU/GELU, Linear)
            # Hook the activation function to get pre/post
            mlp_linear1 = block.mlp[0]  # First linear layer
            mlp_act = block.mlp[1]  # Activation function
            handle_pre = mlp_linear1.register_forward_hook(self._make_mlp_pre_hook(i))
            handle_post = mlp_act.register_forward_hook(self._make_mlp_post_hook(i))
            self._handles.append(handle_pre)
            self._handles.append(handle_post)

            # Hook block output for residual stream
            handle_res = block.register_forward_hook(self._make_residual_hook(i))
            self._handles.append(handle_res)

        return self

    def __exit__(self, *args):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_attn_hook(self, block_idx: int):
        def hook(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            # attn_weights: (batch, seq, seq) when average_attn_weights=True
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attention_patterns.append(attn_weights.detach())
        return hook

    def _make_mlp_pre_hook(self, block_idx: int):
        def hook(module, input, output):
            # output of first linear: (batch, seq, d_mlp) before activation
            self.neuron_pre_activations.append(output.detach())
        return hook

    def _make_mlp_post_hook(self, block_idx: int):
        def hook(module, input, output):
            # output of activation: (batch, seq, d_mlp) after ReLU/GELU
            self.neuron_activations.append(output.detach())
        return hook

    def _make_residual_hook(self, block_idx: int):
        def hook(module, input, output):
            # Block output: (batch, seq, d_model)
            self.residual_stream.append(output.detach())
        return hook

    def clear(self):
        """Clear all cached activations."""
        self.attention_patterns.clear()
        self.neuron_activations.clear()
        self.neuron_pre_activations.clear()
        self.residual_stream.clear()
