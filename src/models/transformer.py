"""Grokking transformer for modular arithmetic (Nanda et al. 2023).

Architecture:
- Token embedding W_E: (p+1) x d_model
- Positional embedding W_pos: 3 x d_model
- 1 transformer block (attention + MLP) with residual connections
- Unembedding W_U: d_model x p
- Predict at position 2 (the = token position)
- No LayerNorm by default
- ReLU activation (not GELU)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Single transformer block: multi-head attention + MLP with residuals.

    Matches Nanda et al.: no LayerNorm by default, ReLU activation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        activation: str = "relu",
        use_layernorm: bool = False,
        mlp_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True, bias=False,
        )
        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=mlp_bias),
            act_fn,
            nn.Linear(d_mlp, d_model, bias=mlp_bias),
        )
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Cache attention weights for visualization
        self._attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        self._attn_weights = attn_weights.detach()  # (batch, seq, seq) averaged over heads
        x = x + attn_out
        if self.use_layernorm:
            x = self.norm1(x)
        x = x + self.mlp(x)
        if self.use_layernorm:
            x = self.norm2(x)
        return x


class GrokkingTransformer(nn.Module):
    """Transformer for modular arithmetic grokking experiments.

    Input: [a, b, =] (3 tokens, vocab_size = p+1)
    Output: logits over p classes (prediction at = position)

    Key differences from kripkenstein ArithmeticTransformer:
    - 3-token input (not 2)
    - Separate unembedding matrix W_U (not tied)
    - Predict at = position (not mean-pool)
    - ReLU activation (not GELU)
    - Scaled normal init (std=0.8/sqrt(d_model)) matching TransformerLens
    """

    def __init__(
        self,
        p: int = 113,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
        n_layers: int = 1,
        activation: str = "relu",
        use_layernorm: bool = False,
        tie_embeddings: bool = False,
        mlp_bias: bool = True,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_bias = mlp_bias
        self.vocab_size = p + 1  # p integers + equals token
        self.num_classes = p

        # Embeddings
        self.W_E = nn.Embedding(self.vocab_size, d_model)  # token embedding
        self.W_pos = nn.Embedding(3, d_model)  # positional embedding (3 positions)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                activation=activation,
                use_layernorm=use_layernorm,
                mlp_bias=mlp_bias,
            )
            for _ in range(n_layers)
        ])

        # Unembedding
        if tie_embeddings:
            # Tie W_U to W_E[:p] (first p rows)
            self.W_U = None
            self._tie_embeddings = True
        else:
            self.W_U = nn.Linear(d_model, p, bias=False)
            self._tie_embeddings = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Scaled normal initialization matching TransformerLens / Nanda et al."""
        init_std = 0.8 / math.sqrt(self.d_model)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 3) integer token indices [a, b, =]

        Returns:
            logits: (batch, p) unnormalized class scores
        """
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device)

        # Embed: token + positional
        h = self.W_E(x) + self.W_pos(positions)  # (batch, 3, d_model)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Extract representation at = position (position 2)
        h_eq = h[:, 2, :]  # (batch, d_model)

        # Unembed
        if self._tie_embeddings:
            logits = h_eq @ self.W_E.weight[:self.p].T  # (batch, p)
        else:
            logits = self.W_U(h_eq)  # (batch, p)

        return logits

    @torch.no_grad()
    def get_logit_table(self, device: torch.device) -> torch.Tensor:
        """Compute full p x p x p logit table for Fourier analysis.

        Returns:
            (p, p, p) tensor where [a, b, c] = logit for class c given input (a, b).
        """
        self.eval()
        p = self.p

        # Build all p^2 inputs
        a_vals = torch.arange(p, device=device).repeat_interleave(p)
        b_vals = torch.arange(p, device=device).repeat(p)
        eq_vals = torch.full((p * p,), self.vocab_size - 1, dtype=torch.long, device=device)
        inputs = torch.stack([a_vals, b_vals, eq_vals], dim=1)  # (p^2, 3)

        # Forward in chunks to avoid OOM
        chunk_size = 1024
        all_logits = []
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            logits = self.forward(chunk)
            all_logits.append(logits)

        logits = torch.cat(all_logits, dim=0)  # (p^2, p)
        logit_table = logits.reshape(p, p, p)  # (a, b, class)
        self.train()
        return logit_table

    def get_attention_patterns(self) -> list[torch.Tensor]:
        """Return cached attention patterns from last forward pass.

        Returns:
            List of (batch, seq, seq) tensors, one per block.
        """
        patterns = []
        for block in self.blocks:
            if block._attn_weights is not None:
                patterns.append(block._attn_weights)
        return patterns

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
