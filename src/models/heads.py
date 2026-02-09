"""
Heads mapping latent contexts to stochastic simulation parameters.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadBase(nn.Module):
    """Base interface for heads producing drift and volatility."""

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - interface
        raise NotImplementedError


class GBMHead(HeadBase):
    """Geometric Brownian Motion parameter head."""

    def __init__(self, latent_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(latent_size)
        self.mu_proj = nn.Linear(latent_size, 1)
        self.sigma_proj = nn.Linear(latent_size, 1)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = self.norm(h_t)
        mu = self.mu_proj(h_t).squeeze(-1)
        sigma = F.softplus(self.sigma_proj(h_t)).squeeze(-1) + 1e-6
        return mu, sigma


class SDEHead(HeadBase):
    """Generic stochastic differential equation parameter head."""

    def __init__(self, latent_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_out = nn.Linear(hidden, 1)
        self.sigma_out = nn.Linear(hidden, 1)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(h_t)
        mu = self.mu_out(features).squeeze(-1)
        sigma = F.softplus(self.sigma_out(features)).squeeze(-1) + 1e-6
        return mu, sigma


# ---------------------------------------------------------------------------
# Horizon-aware head (per-step mu_t, sigma_t via cross-attention)
# ---------------------------------------------------------------------------


class HorizonHead(HeadBase):
    """Predict per-step drift and volatility by cross-attending to backbone output.

    Instead of compressing the full backbone sequence into a single
    ``(mu, sigma)`` pair, this head generates **horizon-length** parameter
    trajectories ``mu_1 … mu_H`` and ``sigma_1 … sigma_H``.  Each horizon
    step is represented by a learned query embedding that cross-attends to
    the full backbone sequence, allowing the model to express time-varying
    dynamics (volatility clustering, momentum decay, regime transitions).

    Architecture::

        Backbone sequence  (batch, seq, d_model)
                │
                ▼
        ┌─────────────────────┐
        │  Learned horizon     │  (horizon_max, d_model) queries
        │  position embeddings │  + sinusoidal base
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Multi-head cross-   │  queries=horizon, keys/values=backbone seq
        │  attention (×n_layers)│
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Per-step projection │  d_model → (mu_t, sigma_t)
        └────────┬────────────┘
                 │
                 ▼
        mu: (batch, horizon)   sigma: (batch, horizon)

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    horizon_max:
        Maximum prediction horizon supported (queries are sliced for shorter).
    nhead:
        Number of attention heads in cross-attention layers.
    n_layers:
        Number of stacked cross-attention + feedforward layers.
    d_ff:
        Hidden size of the per-layer feedforward network.
    dropout:
        Dropout applied inside attention and feedforward.
    """

    def __init__(
        self,
        latent_size: int,
        horizon_max: int = 48,
        nhead: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.horizon_max = horizon_max
        d_ff = d_ff or latent_size * 2

        # Learned horizon query embeddings
        self.horizon_queries = nn.Parameter(
            torch.randn(horizon_max, latent_size) * 0.02
        )

        # Sinusoidal positional encoding for the horizon queries
        pe = torch.zeros(horizon_max, latent_size)
        position = torch.arange(0, horizon_max, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / latent_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: latent_size // 2])
        self.register_buffer("pe", pe)

        # Cross-attention decoder layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            latent_size, nhead, dropout=dropout, batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(latent_size),
                        "ff": nn.Sequential(
                            nn.Linear(latent_size, d_ff),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_ff, latent_size),
                            nn.Dropout(dropout),
                        ),
                        "norm2": nn.LayerNorm(latent_size),
                    }
                )
            )

        # Per-step projection to (mu_t, sigma_t)
        self.mu_proj = nn.Linear(latent_size, 1)
        self.sigma_proj = nn.Linear(latent_size, 1)

    def forward(
        self,
        h_seq: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_seq : (batch, seq_len, d_model) — full backbone sequence output
        horizon : prediction length (≤ horizon_max)

        Returns
        -------
        mu : (batch, horizon) — per-step drift
        sigma : (batch, horizon) — per-step volatility (positive)
        """
        batch = h_seq.shape[0]
        h = min(horizon, self.horizon_max)

        # Build horizon queries: learned embedding + sinusoidal position
        queries = (self.horizon_queries[:h] + self.pe[:h]).unsqueeze(0).expand(batch, -1, -1)

        # Cross-attention layers
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](queries, h_seq, h_seq)
            queries = layer["norm1"](queries + attn_out)
            ff_out = layer["ff"](queries)
            queries = layer["norm2"](queries + ff_out)

        # Project to per-step parameters
        mu = self.mu_proj(queries).squeeze(-1)                           # (batch, horizon)
        sigma = F.softplus(self.sigma_proj(queries)).squeeze(-1) + 1e-6  # (batch, horizon)
        return mu, sigma


# ---------------------------------------------------------------------------
# Neural Bridge Head (hierarchical macro + micro texture)
# ---------------------------------------------------------------------------


class NeuralBridgeHead(HeadBase):
    """Hierarchical head: predicts 1-Hour *macro* move and sub-hour *micro* texture.

    Instead of outputting ``(mu, sigma)`` for an external simulation loop, this
    head produces the path tensor **directly** by combining:

    1. **Macro projection** – a single predicted log-return for the full hour.
    2. **Texture network** – learned deviations (wiggles) between start and end.
    3. **Bridge constraint** – forces ``texture[0] == texture[-1] == 0`` so the
       generated micro path starts at the current price and lands exactly at the
       macro-predicted price.

    Architecture::

        h_t  (batch, d_model)   ← last-step backbone embedding
              │
         ┌────┴────┐
         │         │
         ▼         ▼
      macro_proj  texture_net
      (Linear→1)  (MLP → micro_steps)
         │         │
         │         ▼
         │    Bridge constraint
         │    (zero endpoints)
         │         │
         ▼         ▼
      linear_path + bridge  →  micro_returns (batch, micro_steps)

    The head can optionally convert log-returns to absolute prices when
    ``current_price`` is supplied.

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    micro_steps:
        Number of sub-hour steps to generate (e.g. 12 for 5-min, 60 for 1-min).
    hidden_dim:
        Hidden size of the texture MLP.
    """

    def __init__(self, latent_size: int, micro_steps: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.micro_steps = micro_steps
        self.norm = nn.LayerNorm(latent_size)

        # 1. Macro Projector (Where are we going in 1 hour?)
        self.macro_proj = nn.Linear(latent_size, 1)

        # 2. Texture Generator (How do we get there?)
        self.texture_net = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, micro_steps),
        )

        # 3. Volatility Projector (How uncertain is the path?)
        self.sigma_proj = nn.Linear(latent_size, 1)

    def forward(
        self,
        h_t: torch.Tensor,
        current_price: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_t : (batch, d_model) — last-step backbone embedding
        current_price : (batch,), optional — anchors output as absolute prices

        Returns
        -------
        macro_ret : (batch, 1) — predicted 1H log-return
        micro_path : (batch, micro_steps) — sub-hour mean log-return path
            (or absolute prices when ``current_price`` is given)
        sigma : (batch,) — predicted volatility scale for stochastic sampling
        """
        # Normalize backbone output (critical for purely-linear backbones like DLinear)
        h_t = self.norm(h_t)

        # A. Predict the Macro Destination (1 Hour later)
        macro_ret = self.macro_proj(h_t)  # (batch, 1)

        # B. Predict the Texture (the wiggles)
        raw_texture = self.texture_net(h_t)  # (batch, micro_steps)

        # C. Predict Volatility
        sigma = F.softplus(self.sigma_proj(h_t)).squeeze(-1) + 1e-6  # (batch,)

        # D. Enforce Bridge Constraints (start=0, end=0)
        texture = raw_texture - raw_texture[:, 0:1]  # shift so start == 0
        # Rotate so end is also 0: texture_t -= (t/T) * texture_T
        steps = torch.arange(self.micro_steps, device=h_t.device).float() / (self.micro_steps - 1)
        correction = raw_texture[:, -1:] * steps.unsqueeze(0)
        bridge = texture - correction

        # E. Construct the Final Path
        linear_path = macro_ret * steps.unsqueeze(0)  # (batch, micro_steps)
        micro_returns = linear_path + bridge

        # Optionally convert to absolute prices
        if current_price is not None:
            micro_returns = torch.clamp(micro_returns, min=-20.0, max=20.0)
            return macro_ret, current_price.unsqueeze(-1) * torch.exp(micro_returns), sigma

        return macro_ret, micro_returns, sigma
