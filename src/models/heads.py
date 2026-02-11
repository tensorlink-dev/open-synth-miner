"""
Heads mapping latent contexts to stochastic simulation parameters.
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde

logger = logging.getLogger(__name__)


class HeadBase(nn.Module):
    """Base interface for heads producing stochastic simulation parameters.

    Heads map backbone latent representations to parameters for path simulation.
    Different heads return different parameter sets:

    - **GBMHead, SDEHead**: ``(mu, sigma)`` - drift and volatility scalars
    - **HorizonHead**: ``(mu_seq, sigma_seq)`` - per-step drift and volatility sequences
    - **NeuralBridgeHead**: ``(macro_ret, micro_returns, sigma)`` - macro target, micro
      trajectory, and volatility scale

    Concrete return types are documented in each subclass. SynthModel.forward() handles
    head-specific outputs and enforces consistent (batch, n_paths, horizon) path tensors.
    """

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, ...]:  # pragma: no cover - interface
        """Map latent representation to simulation parameters.

        Parameters
        ----------
        h_t : torch.Tensor
            Backbone output of shape (batch, latent_size)

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Head-specific parameters. See subclass docstrings for exact return types.
        """
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
# Neural SDE head (proper SDE integration via torchsde)
# ---------------------------------------------------------------------------


class _LatentSDE(nn.Module):
    """SDE dynamics conditioned on a backbone latent embedding.

    Implements the ``f`` (drift) and ``g`` (diffusion) interface expected by
    :func:`torchsde.sdeint`.  Both functions are neural networks that receive
    the current state ``y`` (log-price, dim 1), the time ``t``, and the
    backbone context vector, producing state-dependent, time-varying dynamics.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, latent_size: int, hidden: int = 64) -> None:
        super().__init__()
        in_dim = latent_size + 2  # context + y(1) + t(1)
        self.f_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.g_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self._ctx: torch.Tensor | None = None

    def set_context(self, ctx: torch.Tensor) -> None:
        """Bind the backbone context for the current integration call."""
        self._ctx = ctx

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_vec = t.expand(y.shape[0], 1)
        inp = torch.cat([self._ctx, y, t_vec], dim=-1)
        return self.f_net(inp)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_vec = t.expand(y.shape[0], 1)
        inp = torch.cat([self._ctx, y, t_vec], dim=-1)
        return F.softplus(self.g_net(inp)) + 1e-6


class NeuralSDEHead(HeadBase):
    """Neural SDE head using ``torchsde`` for numerically stable path integration.

    Instead of outputting ``(mu, sigma)`` for external Euler-Maruyama simulation,
    this head learns drift and diffusion *networks* and integrates them via
    :func:`torchsde.sdeint` (or the adjoint variant for memory-efficient
    training).

    The SDE operates in log-price space::

        d(log S) = f(t, log S | ctx) dt + g(t, log S | ctx) dW

    where ``f`` and ``g`` are MLPs conditioned on the backbone embedding ``ctx``.
    Paths are exponentiated back to price space before being returned.

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    hidden:
        Hidden dimension of the drift / diffusion networks.
    solver:
        SDE solver passed to ``torchsde.sdeint``.  ``'euler'`` (default) is
        fastest; ``'milstein'`` offers better strong convergence;
        ``'srk'`` gives highest accuracy at greater cost.
    adjoint:
        If ``True``, use the adjoint method (O(1) memory) for backpropagation
        through the solver.  Slightly slower per step but essential for long
        horizons or limited GPU memory.
    """

    def __init__(
        self,
        latent_size: int,
        hidden: int = 64,
        solver: str = "euler",
        adjoint: bool = False,
    ) -> None:
        super().__init__()
        self.sde_func = _LatentSDE(latent_size, hidden)
        self.solver = solver
        self.adjoint = adjoint
        # Summary projections for diagnostics / loss compatibility
        self.mu_proj = nn.Linear(latent_size, 1)
        self.sigma_proj = nn.Linear(latent_size, 1)

    def forward(
        self,
        h_t: torch.Tensor,
        initial_price: torch.Tensor,
        horizon: int,
        n_paths: int = 1000,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_t : (batch, d_model) — backbone embedding
        initial_price : (batch,) — current price per sample
        horizon : number of simulation steps
        n_paths : Monte-Carlo paths per sample
        dt : time step size

        Returns
        -------
        paths : (batch, n_paths, horizon) — simulated price paths
        mu : (batch,) — summary drift estimate (diagnostic)
        sigma : (batch,) — summary volatility estimate (diagnostic)
        """
        batch = h_t.shape[0]
        device = h_t.device

        # Diagnostic summary parameters
        mu = self.mu_proj(h_t).squeeze(-1)
        sigma = F.softplus(self.sigma_proj(h_t)).squeeze(-1) + 1e-6

        # Expand context for n_paths: (batch, d) -> (batch * n_paths, d)
        ctx = h_t.unsqueeze(1).expand(-1, n_paths, -1).reshape(batch * n_paths, -1)
        self.sde_func.set_context(ctx)

        # Initial state in log-space: (batch * n_paths, 1)
        y0 = torch.log(initial_price).unsqueeze(-1)
        y0 = y0.unsqueeze(1).expand(-1, n_paths, -1).reshape(batch * n_paths, 1)

        # Time grid
        ts = torch.linspace(0.0, horizon * dt, horizon + 1, device=device)

        # Solve SDE
        integrate = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint
        ys = integrate(self.sde_func, y0, ts, method=self.solver)
        # ys: (horizon + 1, batch * n_paths, 1)

        # Extract paths (skip t=0), clamp, convert from log-space
        log_paths = ys[1:, :, 0].permute(1, 0)  # (batch * n_paths, horizon)
        log_paths = torch.clamp(log_paths, min=-20.0, max=20.0)
        paths = torch.exp(log_paths).reshape(batch, n_paths, horizon)

        return paths, mu, sigma


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

    Architecture (standard)::

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

    When ``kv_dim`` is specified (for memory efficiency), keys/values are
    compressed via a linear projection before attention::

        Backbone sequence  (batch, seq, d_model)
                │
                ▼
        ┌─────────────────────┐
        │  KV Compression     │  Linear: d_model → kv_dim
        │  (optional)         │
        └────────┬────────────┘
                 │
                 ▼  (batch, seq, kv_dim)
        ┌─────────────────────┐
        │  Multi-head cross-   │  Q: (h, d_model), K/V: (kv_dim)
        │  attention          │
        └────────┬────────────┘

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
    kv_dim:
        If specified, compress key/value sequences to this dimension for memory
        efficiency. Enables linear scaling: O(horizon × kv_dim) instead of
        O(horizon × seq_len). Default None (no compression).
    """

    def __init__(
        self,
        latent_size: int,
        horizon_max: int = 48,
        nhead: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        kv_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.horizon_max = horizon_max
        self.kv_dim = kv_dim
        d_ff = d_ff or latent_size * 2

        # Optional KV compression for memory efficiency
        if kv_dim is not None:
            self.kv_compress = nn.Linear(latent_size, kv_dim)
        else:
            self.kv_compress = None

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
                            kdim=kv_dim, vdim=kv_dim,
                        ) if kv_dim is not None else nn.MultiheadAttention(
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
        if horizon > self.horizon_max:
            logger.warning(
                "HorizonHead: requested horizon=%d exceeds horizon_max=%d; "
                "output will be clipped to %d steps. Increase horizon_max in "
                "the head config to support longer horizons.",
                horizon, self.horizon_max, self.horizon_max,
            )
        h = min(horizon, self.horizon_max)

        # Build horizon queries: learned embedding + sinusoidal position
        queries = (self.horizon_queries[:h] + self.pe[:h]).unsqueeze(0).expand(batch, -1, -1)

        # Optional KV compression for memory efficiency
        if self.kv_compress is not None:
            h_seq_kv = self.kv_compress(h_seq)  # (batch, seq_len, kv_dim)
        else:
            h_seq_kv = h_seq  # (batch, seq_len, d_model)

        # Cross-attention layers
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](queries, h_seq_kv, h_seq_kv)
            queries = layer["norm1"](queries + attn_out)
            ff_out = layer["ff"](queries)
            queries = layer["norm2"](queries + ff_out)

        # Project to per-step parameters
        mu = self.mu_proj(queries).squeeze(-1)                           # (batch, horizon)
        sigma = F.softplus(self.sigma_proj(queries)).squeeze(-1) + 1e-6  # (batch, horizon)
        return mu, sigma


class SimpleHorizonHead(HeadBase):
    """Lightweight horizon head without attention - uses pooling + MLPs.

    This is a memory-efficient alternative to :class:`HorizonHead` that avoids
    the computational overhead of multi-head cross-attention. Instead of
    cross-attending to the full backbone sequence, it:

    1. **Pools** the backbone sequence into a fixed-size context vector
    2. **Combines** learned horizon embeddings with the pooled context
    3. **Projects** through simple feedforward networks to per-step parameters

    Architecture::

        Backbone sequence  (batch, seq, d_model)
                │
                ▼
        ┌─────────────────────┐
        │  Sequence Pooling   │  mean/max → (batch, d_model)
        └────────┬────────────┘
                 │
                 ▼  (batch, d_model)
        ┌─────────────────────┐
        │  Learned horizon     │  (horizon_max, d_model) embeddings
        │  position embeddings │  + sinusoidal base
        └────────┬────────────┘
                 │
                 ▼  broadcast & concat
        ┌─────────────────────┐
        │  Feedforward Network │  (batch, horizon, 2*d_model) → (batch, horizon, d_model)
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  Per-step projection │  d_model → (mu_t, sigma_t)
        └────────┬────────────┘
                 │
                 ▼
        mu: (batch, horizon)   sigma: (batch, horizon)

    **Complexity Comparison**:

    - **HorizonHead**: O(horizon × seq_len × d_model) per attention layer
    - **SimpleHorizonHead**: O(seq_len × d_model) pooling + O(horizon × d_model²) MLP

    For typical values (seq_len=64, horizon=12, d_model=64), SimpleHorizonHead
    is ~10-20x more memory efficient.

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    horizon_max:
        Maximum prediction horizon supported.
    hidden_dim:
        Hidden dimension of the feedforward network. Default: latent_size * 2.
    pool_type:
        Pooling strategy over sequence dimension: "mean", "max", or "mean+max".
        Default: "mean".
    dropout:
        Dropout rate in feedforward network. Default: 0.1.
    """

    def __init__(
        self,
        latent_size: int,
        horizon_max: int = 48,
        hidden_dim: int | None = None,
        pool_type: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.horizon_max = horizon_max
        self.pool_type = pool_type
        hidden_dim = hidden_dim or latent_size * 2

        # Validate pool_type
        if pool_type not in ("mean", "max", "mean+max"):
            raise ValueError(f"pool_type must be 'mean', 'max', or 'mean+max', got {pool_type}")

        # Determine context dimension based on pooling
        context_dim = latent_size * 2 if pool_type == "mean+max" else latent_size

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

        # Feedforward network to combine context + horizon embeddings
        self.ff = nn.Sequential(
            nn.Linear(context_dim + latent_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_size),
            nn.Dropout(dropout),
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
        if horizon > self.horizon_max:
            logger.warning(
                "SimpleHorizonHead: requested horizon=%d exceeds horizon_max=%d; "
                "output will be clipped to %d steps. Increase horizon_max in "
                "the head config to support longer horizons.",
                horizon, self.horizon_max, self.horizon_max,
            )
        h = min(horizon, self.horizon_max)

        # Pool sequence to get context vector
        if self.pool_type == "mean":
            context = h_seq.mean(dim=1)  # (batch, d_model)
        elif self.pool_type == "max":
            context = h_seq.max(dim=1)[0]  # (batch, d_model)
        else:  # mean+max
            mean_pool = h_seq.mean(dim=1)
            max_pool = h_seq.max(dim=1)[0]
            context = torch.cat([mean_pool, max_pool], dim=-1)  # (batch, 2*d_model)

        # Build horizon queries: learned embedding + sinusoidal position
        horizon_emb = (self.horizon_queries[:h] + self.pe[:h]).unsqueeze(0).expand(batch, -1, -1)
        # (batch, horizon, d_model)

        # Broadcast context and concatenate with horizon embeddings
        context_expanded = context.unsqueeze(1).expand(-1, h, -1)  # (batch, horizon, context_dim)
        combined = torch.cat([context_expanded, horizon_emb], dim=-1)  # (batch, horizon, context_dim + d_model)

        # Feedforward network
        features = self.ff(combined)  # (batch, horizon, d_model)

        # Project to per-step parameters
        mu = self.mu_proj(features).squeeze(-1)                           # (batch, horizon)
        sigma = F.softplus(self.sigma_proj(features)).squeeze(-1) + 1e-6  # (batch, horizon)
        return mu, sigma


# ---------------------------------------------------------------------------
# CLT Horizon Head (spectral basis expansion for per-step parameters)
# ---------------------------------------------------------------------------


class CLTHorizonHead(HeadBase):
    """CLT-inspired horizon head using spectral basis functions.

    Produces per-step ``(mu_t, sigma_t)`` via a compact, **deterministic**
    spectral decomposition.  The head predicts a global drift/volatility
    centre plus sinusoidal modulation coefficients that shape smooth
    temporal variation across the horizon.

    Central Limit Theorem motivation: the per-step parameters are
    modelled as deviations around population-level means ``mu_0`` and
    ``sigma_0``.  The basis coefficients control how much each step is
    allowed to deviate.  Because only low-frequency basis functions are
    used, the resulting parameter trajectories are inherently smooth —
    capturing regime drift and volatility clustering without overfitting
    to per-step noise.

    Compared to the other horizon heads:

    - **GBMHead**: constant ``(mu, sigma)`` — no temporal variation.
    - **SimpleHorizonHead**: per-step MLP — O(horizon × d_model²) compute.
    - **HorizonHead**: cross-attention — O(horizon × seq_len × d_model).
    - **CLTHorizonHead**: spectral basis — O(n_basis × horizon) matmul.
      Cheapest deterministic head that can express smooth dynamics.

    Architecture::

        h_t  (batch, d_model)   ← last-step backbone embedding
              │
              ▼
        ┌────────────┐
        │ LayerNorm  │
        └─────┬──────┘
              ▼
        ┌────────────┐
        │  MLP       │  (d_model → hidden → hidden)
        └─────┬──────┘
              │
         ┌────┴─────────────────────────┐
         ▼                              ▼
       μ₀, σ₀  (global centres)     α_k, β_k, γ_k, δ_k  (basis coefficients)
              │                          │
              │    ┌─────────────────────┘
              │    │  sin/cos basis functions  B(t)
              │    ▼
              ▼  mu_t  = μ₀ + [α, β] · B(t)
                 σ_t   = softplus(σ₀ + [γ, δ] · B(t))
              │
              ▼
        mu: (batch, horizon)   sigma: (batch, horizon)

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    hidden:
        Hidden dimension of the shared MLP.  Default: 64.
    n_basis:
        Number of Fourier frequency components (each contributes a sin
        *and* a cos term, so total basis functions = 2 × n_basis).
        Default: 4.  Higher values allow sharper temporal changes but
        increase parameter count and risk of overfitting.
    """

    def __init__(
        self,
        latent_size: int,
        hidden: int = 64,
        n_basis: int = 4,
    ) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.norm = nn.LayerNorm(latent_size)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # Global centres
        self.mu_center = nn.Linear(hidden, 1)
        self.sigma_center = nn.Linear(hidden, 1)
        # Spectral modulation coefficients (sin + cos for each frequency)
        self.mu_basis_proj = nn.Linear(hidden, n_basis * 2)
        self.sigma_basis_proj = nn.Linear(hidden, n_basis * 2)

    def _build_basis(
        self, horizon: int, device: torch.device,
    ) -> torch.Tensor:
        """Construct sinusoidal basis matrix.

        Returns
        -------
        basis : (2 * n_basis, horizon)
        """
        t = torch.linspace(0.0, 1.0, horizon, device=device)  # (horizon,)
        freqs = torch.arange(1, self.n_basis + 1, device=device, dtype=t.dtype)
        # (n_basis, horizon) each
        phase = 2.0 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=0)

    def forward(
        self,
        h_t: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_t : (batch, d_model) — last-step backbone embedding
        horizon : prediction length

        Returns
        -------
        mu_seq : (batch, horizon) — per-step drift
        sigma_seq : (batch, horizon) — per-step volatility (positive)
        """
        h_t = self.norm(h_t)
        features = self.net(h_t)  # (batch, hidden)

        # Global centres
        mu_0 = self.mu_center(features)            # (batch, 1)
        sigma_0 = self.sigma_center(features)      # (batch, 1)

        # Basis coefficients
        mu_coeffs = self.mu_basis_proj(features)       # (batch, 2*n_basis)
        sigma_coeffs = self.sigma_basis_proj(features)  # (batch, 2*n_basis)

        # Basis matrix — (2*n_basis, horizon)
        basis = self._build_basis(horizon, h_t.device)

        # Per-step parameters via spectral expansion
        mu_seq = mu_0 + mu_coeffs @ basis              # (batch, horizon)
        sigma_seq = F.softplus(
            sigma_0 + sigma_coeffs @ basis
        ) + 1e-6                                        # (batch, horizon)

        return mu_seq, sigma_seq


# ---------------------------------------------------------------------------
# Student-t Brownian Walk Head (stochastic parameter paths + fat tails)
# ---------------------------------------------------------------------------


class StudentTHorizonHead(HeadBase):
    """Probabilistic horizon head with Brownian-walk parameter evolution.

    Predicts **meta-parameters** (location and log-scale for drift,
    volatility, and degrees-of-freedom), then generates per-step
    parameter trajectories via cumulative-sum Brownian noise.  The
    resulting smooth, autocorrelated parameter paths capture volatility
    clustering and regime drift naturally.

    This is a *double-stochastic* model:

    1. **Parameter paths** are sampled via Brownian walk in the head.
    2. **Price paths** are sampled with Student-*t* innovations in the
       simulator.

    The ``1 / sqrt(horizon)`` noise scaling keeps total parameter
    variance bounded regardless of horizon length.  Clamped log-scales
    prevent explosion / collapse of the spread meta-parameters.

    Architecture::

        h_t  (batch, d_model)
              │
              ▼
        ┌────────────┐
        │ LayerNorm  │
        └─────┬──────┘
              ▼
        ┌────────────┐
        │  MLP       │  (d_model → hidden → hidden)
        └─────┬──────┘
              │
              ▼  Linear → 6 meta-params
        ┌──────────────────────────────────────────┐
        │ mu_mu, mu_logstd      (drift dist.)      │
        │ sig_mu, sig_logstd    (volatility dist.)  │
        │ nu_mu, nu_logstd      (d.o.f. dist.)      │
        └──────────────┬───────────────────────────┘
                       │
                       ▼  Brownian walk sampling (× horizon)
        ┌──────────────────────────────────────────┐
        │ eps = randn().cumsum() / sqrt(H)         │
        │ mu_t  = mu_mu + exp(mu_logstd) · eps     │
        │ σ_t   = softplus(sig_mu + ...) + 1e-6    │
        │ ν_t   = softplus(nu_mu + ...) + 2.0      │
        └──────────────────────────────────────────┘
              │
              ▼
        mu_seq, sigma_seq, nu_seq   (batch, horizon)

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    hidden:
        Hidden dimension of the shared MLP.  Default: 64.
    """

    def __init__(self, latent_size: int, hidden: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_size)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # 6 meta-parameters: [mu_mu, mu_logstd, sig_mu, sig_logstd, nu_mu, nu_logstd]
        self.param_proj = nn.Linear(hidden, 6)

    def forward(
        self,
        h_t: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_t : (batch, d_model) — last-step backbone embedding
        horizon : prediction length

        Returns
        -------
        mu_seq : (batch, horizon) — per-step drift (Brownian walk)
        sigma_seq : (batch, horizon) — per-step volatility (positive)
        nu_seq : (batch, horizon) — per-step degrees-of-freedom (>2)
        """
        batch = h_t.shape[0]
        device = h_t.device

        h_t = self.norm(h_t)
        feat = self.net(h_t)
        params = self.param_proj(feat)  # (batch, 6)

        # --- extract meta-parameters ---
        mu_mu = params[:, 0:1]
        # Clamp logstd then clamp exp() to prevent noise amplification at
        # long horizons.  exp(0) = 1.0 caps the Brownian noise multiplier
        # so that sigma stays bounded over 288+ steps.
        mu_std = params[:, 1:2].clamp(-5, 0).exp()       # ≤ 1.0

        sig_mu = params[:, 2:3]
        sig_std = params[:, 3:4].clamp(-5, 0).exp()      # ≤ 1.0

        # nu: sigmoid → [2.1, 30.1]
        nu_mu = torch.sigmoid(params[:, 4:5]) * 28.0 + 2.1
        nu_std = params[:, 5:6].clamp(-5, 0).exp()       # ≤ 1.0

        # --- Brownian walk parameter paths ---
        # Scale by 1/sqrt(horizon) to keep total variance constant
        step_scale = 1.0 / math.sqrt(max(horizon, 1))

        # Drift path
        eps_mu = torch.randn(batch, horizon, device=device).cumsum(dim=-1) * step_scale
        mu_seq = (mu_mu + mu_std * eps_mu).clamp(-4.0, 4.0)

        # Volatility path (positive via softplus, clamped pre-activation)
        eps_sig = torch.randn(batch, horizon, device=device).cumsum(dim=-1) * step_scale
        sigma_seq = F.softplus(
            (sig_mu + sig_std * eps_sig).clamp(-4.0, 4.0)
        ) + 1e-6

        # Degrees-of-freedom path (>2 for finite variance)
        eps_nu = torch.randn(batch, horizon, device=device).cumsum(dim=-1) * step_scale
        nu_seq = F.softplus(
            (nu_mu + nu_std * eps_nu).clamp(-4.0, 40.0)
        ) + 2.0

        return mu_seq, sigma_seq, nu_seq


# ---------------------------------------------------------------------------
# Probabilistic Horizon Head (unified spectral / brownian / hybrid / hybrid_ou)
# ---------------------------------------------------------------------------


class ProbabilisticHorizonHead(HeadBase):
    """Unified head supporting multiple forecasting strategies via a mode switch.

    Returns per-step ``(mu_t, sigma_t, nu_t)`` trajectories using one of:

    1. **spectral** — Deterministic sinusoidal basis expansion.  The
       parameter trajectory is a smooth, learnable curve built from low-frequency
       Fourier modes.  Produces the most stable gradients and is best for
       short horizons with predictable cyclicality.

    2. **brownian** — Pure stochastic random-walk evolution.  Parameters
       evolve via cumulative Brownian noise, modelling regime shifts and
       chaotic dynamics.  Meta-stds are clamped and paths are bounded to
       prevent the ``-0.5 * sigma^2`` drift term from collapsing prices
       over long horizons.

    3. **hybrid** — Spectral backbone with gated Brownian perturbations.
       The spectral basis captures the global (macro) trend while Brownian
       residuals add local, high-frequency jitter.  A learnable mixing gate
       (initialized near zero) lets the model gradually introduce stochastic
       variation as training stabilises.  Brownian centers are omitted in
       this mode to prevent redundancy with spectral DC components.

    4. **hybrid_ou** — Spectral backbone with mean-reverting Ornstein-
       Uhlenbeck perturbations.  Like ``hybrid`` but the stochastic
       component is an AR(1) process with a learned reversion coefficient
       ``phi``.  The mean-reversion prevents parameter drift at long
       horizons (H=288+), making this the most stable hybrid mode.

    All modes return three parameters per step—drift ``mu``, volatility
    ``sigma``, and Student-*t* degrees-of-freedom ``nu``—and are designed
    for use with :func:`simulate_t_horizon_paths`.

    Architecture::

        h_t  (batch, d_model)
              │
              ▼
        ┌────────────┐
        │ LayerNorm  │
        └─────┬──────┘
              ▼
        ┌────────────┐
        │  Shared MLP│  (d_model → hidden → hidden)
        └─────┬──────┘
              │
         ┌────┴──────────────────────────────────────────┐
         │ (spectral / hybrid*)                           │ (brownian / hybrid*)
         ▼                                                ▼
      basis_weights → sin/cos expansion            param_proj → meta-params
      → spectral_path  (batch, 3, H)              → stochastic_path (batch, 3, H)
         │                                                │
         │                                          [clamped, gated]
         └──────────────┬─────────────────────────────────┘
                        ▼  element-wise sum
                   total_path  (batch, 3, H)
                        │
                        ▼
              mu, sigma (softplus), nu (sigmoid→[2.1, 30.1])

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    hidden_dim:
        Hidden dimension of the shared MLP.  Default: 64.
    mode:
        One of ``'spectral'``, ``'brownian'``, ``'hybrid'``, or
        ``'hybrid_ou'``.  Default: ``'hybrid'``.
    n_basis:
        Number of Fourier frequency components for spectral / hybrid
        modes.  Each contributes sin *and* cos terms so total basis
        functions = ``2 * n_basis``.  Default: 8.
    """

    _VALID_MODES = {"spectral", "brownian", "hybrid", "hybrid_ou"}
    _SPECTRAL_MODES = {"spectral", "hybrid", "hybrid_ou"}
    _STOCHASTIC_MODES = {"brownian", "hybrid", "hybrid_ou"}
    _HYBRID_MODES = {"hybrid", "hybrid_ou"}

    def __init__(
        self,
        latent_size: int,
        hidden_dim: int = 64,
        mode: str = "hybrid",
        n_basis: int = 8,
    ) -> None:
        super().__init__()
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self._VALID_MODES)}, got {mode!r}"
            )
        self.mode = mode
        self.n_basis = n_basis
        self.norm = nn.LayerNorm(latent_size)

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if mode in self._SPECTRAL_MODES:
            # Learnable weights for 3 params (mu, sig, nu) × 2 (sin/cos) × n_basis
            self.basis_weights = nn.Linear(hidden_dim, 3 * 2 * n_basis)

        if mode in self._STOCHASTIC_MODES:
            # Meta-stats: [mu_mu, mu_std, sig_mu, sig_std, nu_mu, nu_std]
            self.param_proj = nn.Linear(hidden_dim, 6)
            # Zero-init bias to start with centered, low-variance paths
            nn.init.zeros_(self.param_proj.bias)

        if mode in self._HYBRID_MODES:
            # Learnable mixing gate, initialized small so training starts ~spectral.
            # sigmoid(-2.0) ≈ 0.12 — brownian contributes ~12% at init.
            self.mix_logit = nn.Parameter(torch.tensor(-2.0))

        if mode == "hybrid_ou":
            # OU reversion coefficient phi = sigmoid(reversion_logit).
            # sigmoid(3.0) ≈ 0.95 — moderate mean-reversion at init.
            self.reversion_logit = nn.Parameter(torch.tensor(3.0))

    def _get_spectral_path(
        self, feat: torch.Tensor, horizon: int,
    ) -> torch.Tensor:
        """Build smooth parameter trajectories from sinusoidal basis.

        Uses a matmul of learned coefficients against a sin/cos basis matrix,
        following the same pattern as :class:`CLTHorizonHead`.

        Returns
        -------
        spectral_path : (batch, 3, horizon)
        """
        batch_size = feat.shape[0]
        device = feat.device

        # Basis matrix: (2*K, H)
        t = torch.linspace(0.0, 1.0, horizon, device=device)
        freqs = torch.arange(1, self.n_basis + 1, device=device, dtype=feat.dtype)
        phase = 2.0 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0)  # (K, H)
        basis = torch.cat([torch.sin(phase), torch.cos(phase)], dim=0)  # (2*K, H)

        # Coefficients: (B, 3, 2*K)
        coeffs = self.basis_weights(feat).view(batch_size, 3, 2 * self.n_basis)

        # Spectral path: (B, 3, 2*K) @ (2*K, H) → (B, 3, H)
        return coeffs @ basis

    def _get_brownian_noise(
        self, batch_size: int, horizon: int, device: torch.device,
    ) -> torch.Tensor:
        """Cumulative-sum Brownian noise scaled by ``1 / sqrt(H)``.

        Returns
        -------
        noise : (batch, 3, horizon) with endpoint std ≈ 1.0
        """
        step_scale = 1.0 / math.sqrt(max(horizon, 1))
        return (
            torch.randn(batch_size, 3, horizon, device=device).cumsum(dim=-1)
            * step_scale
        )

    def _get_ou_noise(
        self, batch_size: int, horizon: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Ornstein-Uhlenbeck (AR(1)) noise with learned reversion speed.

        Unlike pure Brownian, the OU process pulls noise back toward zero,
        preventing parameter drift at long horizons.  Uses an explicit
        scan for numerical stability across all ``phi`` values.

        Returns
        -------
        ou_noise : (batch, 3, horizon)
        """
        phi = torch.sigmoid(self.reversion_logit)
        step_scale = 1.0 / math.sqrt(max(horizon, 1))
        eps = torch.randn(batch_size, 3, horizon, device=device) * step_scale

        # AR(1) scan: x_t = phi * x_{t-1} + eps_t
        # Unrolled to avoid O(H^2) memory from vectorized alternatives.
        # H=288 × element-wise ops is fast on GPU.
        x = torch.zeros(batch_size, 3, 1, device=device, dtype=dtype)
        steps = []
        for t in range(horizon):
            x = phi * x + eps[:, :, t : t + 1]
            steps.append(x)
        return torch.cat(steps, dim=-1)  # (B, 3, H)

    def forward(
        self,
        h_t: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_t : (batch, d_model) — last-step backbone embedding
        horizon : prediction length

        Returns
        -------
        mu_seq : (batch, horizon) — per-step drift
        sigma_seq : (batch, horizon) — per-step volatility (positive)
        nu_seq : (batch, horizon) — per-step degrees-of-freedom (in [2.1, 30.1])
        """
        h_t = self.norm(h_t)
        feat = self.net(h_t)
        batch_size = h_t.shape[0]

        # 1. Spectral component
        if self.mode in self._SPECTRAL_MODES:
            spec_path = self._get_spectral_path(feat, horizon)
        else:
            spec_path = torch.zeros(batch_size, 3, horizon, device=h_t.device)

        # 2. Stochastic component
        if self.mode in self._STOCHASTIC_MODES:
            p = self.param_proj(feat)  # (batch, 6)

            # Build noise: Brownian or OU depending on mode
            if self.mode == "hybrid_ou":
                noise = self._get_ou_noise(batch_size, horizon, h_t.device, feat.dtype)
            else:
                noise = self._get_brownian_noise(batch_size, horizon, h_t.device)

            # Clamp meta-stds to prevent sigma blow-up at long horizons.
            # softplus ∈ (0, ∞) → clamp to (0, 1.0] keeps paths bounded.
            stds = F.softplus(p[:, [1, 3, 5]]).clamp(max=1.0).unsqueeze(-1)

            if self.mode == "brownian":
                # Pure brownian: centers provide the DC level
                centers = p[:, [0, 2, 4]].unsqueeze(-1)
                stoch_path = (centers + stds * noise).clamp(-4.0, 4.0)
            else:
                # Hybrid modes: spectral handles DC, stochastic is perturbation only.
                # Omitting centers prevents double-parameterization instability.
                stoch_path = (stds * noise).clamp(-4.0, 4.0)
        else:
            stoch_path = torch.zeros(batch_size, 3, horizon, device=h_t.device)

        # 3. Combine paths
        if self.mode in self._HYBRID_MODES:
            mix = torch.sigmoid(self.mix_logit)
            total_path = spec_path + mix * stoch_path
        elif self.mode == "brownian":
            total_path = stoch_path
        else:
            total_path = spec_path

        # 4. Final parameter mapping
        mu_seq = total_path[:, 0, :]
        sigma_seq = F.softplus(total_path[:, 1, :]) + 1e-6
        # Map nu to [2.1, 30.1] range
        nu_seq = torch.sigmoid(total_path[:, 2, :]) * 28.0 + 2.1

        return mu_seq, sigma_seq, nu_seq


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
