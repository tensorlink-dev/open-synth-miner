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
# CLT Horizon Head (sample per-step params from learned hyperparameters)
# ---------------------------------------------------------------------------


class CLTHorizonHead(HeadBase):
    """CLT-inspired horizon head: samples per-step params from learned hyperparameters.

    Uses the Central Limit Theorem intuition that per-step drift and volatility
    are drawn from underlying distributions.  The head predicts four
    hyperparameters — the mean and spread of the per-step drift distribution
    and the location and spread of the per-step log-volatility distribution —
    then samples per-step parameters via the reparameterization trick.

    This sits between:

    - **GBMHead**: constant ``(mu, sigma)`` across all steps (too rigid)
    - **HorizonHead**: fully independent ``(mu_t, sigma_t)`` per step
      (expensive, risk of overfitting)

    The double stochasticity (parameter sampling *then* path sampling) yields a
    mixture-of-Gaussians distribution for log-returns at each step, producing
    heavier tails that better match financial data.

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
         ┌────┴────────────┬──────────────┬──────────────┐
         ▼                 ▼              ▼              ▼
       μ_μ  (drift mean)  σ_μ (drift    μ_logσ (vol    σ_logσ (vol
                           spread)       location)      spread)
              │
              ▼  reparameterized sampling  (× horizon)
        ┌─────────────────────────────────────────┐
        │ mu_t  = μ_μ + σ_μ · ε₁                 │
        │ σ_t   = softplus(μ_logσ + σ_logσ · ε₂) │
        └─────────────────────────────────────────┘
              │
              ▼
        mu: (batch, horizon)   sigma: (batch, horizon)

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
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # Hyperparameter projections — 4 scalars per sample
        self.mu_mu_proj = nn.Linear(hidden, 1)          # mean of per-step drifts
        self.mu_spread_proj = nn.Linear(hidden, 1)      # spread of per-step drifts
        self.sigma_loc_proj = nn.Linear(hidden, 1)      # location of per-step log-vol
        self.sigma_spread_proj = nn.Linear(hidden, 1)   # spread of per-step log-vol

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
        mu_seq : (batch, horizon) — per-step drift (sampled)
        sigma_seq : (batch, horizon) — per-step volatility (sampled, positive)
        """
        batch = h_t.shape[0]
        device = h_t.device

        h_t = self.norm(h_t)
        features = self.net(h_t)  # (batch, hidden)

        # --- predict hyperparameters ---
        mu_mu = self.mu_mu_proj(features).squeeze(-1)                              # (batch,)
        mu_spread = F.softplus(self.mu_spread_proj(features).squeeze(-1)) + 1e-6   # (batch,)
        sigma_loc = self.sigma_loc_proj(features).squeeze(-1)                      # (batch,)
        sigma_spread = F.softplus(self.sigma_spread_proj(features).squeeze(-1)) + 1e-6  # (batch,)

        # --- reparameterized sampling of per-step drift ---
        # mu_t ~ N(mu_mu, mu_spread^2)
        eps_mu = torch.randn(batch, horizon, device=device)
        mu_seq = mu_mu.unsqueeze(-1) + mu_spread.unsqueeze(-1) * eps_mu  # (batch, horizon)

        # --- reparameterized sampling of per-step log-volatility ---
        # log_sigma_t ~ N(sigma_loc, sigma_spread^2), then softplus for positivity
        eps_sigma = torch.randn(batch, horizon, device=device)
        log_sigma_seq = sigma_loc.unsqueeze(-1) + sigma_spread.unsqueeze(-1) * eps_sigma
        sigma_seq = F.softplus(log_sigma_seq) + 1e-6  # (batch, horizon)

        return mu_seq, sigma_seq


# ---------------------------------------------------------------------------
# Student-t CLT Horizon Head (fat-tailed per-step parameter sampling)
# ---------------------------------------------------------------------------


class StudentTHorizonHead(HeadBase):
    """CLT horizon head with Student's *t*-distributed parameter sampling.

    Extends the :class:`CLTHorizonHead` idea by replacing Gaussian
    parameter sampling with a location-scale Student's *t*-distribution.
    This directly models fat tails: when the per-step variance is itself
    stochastic (Inverse-Gamma), the marginal return distribution is
    exactly Student's *t*.

    The head predicts **six** hyperparameters per sample:

    ========  =======================  ========================================
    Symbol    Meaning                  Constraint
    ========  =======================  ========================================
    μ_μ       mean of per-step drifts  unconstrained
    σ_μ       spread of drifts         softplus → positive
    μ_logσ    location of log-vol      unconstrained
    σ_logσ    spread of log-vol        softplus → positive
    μ_ν       centre of d.o.f.         sigmoid → [2.1, 50]
    σ_ν       spread of d.o.f.         softplus → positive
    ========  =======================  ========================================

    Per-step parameters are sampled via reparameterized *t*-noise::

        Z ~ N(0,1),  V ~ Chi²(ν),  t_sample = Z / √(V/ν)
        mu_t  = μ_μ  + σ_μ  · t_sample_1
        σ_t   = softplus(μ_logσ + σ_logσ · t_sample_2)

    Additionally, a per-step ``nu_t`` is returned so the **path simulator**
    can draw innovations from ``StudentT(nu_t)`` instead of ``N(0,1)``,
    giving fat-tailed *price* dynamics as well.

    Architecture::

        h_t  (batch, d_model)
              │
              ▼
        ┌────────────┐
        │ LayerNorm  │
        └─────┬──────┘
              ▼
        ┌────────────┐
        │    MLP     │
        └─────┬──────┘
              │
         ┌────┴──────┬──────────┬──────────┬──────────┬──────────┐
         ▼           ▼          ▼          ▼          ▼          ▼
       μ_μ         σ_μ       μ_logσ     σ_logσ      μ_ν       σ_ν
              │
              ▼  reparameterized t-sampling  (× horizon)
        mu_seq, sigma_seq, nu_seq

    Parameters
    ----------
    latent_size:
        Must match the backbone ``d_model``.
    hidden:
        Hidden dimension of the shared MLP.  Default: 64.
    nu_min:
        Minimum degrees-of-freedom (must be >2 for finite variance).
        Default: 2.1.
    nu_max:
        Maximum degrees-of-freedom.  As ν → ∞ the *t* → Gaussian, so
        this caps how "normal" the model can choose to be.  Default: 50.
    """

    def __init__(
        self,
        latent_size: int,
        hidden: int = 64,
        nu_min: float = 2.1,
        nu_max: float = 50.0,
    ) -> None:
        super().__init__()
        self.nu_min = nu_min
        self.nu_range = nu_max - nu_min

        self.norm = nn.LayerNorm(latent_size)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # Drift hyper-parameters
        self.mu_mu_proj = nn.Linear(hidden, 1)
        self.mu_spread_proj = nn.Linear(hidden, 1)
        # Volatility hyper-parameters
        self.sigma_loc_proj = nn.Linear(hidden, 1)
        self.sigma_spread_proj = nn.Linear(hidden, 1)
        # Degrees-of-freedom hyper-parameters
        self.nu_loc_proj = nn.Linear(hidden, 1)
        self.nu_spread_proj = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _reparam_t_sample(
        loc: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
        shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """Reparameterized location-scale Student-*t* sample.

        ``t = loc + scale * (Z / sqrt(V / nu))``
        where ``Z ~ N(0,1)`` and ``V ~ Chi²(nu)``.

        Using the Gamma parameterisation of Chi² for differentiability:
        ``Chi²(nu) = 2 · Gamma(nu/2, 1)`` → ``V/nu = Gamma(nu/2, 1) / (nu/2)``.
        """
        z = torch.randn(shape, device=device)
        # Gamma(alpha=nu/2, beta=1) — torch uses (concentration, rate)
        alpha = (nu / 2.0).unsqueeze(-1).expand(shape).clamp(min=0.5)
        v = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).rsample()
        # v / (nu/2) ~ Chi²(nu) / nu
        t_noise = z / (v / alpha + 1e-8).sqrt()
        return loc.unsqueeze(-1) + scale.unsqueeze(-1) * t_noise

    # ------------------------------------------------------------------
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
        nu_seq : (batch, horizon) — per-step degrees-of-freedom (>2)
        """
        batch = h_t.shape[0]
        device = h_t.device

        h_t = self.norm(h_t)
        features = self.net(h_t)  # (batch, hidden)

        # --- drift hyper-parameters ---
        mu_mu = self.mu_mu_proj(features).squeeze(-1)                              # (batch,)
        mu_spread = F.softplus(self.mu_spread_proj(features).squeeze(-1)) + 1e-6

        # --- volatility hyper-parameters ---
        sigma_loc = self.sigma_loc_proj(features).squeeze(-1)
        sigma_spread = F.softplus(self.sigma_spread_proj(features).squeeze(-1)) + 1e-6

        # --- degrees-of-freedom hyper-parameters ---
        # sigmoid → [0, 1] then scale to [nu_min, nu_max]
        nu_loc = (
            torch.sigmoid(self.nu_loc_proj(features).squeeze(-1))
            * self.nu_range
            + self.nu_min
        )  # (batch,)
        nu_spread = F.softplus(self.nu_spread_proj(features).squeeze(-1)) + 1e-6

        sample_shape = (batch, horizon)

        # --- sample per-step drift from t(mu_mu, mu_spread, nu_loc) ---
        mu_seq = self._reparam_t_sample(
            mu_mu, mu_spread, nu_loc, sample_shape, device,
        )  # (batch, horizon)

        # --- sample per-step log-volatility from t(sigma_loc, sigma_spread, nu_loc) ---
        log_sigma_seq = self._reparam_t_sample(
            sigma_loc, sigma_spread, nu_loc, sample_shape, device,
        )
        sigma_seq = F.softplus(log_sigma_seq) + 1e-6  # (batch, horizon)

        # --- sample per-step nu ---
        eps_nu = torch.randn(batch, horizon, device=device)
        nu_seq = F.softplus(
            nu_loc.unsqueeze(-1) + nu_spread.unsqueeze(-1) * eps_nu
        ) + self.nu_min  # (batch, horizon), always > nu_min

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
