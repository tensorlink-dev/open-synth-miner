# Philosophy & Design Decisions

This document explains *why* Open Synth Miner is built the way it is — the beliefs, trade-offs, and hard-won lessons that shaped every major decision. If `ARCHITECTURE.md` is the *what*, this is the *why*.

---

## Table of Contents

- [The Core Problem](#the-core-problem)
- [Guiding Beliefs](#guiding-beliefs)
  - [1. Distributions Beat Point Estimates](#1-distributions-beat-point-estimates)
  - [2. Configuration Is Architecture](#2-configuration-is-architecture)
  - [3. Build in Public](#3-build-in-public)
  - [4. Composability Over Monoliths](#4-composability-over-monoliths)
  - [5. Research Velocity Over Production Polish](#5-research-velocity-over-production-polish)
- [Key Design Decisions](#key-design-decisions)
  - [Why Hydra for Configuration](#why-hydra-for-configuration)
  - [Why a Registry with Auto-Discovery](#why-a-registry-with-auto-discovery)
  - [Why Shape Contracts Instead of Runtime Checks](#why-shape-contracts-instead-of-runtime-checks)
  - [Why Multiple Heads with Different Return Types](#why-multiple-heads-with-different-return-types)
  - [Why Leak-Safe Data Loading Matters](#why-leak-safe-data-loading-matters)
  - [Why CRPS Over Traditional Loss Functions](#why-crps-over-traditional-loss-functions)
  - [Why 1,000 Paths](#why-1000-paths)
  - [Why Numerical Stability Is First-Class](#why-numerical-stability-is-first-class)
  - [Why Champion vs. Challenger](#why-champion-vs-challenger)
  - [Why W&B + Hugging Face Hub Together](#why-wb--hugging-face-hub-together)
- [Architectural Trade-offs We Made](#architectural-trade-offs-we-made)
  - [Flexibility vs. Type Safety](#flexibility-vs-type-safety)
  - [Declarative Configs vs. Code Readability](#declarative-configs-vs-code-readability)
  - [Graceful Degradation vs. Fail-Fast](#graceful-degradation-vs-fail-fast)
  - [Research Notebooks as First-Class Citizens](#research-notebooks-as-first-class-citizens)
- [What We Tried and Walked Back](#what-we-tried-and-walked-back)
- [Principles for Contributors](#principles-for-contributors)

---

## The Core Problem

Bittensor SN50 (Synth) asks miners to produce *probabilistic price forecasts* — not a single predicted price, but a distribution of plausible future paths. This is fundamentally different from the typical time series forecasting task where you predict a point estimate and measure MAE or RMSE.

The scoring mechanism rewards miners whose distributions *calibrate well against reality*. A model that produces a tight fan of paths and happens to be right scores worse than one that honestly expresses uncertainty and covers the realized outcome. This means:

- We need to generate many stochastic paths (Monte Carlo simulation), not just one prediction.
- We need to evaluate with proper scoring rules (CRPS), not point-estimate metrics.
- We need to iterate on architectures quickly, because no single model family dominates probabilistic forecasting.

Everything in Open Synth Miner flows from these three requirements.

---

## Guiding Beliefs

### 1. Distributions Beat Point Estimates

Traditional forecasting asks "what will the price be?" We ask "what does the space of plausible futures look like?"

This belief drives the entire model architecture. Every model in this codebase produces *parameters for a stochastic process* (drift, volatility, bridge targets, mixture weights), never a point prediction. The simulation layer then generates 1,000 differentiable sample paths from those parameters.

This is why we have `simulate_gbm_paths`, `simulate_horizon_paths`, `simulate_bridge_paths`, and `simulate_mixture_paths` as separate functions — each implements a different stochastic process, because the right generative model for price uncertainty is itself an open research question.

### 2. Configuration Is Architecture

A key insight that shaped the codebase: in a research context, the model architecture *is* the experiment. Changing a block, swapping a head, or adjusting a hyperparameter should not require touching Python code.

This is why Hydra drives everything. A model is fully specified by a YAML file:

```yaml
model:
  backbone:
    blocks:
      - _target_: src.models.registry.TransformerBlock
        d_model: 32
        nhead: 4
      - _target_: src.models.registry.LSTMBlock
        d_model: 32
  head:
    _target_: src.models.heads.GBMHead
    latent_size: 32
```

The `_target_` pattern means Hydra instantiates the actual Python objects. The YAML *is* the architecture. This makes experiment tracking trivial — log the resolved config and you've captured everything needed to reproduce the model.

### 3. Build in Public

We believe research should be transparent. Every training run:

1. Logs metrics to Weights & Biases (CRPS, log-likelihood, variance spread).
2. Uploads the checkpoint to Hugging Face Hub under a taxonomy-structured path.
3. Generates a model card linking back to the W&B run.
4. Produces a shareable report suitable for posting publicly.

This isn't just a nice-to-have — it's a core architectural commitment. `HubManager` exists as a dedicated module because artifact publishing is as important as training itself. When someone asks "how did you get that CRPS score?", the answer is a link, not a shrug.

### 4. Composability Over Monoliths

Time series forecasting has no single winning architecture. Transformers, LSTMs, DLinear, TimesNet, FEDformer — each has strengths in different regimes. Rather than pick one and optimize it, we built a system where architectures are assembled from interchangeable parts.

The `HybridBackbone` stitches blocks from a recipe. Each block has a simple contract: accept `(batch, seq, d_model)`, return `(batch, seq, d_model)`. You can stack a TransformerBlock on top of an LSTMBlock on top of a ResConvBlock. You can run blocks in parallel via `ParallelFusion`. You can auto-insert LayerNorm between blocks with a single config flag.

The registry makes this work. Blocks register themselves with decorators. A new block file dropped into `src/models/components/` is automatically discovered at startup — no imports to update, no factory functions to modify.

### 5. Research Velocity Over Production Polish

This is a research framework, not a production service. When faced with a trade-off between "move faster" and "bulletproof everything", we lean toward speed — but with guardrails.

We chose to:
- Validate shapes at construction time (smoke test) rather than sprinkle assertions everywhere.
- Use `@torch.no_grad()` for inference rather than build a separate inference graph.
- Keep the training loop simple (single `train_step` + `evaluate_and_log`) rather than add callbacks, hooks, and lifecycle managers.
- Trust framework guarantees (PyTorch, Hydra, NumPy) rather than defensively re-validate their outputs.

The exception: data integrity. We *are* defensive about data loading, because look-ahead bias in time series is silent and catastrophic.

---

## Key Design Decisions

### Why Hydra for Configuration

**Alternatives considered:** argparse, plain YAML, dataclass configs, gin-config.

**Why Hydra won:**
- **Composition**: We can assemble a full experiment from independent config fragments (`model/hybrid_v2.yaml` + `data/ohlcv_loader.yaml`). No other system does this as cleanly.
- **`_target_` instantiation**: Objects are constructed directly from config. No factory boilerplate mapping string names to classes.
- **Override syntax**: `python main.py training.batch_size=8 model.backbone.d_model=64` — command-line overrides without touching files.
- **Reproducibility**: The resolved config captures every parameter. Log it once, reproduce forever.

The cost is a learning curve and occasional magic (Hydra silently changes working directories, for example). We accept this cost because the alternative — manually wiring up every experiment variant — doesn't scale.

### Why a Registry with Auto-Discovery

**The problem:** As we add more blocks (Transformer, LSTM, RNN, GRU, DLinear, TimesNet, PatchTST, FEDformer, etc.), maintaining a central import file becomes painful. Every new block needs an import, a factory entry, and documentation updates.

**The solution:** Decorator-based registration + recursive directory scanning.

```python
@registry.register_block("transformerblock")
class TransformerBlock(nn.Module):
    ...
```

At startup, `discover_components("src/models/components")` walks the directory tree and imports every `.py` file. Import triggers the decorator, which registers the class. Hydra configs reference registered blocks by their `_target_` path.

**Why this matters:** A researcher can add a new block by creating a single file. No other files need changing. The block is immediately available in YAML configs. This removes friction from the most common research task: "try a different architecture."

### Why Shape Contracts Instead of Runtime Checks

**The belief:** Shape errors are the most common bug in deep learning code. Catching them early saves hours of debugging NaN losses.

**The approach:** `HybridBackbone.validate_shapes()` runs at construction time. It pushes a dummy tensor through every block and checks that the shape contract (`batch, seq, d_model`) is preserved. If a block breaks the contract, you get a clear error message *before any data loads*.

**Why not sprinkle `assert` everywhere?** Because it's noisy, it slows down the hot path, and it catches errors too late — after you've already loaded data and started training. Construction-time validation is a one-time cost that prevents an entire class of bugs.

Blocks that intentionally change shapes (like `FlexiblePatchEmbed`) declare this via metadata (`preserves_seq_len=False`), and the validator adjusts expectations accordingly.

### Why Multiple Heads with Different Return Types

**The tension:** Clean interfaces want uniform return types. Research reality requires different stochastic processes with fundamentally different parameterizations.

A `GBMHead` returns `(mu, sigma)` — two scalars per batch element. A `HorizonHead` returns `(mu_seq, sigma_seq)` — a drift and volatility *for every time step*. A `NeuralBridgeHead` returns `(macro_ret, micro_returns, sigma)` — three tensors of different shapes.

**Why not force a common interface?** Because the stochastic processes these parameters feed into are genuinely different. Forcing a `GBMHead` to return per-step parameters would add artificial complexity. Forcing a `NeuralBridgeHead` to drop its macro target would lose information.

**The resolution:** `SynthModel.forward()` acts as the adapter. It inspects the head type, calls the appropriate simulation function, and always returns `(batch, n_paths, horizon)` paths. Downstream code (metrics, backtesting, logging) never sees head-specific details. The polymorphism lives in one place, documented with clear contracts.

### Why Leak-Safe Data Loading Matters

**The risk:** In time series, it is trivially easy to accidentally use future data when constructing features. A rolling z-score computed over the full series leaks information. A train/test split that ignores temporal ordering is meaningless. Normalizing across the entire dataset before splitting contaminates every fold.

**The approach:** `MarketDataLoader` and `FeatureEngineer` are designed from the ground up to be causal:

- `prepare_cache()` computes features *once* over the full series, but only using *causal* operations (rolling windows, forward-fill, no look-ahead).
- `make_input()` slices features strictly from `[start, start+length)` — no peeking ahead.
- `static_holdout()` splits on a date boundary. Train data ends before validation data begins. Period.
- Volatility stratification for regime-aware sampling uses only backward-looking windows.

**Why a dedicated abstraction?** Because data integrity is not negotiable. If the model trains on leaked data, every metric is meaningless. By centralizing loading in a purpose-built class with `FeatureEngineer` strategies, we make it structurally difficult to introduce leakage.

### Why CRPS Over Traditional Loss Functions

**The problem:** MSE rewards point accuracy. Log-likelihood rewards sharpness. Neither directly evaluates *calibration* — whether the predicted distribution honestly covers realized outcomes.

**The solution:** Continuous Ranked Probability Score (CRPS). It is a *proper scoring rule*: the expected score is minimized when the forecast distribution matches the true data-generating process. You cannot game it by being overconfident or underconfident.

**Implementation detail:** We use a sort-based O(n log n) algorithm instead of the naive O(n^2) pairwise computation. With 1,000 paths per sample, this matters.

**The afCRPS extension:** Standard CRPS has finite-ensemble bias. We implemented the *almost-fair CRPS* (afCRPS) from Lang et al. (2024), which interpolates between CRPS and fair CRPS to remove bias without the degeneracy problems of pure fair CRPS.

### Why 1,000 Paths

**The trade-off:** More paths = better density estimates = more GPU memory and compute.

At 1,000 paths:
- CRPS estimates are stable (variance of the estimator is low).
- Fan charts are visually smooth for human inspection.
- Memory fits comfortably on a single GPU with batch sizes of 4-16.
- The afCRPS finite-ensemble correction is small.

At 100 paths, CRPS variance is noticeably higher. At 10,000, memory becomes a constraint on consumer GPUs. 1,000 is the sweet spot for research iteration speed.

### Why Numerical Stability Is First-Class

**The reality:** During early training, models predict garbage — extreme drift values, near-zero volatility, huge log-returns. Without guardrails, `exp(large_number)` produces `Inf`, which propagates to `NaN` in CRPS, which kills the training run.

**The approach:**
- **Log-return clamping**: `MAX_LOG_RETURN_CLAMP = 20.0` — `exp(20) ≈ 4.85e8`, safely large for financial returns, prevents overflow.
- **Volatility floor**: Every head adds `1e-6` to sigma after `softplus`. Prevents division by zero in simulation.
- **Guard clauses**: Functions like CRPS scoring return sentinel values (`np.nan`) for degenerate inputs (empty intervals, zero-length horizons) rather than raising exceptions.

**Why not just catch exceptions?** Because NaN propagation is silent. A single NaN in a batch corrupts the loss, the gradient, and the parameter update — and you don't find out until the loss diverges epochs later. Prevention is strictly better than detection.

### Why Champion vs. Challenger

**The problem:** Absolute metrics (CRPS = 0.03) are hard to interpret. Is that good? For this asset? In this volatility regime? Compared to what?

**The solution:** Every new model (challenger) is evaluated side-by-side against the current best (champion) on the *exact same data windows*. The backtest engine:

1. Loads the champion from Hugging Face Hub.
2. Instantiates the challenger from the current config.
3. Runs both on aligned sliding windows.
4. Computes per-window CRPS and variance spread for both.
5. Logs overlapping fan charts to W&B.

**Why this matters:** It makes improvement claims concrete. "Challenger CRPS 0.028 vs. champion CRPS 0.031 on the same 50 windows" is actionable. "CRPS improved by 10%" without a shared baseline is not.

### Why W&B + Hugging Face Hub Together

**W&B** excels at: metrics over time, hyperparameter sweeps, interactive tables, comparing runs.

**Hugging Face Hub** excels at: model storage, version control for checkpoints, model cards, community sharing.

**Neither alone is sufficient.** W&B doesn't host model files well. Hugging Face doesn't do interactive experiment dashboards. By bridging them through `HubManager`, we get:

- W&B runs that link to their Hugging Face artifacts.
- Hugging Face model cards that link back to the W&B run.
- A single `save_and_push()` call that handles both.

---

## Architectural Trade-offs We Made

### Flexibility vs. Type Safety

We chose flexibility. Head return types are not enforced by the type system — `SynthModel.forward()` uses `isinstance` checks. A strict approach would use tagged unions or a common `HeadOutput` dataclass. We chose not to because:

- It would force every head author to know about a central type.
- The `isinstance` dispatch lives in one place and is easy to audit.
- Research moves fast; premature type constraints slow iteration.

**The cost:** Adding a new head type requires updating `SynthModel.forward()`. We accept this because new heads are infrequent, and the routing logic is well-documented.

### Declarative Configs vs. Code Readability

Hydra configs are powerful but opaque. A newcomer reading `configs/model/hybrid_v2.yaml` might not immediately understand that `_target_: src.models.registry.TransformerBlock` constructs a Python object. The indirection through the registry adds another layer.

**We chose this trade-off** because the alternative — hardcoded model construction in Python — doesn't support the experiment velocity we need. The cost is a steeper onboarding ramp, which we mitigate with documentation and example notebooks.

### Graceful Degradation vs. Fail-Fast

We follow a dual strategy:

- **Fail fast** for programmer errors: empty block lists, wrong API usage, missing configuration keys. These are `ValueError` or `RuntimeError`.
- **Degrade gracefully** for boundary conditions: short horizons, empty intervals, near-zero volatility. These return sentinel values or clamped results.

**The rationale:** Programmer errors should be loud and immediate — they indicate a bug. Boundary conditions often arise legitimately during training (a batch with degenerate values, an edge case in CRPS scoring) and should not crash a multi-hour training run.

### Research Notebooks as First-Class Citizens

Notebooks (`notebooks/`) are not afterthoughts or demos. They are active research tools:

- `dlinear_train_and_backtest.ipynb` — end-to-end workflow for DLinear experiments.
- `sde_head_with_sde_block.ipynb` — exploring neural SDE architectures.
- `walkforward_dlinear_regime.ipynb` — regime-aware walk-forward validation.

**Why this matters:** Notebooks lower the barrier to experimentation. A researcher can clone the repo, open a notebook, and run a full train-backtest cycle without understanding the CLI, Hydra, or the module structure. They also serve as executable documentation — if a notebook runs, the documented workflow works.

---

## What We Tried and Walked Back

Research is iterative. Some decisions were made, tested, and reversed:

- **Experimental heads with high loss**: Several head variants (GaussianSpectralHead, StudentTHorizonHead with spectral basis, ProbabilisticHorizonHead) were implemented, merged, and then reverted after evaluation showed they produced high losses or numerical instability at longer horizons (H=288). The commit history records these as deliberate reverts, not accidents. The lesson: a theoretically elegant head means nothing if it blows up in practice.

- **Defensive shape handling everywhere**: Early versions had `if paths.ndim == 2: paths = paths.unsqueeze(-1)` scattered throughout the codebase. We replaced this with a single shape contract enforced at the `SynthModel` level. Downstream code now trusts the contract. This was a deliberate shift from "handle everything everywhere" to "guarantee it once at the boundary."

- **Magic numbers**: Early simulation code used inline constants (`torch.clamp(x, -20, 20)`). These were extracted to named constants (`MAX_LOG_RETURN_CLAMP`) with documenting comments explaining why the value was chosen. Small change, big readability gain.

---

## Principles for Contributors

1. **New blocks go in `src/models/components/`**. Use the `@registry.register_block()` decorator. Don't modify imports elsewhere. Auto-discovery handles the rest.

2. **New heads go in `src/models/heads.py`** and must be added to the `isinstance` dispatch in `SynthModel.forward()`. Document the return signature.

3. **Config changes are experiments**. Create a new YAML file in `configs/model/` rather than modifying an existing one. This preserves reproducibility.

4. **Respect the shape contract**: `(batch, seq, d_model)` in, `(batch, seq, d_model)` out. If your block changes dimensions, declare it in the registry metadata.

5. **Test numerically**: Check for NaN and Inf with extreme inputs. If your block uses `exp()`, `log()`, or division, add clamping or floors.

6. **Don't break the data pipeline**: Features must be causal. If you add a new `FeatureEngineer`, ensure `prepare_cache()` uses only backward-looking operations.

7. **Log everything**: If it affects model quality, it should appear in W&B. If it's a publishable artifact, it should go to Hugging Face Hub.

8. **Prefer reverting over patching**: If a new head or block produces worse results, revert it cleanly rather than adding workarounds. The git history is the lab notebook.

---

*This document reflects the state of the codebase as of February 2026. As the project evolves, so will the philosophy — but the core commitments to probabilistic forecasting, composability, and transparency are foundational.*
