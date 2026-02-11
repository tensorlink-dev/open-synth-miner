# Building Open Synth Miner: 59 PRs, 6 Weeks, and an AI Pair Programmer

*A build-in-public narrative of how we went from an empty repo to a composable neural forecasting framework for Bittensor SN50 -- with Codex and Claude writing most of the code.*

---

## The premise

We set out to build an open-source research framework for [Bittensor Subnet 50 (Synth)](https://github.com/synth-subnet): a competition where miners submit probabilistic price forecasts -- 1,000 Monte Carlo paths per prediction -- and are scored by how well those paths capture reality. The catch: forecasts need to be fast, the scoring metric (CRPS) is unforgiving, and the design space is enormous. Transformers? LSTMs? Stochastic differential equations? All of the above stitched together?

We decided to find out by building in public, tracking every experiment to Weights & Biases, and publishing every artifact to Hugging Face Hub. This is the story of what happened across 59 merged pull requests.

---

## Phase 1: Scaffolding (PRs #1 -- #11, Jan 1 -- Jan 4)

**The goal: get something -- anything -- running.**

PR #1 landed the boilerplate: a Hydra-driven config system, a registry for neural network blocks, and a factory that could stitch those blocks into a `HybridBackbone` from YAML. The very first model recipe was a Transformer block followed by an LSTM block, topped with a GBM (Geometric Brownian Motion) head that turned a latent vector into 1,000 simulated price paths.

What followed was the unglamorous reality of early-stage ML engineering. PR #3 added `requirements.txt` so `uv` could install dependencies. PR #4 added `setup.py`. PR #5 fixed an `AttributeError` on the `TransformerBlock` because the registry entries didn't support attribute access. PR #6 patched syntax warnings in the `properscoring` integration. PRs #7 and #8 wrestled with tensor shape mismatches in the backbone -- the kind of bugs where `(batch, seq, d_model)` silently becomes `(batch, d_model)` and everything downstream collapses.

By PR #9 the `HybridBackbone` could tolerate extra Hydra kwargs without crashing, and PR #10 fixed rendering issues in the README. PR #11 finally resolved a persistent `TypeError` when calling the model.

**Lesson learned:** The first 11 PRs produced zero scientific results. They produced something more important -- a model that could accept a batch, produce 1,000 paths, and not crash. That foundation made everything else possible.

---

## Phase 2: The Data Pipeline (PRs #12 -- #22, Jan 3 -- Jan 4)

**The goal: leak-safe market data loading.**

Financial ML has a unique hazard: data leakage. If your feature engineering peeks at future prices, your backtest results are worthless. PR #12 built a production-grade `MarketDataLoader` with a `HFParquetSource` that pulled OHLCV data from our Hugging Face dataset (`tensorlink-dev/open-synth-training-data`). It was leak-safe by design -- windowed extraction with causal boundaries only.

Then came the parade of data fixes. PR #13 fixed unauthorized access errors when hitting the Hugging Face API (turns out you need `repo_type="dataset"`). PR #14 added fractional cutoff support for train/val/test splits. PR #15 sorted DataFrames by date to prevent leakage at split boundaries. PR #16 removed temporal assertions that were too strict for static holdout validation. PR #17 built the integration layer connecting the loader to the training loop. PR #18 created a unified `AblationExperiment` class so we could systematically compare configurations.

PRs #19 and #20 unified the two competing `MarketDataLoader` classes that had diverged, and PR #21 refactored ablation support for multi-config sweeps. PR #22 taught the `HFParquetSource` to handle repositories with multiple parquet files.

**Lesson learned:** We spent more PRs on the data pipeline than on any single model architecture. This is normal. In financial ML, if you can't trust your data splits, you can't trust your results.

---

## Phase 3: Architecture Explosion (PRs #23 -- #33, Jan 4 -- Feb 9)

**The goal: explore the hybrid design space.**

With solid data infrastructure in place, we started experimenting with model architectures -- and things moved fast.

PR #23 added advanced time series blocks: flexible patching utilities and a last-step adapter. PRs #24 and #25 polished the README with architecture diagrams and fixed the final `TypeError` in model calls.

Then PR #26 merged the dev branch and everything accelerated. PR #27 was the big one: it added **FEDformer**, **DLinear**, **TimesNet**, and **TimeMixer** blocks, along with a `HorizonHead` (per-step drift/volatility via cross-attention) and a `NeuralBridgeHead` (hierarchical macro/micro path generation). It also introduced `OHLCVEngineer` for real candlestick data with a 1-hour feature engineering pipeline.

PR #28 created separate training notebooks for each architecture so we could run them side-by-side. PR #29 fixed a crash when DLinear resampled data and the index bounds went stale. PR #30 fixed an `IndexError` in the Trainer when using `NeuralBridgeHead`. PR #31 generalized the regime-aware loader with pluggable ABC extension points.

PR #32 tackled a critical numerical stability issue: GBM log-returns were overflowing in `exp()`, producing NaN paths. The fix was a simple clamp to +/-20, but finding it required tracing NaN propagation through the entire simulation pipeline.

PR #33 solved NaN training with DLinear + NeuralBridgeHead by switching to stochastic ensemble path generation.

**Lesson learned:** Adding new architectures is easy. Making them numerically stable with stochastic path simulation is where the real work lives. Every new head type introduced a new way for gradients to explode.

---

## Phase 4: Numerical Stability and the NaN Wars (PRs #33 -- #42, Feb 9)

**The goal: make training actually converge.**

This was the most intense period -- nine PRs merged in a single day, most of them fighting NaN values and numerical instability.

PR #35 added a stride parameter to `MarketDataLoader` so we could control window overlap and get enough training batches from resampled OHLCV data. PR #36 exposed this stride parameter at the top level to fix chronically low batch counts.

PR #37 debugged NaN in `NeuralBridgeHead` by adding LayerNorm to both `GBMHead` and `NeuralBridgeHead` -- the DLinear backbone was producing unbounded activations that the heads couldn't handle.

PR #38 was a cleanup pass: removing redundant defensive code that had accumulated during debugging.

PR #39 introduced `NeuralSDEHead`, which used `torchsde` for proper stochastic differential equation integration rather than our hand-rolled Euler-Maruyama simulation.

PR #40 was a breakthrough: it fixed NaN/Infinity across all path simulation by switching from direct price accumulation to **log-space accumulation**. Instead of `price *= exp(return)`, we accumulated `log_price += return` and exponentiated once at the end. This single change eliminated an entire class of overflow bugs.

PR #41 added a composable `LayerNormBlock` so users could insert normalization between any two blocks in a hybrid recipe -- via YAML config, not code changes.

PR #42 aligned the loader's resample frequency with `NeuralBridgeHead`'s micro-step assumptions, fixing a subtle mismatch where the model expected 5-minute bars but received 1-hour bars.

**Lesson learned:** Stochastic simulation is a numerical minefield. Log-space accumulation (PR #40) was the single most impactful fix in the entire project history. If you're simulating GBM paths in PyTorch, do your arithmetic in log space.

---

## Phase 5: Research Tooling (PRs #43 -- #50, Feb 9 -- Feb 10)

**The goal: make experiments reproducible and shareable.**

With training working reliably, we shifted focus to research infrastructure.

PR #43 created a notebook demonstrating the `SDEHead` with `SDEEvolutionBlock` backbone, showing how the pieces compose. PR #44 documented available feature engineers. PR #45 built a walk-forward DLinear notebook with regime detection -- the first proper out-of-sample evaluation pipeline.

PR #46 fixed the CRPS multi-interval scorer to adapt to different horizon lengths (it had been hardcoded for horizon=12). PR #47 added `SimpleHorizonHead`, a lightweight alternative to the attention-based `HorizonHead` that used pooling + MLP instead of cross-attention -- fewer parameters, faster training, competitive CRPS.

PR #48 added RevIN (Reversible Instance Normalization) and patching blocks to the SDE notebook, pushing predictions from 12 steps to 288 steps. PR #49 added automatic LayerNorm insertion between blocks in `HybridBackbone` -- a one-line config flag that improved training stability across every architecture we tested.

PR #50 synchronized all notebooks with Colab installation cells so anyone could run experiments without local setup.

**Lesson learned:** Research notebooks are as important as the core framework. They serve as living documentation, reproducibility checkpoints, and onboarding material for new contributors.

---

## Phase 6: Scaling and Scoring (PRs #51 -- #55, Feb 10)

**The goal: fix the evaluation pipeline.**

PR #51 added auto-detection of feature dimensions from engineer classes, eliminating a common config mismatch where `feature_dim: 3` in the YAML didn't match the engineer's actual output.

PR #52 was a performance fix: the CRPS ensemble computation was using an O(n^2) pairwise distance matrix that caused CUDA out-of-memory errors with 1,000 paths. The fix replaced it with an O(n) sort-based algorithm -- same mathematical result, dramatically lower memory.

PR #53 added comprehensive shape validation to catch model design errors at construction time rather than during training.

PR #54 implemented **almost-fair CRPS (afCRPS)** from the AIFS-CRPS paper -- a bias-corrected variant of the ensemble CRPS loss that accounts for finite sample sizes. This was a subtle but important change: standard CRPS with finite ensembles has a known negative bias, and afCRPS corrects for it.

PR #55 fixed RevIN denormalization during inference. The model was producing paths in z-score space rather than price space, making the fan charts uninterpretable. The fix added automatic denormalization when `model.eval()` is called.

**Lesson learned:** Evaluation metrics deserve as much engineering rigor as model architectures. A biased loss function (standard CRPS with finite ensembles) or a broken denormalization step (RevIN) can silently undermine months of architecture work.

---

## Phase 7: Advanced Heads (PRs #56 -- #59, Feb 10 -- Feb 11)

**The goal: push the frontier on path generation.**

The final stretch introduced the most sophisticated simulation heads.

PR #56 fixed `HybridBackbone` crashes with sequence-length-changing blocks like `FlexiblePatchEmbed`. PR #57 fixed a training loader shape mismatch caused by channel-independent patch embedding.

PR #58 added `CLTHorizonHead` and `StudentTHorizonHead` -- heads that used Central Limit Theorem-based path generation and heavy-tailed (Student-t) innovations for more realistic fat-tailed return distributions.

PR #59 iterated on these heads, replacing the initial spectral basis approach with Brownian walk parameters. The spectral method was mathematically elegant but produced paths that were too smooth; the Brownian walk parameters captured the rough, jagged character of real price movements.

**Lesson learned:** Financial returns are fat-tailed. Gaussian assumptions (GBM) are a useful starting point, but Student-t innovations and CLT-based generation produce more realistic path ensembles. The right simulation head matters as much as the right backbone.

---

## By the numbers

| Metric | Value |
|--------|-------|
| Total PRs merged | 59 |
| Calendar time | ~6 weeks (Jan 1 -- Feb 11) |
| Distinct model architectures explored | 7 (Transformer, LSTM, DLinear, FEDformer, TimesNet, TimeMixer, SDE) |
| Simulation head types | 8 (GBM, SDE, Horizon, SimpleHorizon, CLT, StudentT, NeuralBridge, NeuralSDE) |
| NaN-related bug fix PRs | 6 |
| PRs dedicated to data pipeline | 11 |
| AI agent PRs (Codex + Claude) | 59 of 59 |

---

## What we shipped

Open Synth Miner is a framework where you can:

1. **Define a hybrid model in YAML** -- stack a Transformer, an LSTM, a DLinear block, add LayerNorm between them, top it with a StudentTHorizonHead, and have it all instantiated automatically.

2. **Train on leak-safe market data** -- with pluggable feature engineers, regime-aware sampling, and configurable windowing.

3. **Generate 1,000 probabilistic price paths** -- using GBM, SDE integration, cross-attention horizon heads, or CLT-based heavy-tailed simulation.

4. **Evaluate with proper scoring rules** -- bias-corrected afCRPS, multi-interval breakdowns, and challenger-vs-champion backtesting.

5. **Publish everything automatically** -- checkpoints to Hugging Face Hub, metrics to W&B, model cards linking runs to artifacts.

---

## What we learned about building with AI agents

Every single one of these 59 PRs was authored by an AI coding agent (first Codex, then Claude). The pattern that emerged:

- **AI is great at scaffolding.** The initial boilerplate, registry system, and factory pattern came together quickly.
- **AI is great at systematic fixes.** Once we identified the NaN-in-log-space pattern, applying it across all simulation heads was mechanical.
- **AI needs human direction at architecture boundaries.** Deciding *which* head to try next, *whether* to use cross-attention or pooling, *what* the right CRPS variant is -- these required domain knowledge that the human brought.
- **The PR-per-task workflow works.** Each PR had a focused scope. When something broke (and things broke constantly), the blast radius was small and the fix was usually one more PR away.

Building in public with AI pair programming isn't about the AI writing perfect code. It's about maintaining velocity: 59 PRs in 6 weeks, each one small enough to review, merge, and build on. The framework emerged not from a grand design document, but from the accumulation of focused, testable increments.

---

*Open Synth Miner is open source at [tensorlink-dev/open-synth-miner](https://github.com/tensorlink-dev/open-synth-miner). Contributions welcome.*
