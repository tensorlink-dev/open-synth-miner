"""Entry point for running Synth hybrid miner experiments."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

from src.data import MarketDataLoader
from src.models.registry import discover_components
from src.research.backtest import ChallengerVsChampion
from src.research.experiment_mgr import run_experiment
from src.research.ablation_grid import (
    AblationGridSpec,
    generate_ablation_grid,
    describe_grid,
)
from src.tracking.hub_manager import HubManager


def _train_flow(cfg: DictConfig) -> None:
    discover_components("src/models/components")
    result = run_experiment(cfg)
    model = result["model"]
    metrics = result["metrics"]
    resolved_cfg = result["config"]
    run = result["run"]
    recipe = result.get("recipe", [])
    block_hash = result.get("block_hash", "na")

    backbone = cfg.get("model", {}).get("backbone", {}).get("_target_", "unknown")
    head = cfg.get("model", {}).get("head", {}).get("_target_", "unknown")
    architecture_graph = cfg.get("model", {}).get("backbone", {})

    manager = HubManager(
        run=run,
        backbone_name=backbone,
        head_name=head,
        block_hash=block_hash,
        recipe=recipe,
        architecture_graph=architecture_graph,
        resolved_config=OmegaConf.to_container(cfg, resolve=True),
    )
    crps_score = float(metrics.get("crps", 0.0)) if metrics else 0.0
    hf_link = manager.save_and_push(model=model, crps_score=crps_score)
    shareable = manager.get_shareable_report(crps_score=crps_score, hf_link=hf_link)
    print(shareable)
    wandb.finish()


def _backtest_flow(cfg: DictConfig) -> None:
    discover_components("src/models/components")
    data_cfg = cfg.get("data")
    loader: MarketDataLoader = hydra.utils.instantiate(data_cfg)
    prices = loader.get_price_series()

    challenger_cfg = cfg
    champion_cfg = {"model": {"hf_repo_id": cfg.backtest.champion_repo_id, "architecture": cfg.model}}

    engine = ChallengerVsChampion(
        challenger_cfg=challenger_cfg,
        champion_cfg=champion_cfg,
        data_window=prices,
        time_increment=cfg.backtest.time_increment,
        horizon=cfg.backtest.horizon,
        n_paths=cfg.backtest.n_paths,
        device=cfg.backtest.get("device", "cpu"),
    )
    results = engine.run(log_to_wandb=True)
    print("Backtest results:", results)
    wandb.finish()


def _ablation_flow(cfg: DictConfig) -> None:
    """Run a grid ablation study over engineers, RevIN/DLinear combos, kernel sizes, and heads.

    Groups configs by feature_dim so each group gets a matching data loader.
    """
    discover_components("src/models/components")

    abl_cfg = cfg.get("ablation", {})
    spec = AblationGridSpec(
        engineers=list(abl_cfg.get("engineers", ["zscore", "wavelet"])),
        revin_dlinear=list(abl_cfg.get("revin_dlinear", ["none", "revin", "dlinear", "revin_dlinear"])),
        kernel_sizes=list(abl_cfg.get("kernel_sizes", [15, 25, 51])),
        heads=list(abl_cfg.get("heads", ["gbm", "sde", "simple_horizon", "clt_horizon", "gaussian_spectral"])),
        d_model=int(abl_cfg.get("d_model", 32)),
    )

    training_overrides = OmegaConf.to_container(abl_cfg.get("training", {}), resolve=True)
    configs = generate_ablation_grid(spec, training_overrides=training_overrides or None)

    print(describe_grid(configs))
    print()

    from src.research.ablation import AblationExperiment
    from src.research.experiment_mgr import _build_dummy_batch
    from torch.utils.data import DataLoader, TensorDataset

    # Group configs by feature_dim (different engineers produce different dims)
    groups: dict = {}
    for name, exp_cfg in configs.items():
        fdim = int(exp_cfg.training.feature_dim)
        groups.setdefault(fdim, {})[name] = exp_cfg

    all_results: dict = {}
    for fdim, group_configs in groups.items():
        print(f"\n--- Running group: feature_dim={fdim} ({len(group_configs)} configs) ---")

        # Build a matching dummy loader for this feature_dim group
        sample_cfg = next(iter(group_configs.values()))
        t_cfg = OmegaConf.to_container(sample_cfg.get("training", {}), resolve=True)
        batch = _build_dummy_batch(t_cfg)
        dataset = TensorDataset(batch["history"], batch["initial_price"])
        loader = DataLoader(dataset, batch_size=t_cfg.get("batch_size", 4))

        experiment = AblationExperiment(configs=group_configs, mode="train")
        results = experiment.run(train_loader=loader, val_loader=loader)
        all_results.update(results)

    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    for name, metrics in sorted(all_results.items()):
        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"  {name}: {metric_str}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = cfg.get("mode", "train")
    if mode == "train":
        _train_flow(cfg)
    elif mode == "backtest":
        _backtest_flow(cfg)
    elif mode == "ablation":
        _ablation_flow(cfg)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
