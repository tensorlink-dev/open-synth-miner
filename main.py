"""Entry point for running Synth hybrid miner experiments."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

from src.data import MarketDataLoader
from src.models.registry import discover_components
from src.research.backtest import ChallengerVsChampion
from src.research.experiment_mgr import run_experiment
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


def _optimize_flow(cfg: DictConfig) -> None:
    from src.research.optimizer import FeatureOptimizer

    opt_cfg = cfg.optimize
    eng_label = opt_cfg.engineer.get("_target_", opt_cfg.engineer.get("type", "?"))
    print(f"Starting Feature Optimization (trials={opt_cfg.n_trials}, "
          f"engineer={eng_label}, sampler={opt_cfg.get('sampler', 'tpe')})")

    optimizer = FeatureOptimizer(cfg)
    result = optimizer.run()

    params = result["best_params"]

    # Separate toggles, windows, and other params for readability.
    toggles_on = {k: v for k, v in params.items() if k.startswith("use_") and v is True}
    toggles_off = {k: v for k, v in params.items() if k.startswith("use_") and v is False}
    windows = {k: v for k, v in params.items() if k.endswith("_window")}
    other = {k: v for k, v in params.items()
             if not k.startswith("use_") and not k.endswith("_window")}

    print(f"\nIntrinsic Dimension: {result['best_value']:.4f}")

    if toggles_on:
        print("\nEnabled features:")
        for k in sorted(toggles_on):
            print(f"  {k}: true")
    if toggles_off:
        print("\nDisabled features:")
        for k in sorted(toggles_off):
            print(f"  {k}: false")
    if windows:
        print("\nOptimal windows:")
        for k, v in sorted(windows.items()):
            print(f"  {k}: {v}")
    if other:
        print("\nOther parameters:")
        for k, v in sorted(other.items()):
            print(f"  {k}: {v}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = cfg.get("mode", "train")
    if mode == "train":
        _train_flow(cfg)
    elif mode == "backtest":
        _backtest_flow(cfg)
    elif mode == "optimize":
        _optimize_flow(cfg)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
