"""Challenger vs Champion backtesting utilities."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import wandb
from torch.utils.data import DataLoader

from osa.data.base_dataset import StridedTimeSeriesDataset
from osa.models.factory import get_model
from osa.research.metrics import CRPSMultiIntervalScorer, crps_ensemble
from osa.tracking.wandb_logger import log_backtest_results


class ChallengerVsChampion:
    """Run challenger vs champion comparisons on shared data."""

    def __init__(
        self,
        challenger_cfg,
        champion_cfg,
        data_window: torch.Tensor,
        *,
        time_increment: int,
        horizon: int,
        n_paths: int = 1000,
        device: str | torch.device = "cpu",
    ) -> None:
        self.challenger_cfg = challenger_cfg
        self.champion_cfg = champion_cfg
        self.data_window = data_window
        self.time_increment = time_increment
        self.horizon = horizon
        self.n_paths = n_paths
        self.device = torch.device(device)
        self.scorer = CRPSMultiIntervalScorer(time_increment=time_increment)

    def _make_dataloader(self) -> DataLoader:
        target = self.data_window
        dataset = StridedTimeSeriesDataset(target=target, context_len=target.shape[0] - 2, pred_len=2, stride=1)
        return DataLoader(dataset, batch_size=1)

    def _variance_spread(self, paths_a: torch.Tensor, paths_b: torch.Tensor) -> float:
        var_a = paths_a[:, :, -1].var(dim=1).mean()
        var_b = paths_b[:, :, -1].var(dim=1).mean()
        return float((var_a - var_b).abs())

    def run(self, log_to_wandb: bool = True, step: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        dataloader = self._make_dataloader()
        champion = get_model(self.champion_cfg).to(self.device).eval()
        challenger = get_model(self.challenger_cfg).to(self.device).eval()

        aggregated: Dict[str, Dict[str, float]] = {"champion": {}, "challenger": {}, "spread": {}}

        for batch in dataloader:
            history = batch["history"].to(self.device)
            initial_price = batch["initial_price"].to(self.device)
            actual_series = batch["actual_series"].to(self.device)

            with torch.no_grad():
                champ_paths, _, _ = champion(history, initial_price=initial_price, horizon=self.horizon, n_paths=self.n_paths)
                chall_paths, _, _ = challenger(history, initial_price=initial_price, horizon=self.horizon, n_paths=self.n_paths)

            champ_total, champ_detail = self.scorer(champ_paths[0], actual_series[0])
            chall_total, chall_detail = self.scorer(chall_paths[0], actual_series[0])

            aggregated["champion"] = {row["Interval"]: float(row["CRPS"]) for row in champ_detail if row["Increment"] == "Total"}
            aggregated["challenger"] = {row["Interval"]: float(row["CRPS"]) for row in chall_detail if row["Increment"] == "Total"}
            aggregated["spread"]["variance_spread"] = self._variance_spread(champ_paths, chall_paths)

            champ_fan = champ_paths[0].detach()
            chall_fan = chall_paths[0].detach()
            overlap_crps = crps_ensemble(champ_fan[:, -1], chall_fan[:, -1]).mean().item()
            aggregated["spread"]["crps_overlap"] = overlap_crps

            if log_to_wandb:
                combined = {
                    "champion": aggregated["champion"],
                    "challenger": aggregated["challenger"],
                    "spread": aggregated["spread"],
                }
                log_backtest_results(combined, step=step, prefix="champion_vs_challenger")
                table = wandb.Table(columns=["type", "path"])
                table.add_data("champion", wandb.Histogram(champ_fan[:, -1].cpu().numpy()))
                table.add_data("challenger", wandb.Histogram(chall_fan[:, -1].cpu().numpy()))
                wandb.log({"backtest/final_price_overlap": table}, step=step)

        return aggregated


__all__ = ["ChallengerVsChampion"]
