"""Bridge utilities between W&B runs and the Hugging Face Hub for hybrid models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
import yaml
from huggingface_hub import HfApi, ModelCard, ModelCardData


class HubManager:
    """Handles local saving, Hub uploads, architecture artifacts, and reporting."""

    def __init__(
        self,
        run: wandb.sdk.wandb_run.Run,
        backbone_name: str,
        head_name: str,
        block_hash: str,
        recipe: Optional[list],
        architecture_graph: Optional[Dict] = None,
        resolved_config: Optional[Dict] = None,
        output_root: str = "outputs",
        repo_id: str = "username/SN50-Hybrid-Hub",
    ) -> None:
        self.run = run
        self.backbone_name = backbone_name
        self.head_name = head_name
        self.block_hash = block_hash
        self.recipe = recipe or []
        self.architecture_graph = architecture_graph or {}
        self.resolved_config = resolved_config or {}
        self.output_root = Path(output_root)
        self.repo_id = repo_id
        self.api = HfApi()

    @property
    def output_dir(self) -> Path:
        run_id = self.run.id if self.run is not None else "offline"
        path = self.output_root / self.backbone_name / self.block_hash / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _hf_subpath(self) -> str:
        run_id = self.run.id if self.run is not None else "offline"
        return f"{self.backbone_name}/{self.block_hash}/{run_id}"

    def _hf_folder_link(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/tree/main/{self._hf_subpath()}"

    def _write_model_card(self, save_dir: Path, crps_score: float) -> Path:
        tags = ["bittensor", "sn50", "time-series"]
        run_id = self.run.id if self.run is not None else "offline"
        run_url = self.run.url if self.run is not None else ""
        card_data = ModelCardData(
            model_name=f"{self.backbone_name}-{self.head_name}-{run_id}",
            tags=tags,
        )
        card_data.backbone = self.backbone_name
        card_data.head = self.head_name
        card_data.crps_score = crps_score
        card_data.hybrid_recipe = self.recipe

        description_lines = [
            f"W&B Run: {run_url}",
            f"Hybrid Recipe: {json.dumps(self.recipe)}",
            f"Hugging Face folder: {self._hf_folder_link()}",
            f"Backbone: {self.backbone_name}",
            f"Head: {self.head_name}",
            f"CRPS Score: {crps_score:.6f}",
        ]
        body = "\n".join(description_lines)

        card = ModelCard.from_template(card_data=card_data, model_description=body)
        readme_path = save_dir / "README.md"
        card.save(readme_path)
        return readme_path

    def _log_architecture_artifact(self, save_dir: Path) -> Optional[str]:
        if self.run is None:
            return None
        arch_path = save_dir / "architecture.json"
        with arch_path.open("w", encoding="utf-8") as f:
            json.dump(self.architecture_graph or {"recipe": self.recipe}, f, indent=2)
        artifact = wandb.Artifact(name=f"hybrid-architecture-{self.run.id}", type="architecture")
        artifact.add_file(str(arch_path))
        logged_art = self.run.log_artifact(artifact)
        return logged_art.id if logged_art else None

    def _write_resolved_config(self, save_dir: Path) -> Optional[Path]:
        if not self.resolved_config:
            return None
        cfg_path = save_dir / "resolved_config.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.resolved_config, f, sort_keys=False)
        return cfg_path

    def save_and_push(self, model: torch.nn.Module, crps_score: float) -> str:
        """Save model artifacts locally, push to HF, and update W&B summary."""

        save_dir = self.output_dir
        weights_path = save_dir / "model.pt"
        torch.save(model.state_dict(), weights_path)

        self._log_architecture_artifact(save_dir)
        self._write_model_card(save_dir, crps_score)
        self._write_resolved_config(save_dir)

        self.api.upload_folder(
            folder_path=str(save_dir),
            path_in_repo=self._hf_subpath(),
            repo_id=self.repo_id,
            repo_type="model",
        )

        hf_link = self._hf_folder_link()
        if self.run is not None:
            self.run.summary.update({"huggingface_link": hf_link, "hybrid_recipe": self.recipe})
        return hf_link

    def get_shareable_report(self, crps_score: float, hf_link: Optional[str] = None) -> str:
        link = hf_link or self._hf_folder_link()
        run_url = self.run.url if self.run is not None else ""
        recipe_text = json.dumps(self.recipe)
        return (
            "Model: "
            f"{self.backbone_name} + {self.head_name}\n"
            f"Hybrid Recipe: {recipe_text}\n"
            f"CRPS Score: {crps_score:.6f}\n"
            f"HF Folder: {link}\n"
            f"W&B Dashboard: {run_url}"
        )


__all__ = ["HubManager"]
