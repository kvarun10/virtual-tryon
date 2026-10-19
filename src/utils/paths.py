"""Storage layout (local or Google Drive) derived from the config.

Layout under ``paths.drive_root`` (e.g. /content/drive/MyDrive/VirtualTryOn)::

    datasets/      raw datasets you download yourself (never committed)
    pretrained/    third-party weights (SCHP, ...) with a provenance JSON next to each file
    checkpoints/   weights trained by this project, one folder per experiment
    cache/         indexes, splits, cached parser outputs (safe to delete, will be rebuilt)
    outputs/       figures, reports, generated images, one folder per experiment
    logs/          text and JSONL logs
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.config import ProjectConfig


def is_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


@dataclass
class ProjectPaths:
    drive_root: Path
    datasets: Path
    pretrained: Path
    checkpoints: Path
    cache: Path
    outputs: Path
    logs: Path

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> "ProjectPaths":
        p = cfg.paths
        return cls(
            drive_root=Path(p.drive_root),
            datasets=Path(p.datasets_dir),
            pretrained=Path(p.pretrained_dir),
            checkpoints=Path(p.checkpoints_dir),
            cache=Path(p.cache_dir),
            outputs=Path(p.outputs_dir),
            logs=Path(p.logs_dir),
        )

    def ensure(self) -> "ProjectPaths":
        for path in (self.datasets, self.pretrained, self.checkpoints, self.cache, self.outputs, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def experiment_dir(self, kind: str, experiment: str) -> Path:
        base = {"checkpoints": self.checkpoints, "outputs": self.outputs, "logs": self.logs}[kind]
        path = base / experiment
        path.mkdir(parents=True, exist_ok=True)
        return path
