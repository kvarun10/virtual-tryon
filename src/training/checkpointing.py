"""Checkpoints that survive Colab disconnects.

* Files are written to ``<name>.tmp`` and then renamed (atomic), so an interrupted save never
  corrupts the previous checkpoint.
* ``last.pt`` is overwritten every ``checkpoint_every`` steps; ``best.pt`` holds the best validation
  score; ``step_XXXXXXX.pt`` snapshots keep the most recent ``keep`` files.
* A checkpoint stores model and optimiser states, the AMP scaler, the step / epoch / batch position,
  RNG states, the resolved config and a *model signature*. Resuming with a different architecture or
  label set raises :class:`CheckpointMismatchError` instead of loading wrong weights.
* Everything is saved as tensors / plain Python types, so ``torch.load(weights_only=True)`` works.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


class CheckpointMismatchError(RuntimeError):
    pass


def rng_state() -> Dict:
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": [s for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "numpy": [np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "python": list(random.getstate()[1]) + [random.getstate()[0], random.getstate()[2]],
    }


def _byte_cpu(t: torch.Tensor) -> torch.Tensor:
    """RNG setters require CPU uint8 tensors, whatever map_location the checkpoint was loaded with."""
    return t.detach().to("cpu", torch.uint8)


def set_rng_state(state: Dict) -> None:
    torch.set_rng_state(_byte_cpu(state["torch"]))
    if state.get("cuda") and torch.cuda.is_available():
        cuda_states = [_byte_cpu(s) for s in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
    n = state["numpy"]
    np.random.set_state((n[0], n[1].detach().cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))
    p = state["python"]
    random.setstate((p[-2], tuple(p[:-2]), p[-1]))


def atomic_save(obj: Dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


class CheckpointManager:
    def __init__(self, directory: Path, keep: int = 3):
        self.dir = Path(directory)
        self.keep = keep

    @property
    def last(self) -> Path:
        return self.dir / "last.pt"

    @property
    def best(self) -> Path:
        return self.dir / "best.pt"

    def save(self, state: Dict, step: int, is_best: bool = False, snapshot: bool = True) -> Path:
        atomic_save(state, self.last)
        if is_best:
            atomic_save(state, self.best)
        if snapshot and self.keep > 0:
            atomic_save(state, self.dir / f"step_{step:07d}.pt")
            snaps = sorted(self.dir.glob("step_*.pt"))
            for old in snaps[: -self.keep]:
                old.unlink(missing_ok=True)
        return self.last

    def load(self, path: Optional[Path] = None, signature: Optional[Dict] = None, map_location="cpu") -> Optional[Dict]:
        path = Path(path) if path else self.last
        if not path.is_file():
            return None
        try:
            state = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:  # noqa: BLE001
            raise CheckpointMismatchError(f"Cannot read checkpoint {path}: {type(exc).__name__}: {exc}") from exc
        if signature is not None and state.get("signature") != signature:
            diff = {k: (state.get("signature", {}).get(k), v) for k, v in signature.items()
                    if state.get("signature", {}).get(k) != v}
            raise CheckpointMismatchError(
                f"Checkpoint {path} was made for a different model/label setup (checkpoint vs current): {diff}. "
                "Use another experiment.name or delete the checkpoint folder to start fresh."
            )
        return state
