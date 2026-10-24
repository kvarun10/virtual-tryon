"""Shared GAN training loop for the warp (Stage B) and texture (Stage C) networks.

A subclass provides the parts that differ, in ``build()``:
``train_ds`` (with ``set_epoch``), ``val_ds``, ``collate``, ``gen``, ``disc``, ``opt_g``, ``opt_d``
and ``signature``. It also implements ``micro_step``, ``adv_active`` and ``validate``. Everything
else is shared:

* AMP (fp16 autocast + GradScaler) on CUDA only, gradient accumulation, generator grad-norm clip at 10
* atomic checkpoints (``last.pt`` / ``best.pt`` / step snapshots) under ``checkpoints/<experiment>/<stage>/``
* exact resume, including the position inside an epoch and the RNG states (checkpoints are loaded on CPU)
* validation every ``val_every`` steps; ``best.pt`` keeps the best ``score_key``; optional early stopping
* JSONL log at ``logs/<experiment>/<stage>_train.jsonl``

Checkpoint keys are unchanged from the original warp trainer, so existing warp checkpoints still load.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ProjectConfig
from src.data.dataset import PreparedData, prepare_dataset
from src.training.checkpointing import CheckpointManager, ensure_not_frozen, rng_state, set_rng_state
from src.utils.device import select_device
from src.utils.logging import JsonlLogger
from src.utils.paths import ProjectPaths
from src.utils.reproducibility import seed_everything


class GANTrainer:
    stage = "base"  # checkpoint / output / log sub-folder name
    score_key = "score"  # validation metric that selects best.pt (higher is better)
    metric_version = 1  # bump when the meaning of score_key changes (resets best on resume)

    def __init__(self, cfg: ProjectConfig, prepared: Optional[PreparedData] = None,
                 device: Optional[torch.device] = None, verbose: bool = True, **build_kwargs):
        self.cfg, self.t = cfg, cfg.training
        self.verbose = verbose
        seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
        self.device = device or select_device(cfg.experiment.device)
        self.prepared = prepared or prepare_dataset(cfg)
        self.build(**build_kwargs)
        self.use_amp = bool(self.t.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        paths = ProjectPaths.from_config(cfg)
        self.run_name = getattr(self.t, f"{self.stage}_run_name", self.stage)  # e.g. warp, warp_pairfilter
        self.ckpt = CheckpointManager(paths.checkpoints / cfg.experiment.name / self.run_name)
        self.out_dir = paths.outputs / cfg.experiment.name / self.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = JsonlLogger(paths.logs / cfg.experiment.name / f"{self.run_name}_train.jsonl")
        self.step = self.epoch = self.batch_in_epoch = 0
        self.best = -math.inf
        self.bad_evals = 0
        self.history = []

    # ------------------------------------------------------------------ hooks
    def build(self, **kwargs) -> None:
        raise NotImplementedError

    def adv_active(self) -> bool:
        raise NotImplementedError

    def micro_step(self, batch: Dict, accum: int) -> Dict[str, float]:
        raise NotImplementedError

    def validate(self, max_batches: Optional[int] = None, save_grid: bool = True, kind: str = "val") -> Dict:
        raise NotImplementedError

    def describe_run(self, per_epoch: int, total: int) -> str:
        return (f"training {self.stage}: {len(self.train_ds)} samples, {per_epoch} steps/epoch, target {total} steps, "
                f"device {self.device}, amp {self.use_amp}")

    # ------------------------------------------------------------------ checkpointing
    def state(self) -> Dict:
        return {"gen": self.gen.state_dict(), "disc": self.disc.state_dict(), "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(), "scaler": self.scaler.state_dict(), "step": self.step,
                "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch, "best": self.best,
                "bad_evals": self.bad_evals, "rng": rng_state(), "signature": self.signature,
                "metric_version": self.metric_version, "config": json.loads(json.dumps(self.cfg.to_dict()))}

    def save(self, is_best: bool = False) -> None:
        self.ckpt.save(self.state(), self.step, is_best=is_best)

    def resume(self, path: Optional[Path] = None) -> bool:
        # Load on CPU: load_state_dict copies weights to the model's device and Adam moves its own state.
        # (Loading straight to CUDA also moved the RNG states, which torch only accepts as CPU byte tensors.)
        st = self.ckpt.load(path, signature=self.signature, map_location="cpu")
        if st is None:
            return False
        self.gen.load_state_dict(st["gen"])
        self.disc.load_state_dict(st["disc"])
        self.opt_g.load_state_dict(st["opt_g"])
        self.opt_d.load_state_dict(st["opt_d"])
        self.scaler.load_state_dict(st["scaler"])
        self.step, self.epoch, self.batch_in_epoch = st["step"], st["epoch"], st["batch_in_epoch"]
        self.best, self.bad_evals = st["best"], st["bad_evals"]
        if st.get("metric_version") != self.metric_version:
            self._print("checkpoint has a best-score from an older metric definition; resetting best/early-stop counters")
            self.best, self.bad_evals = -math.inf, 0
        set_rng_state(st["rng"])
        self._print(f"resumed from step {self.step} (epoch {self.epoch}, batch {self.batch_in_epoch})")
        return True

    def evaluate_checkpoint(self, which: str = "last") -> Dict:
        """Load generator weights from ``last`` / ``best`` / a path and score them on the WHOLE val split."""
        path = {"last": self.ckpt.last, "best": self.ckpt.best}.get(which, Path(which))
        st = self.ckpt.load(path, signature=self.signature, map_location="cpu")
        if st is None:
            raise FileNotFoundError(f"No checkpoint at {path}")
        self.gen.load_state_dict(st["gen"])
        self.step = st["step"]
        metrics = self.validate(max_batches=0, kind=f"eval_{Path(path).stem}")
        metrics["checkpoint"] = str(path)
        return metrics

    # ------------------------------------------------------------------ optimisation
    def optimizer_step(self, adv: bool) -> float:
        self.scaler.unscale_(self.opt_g)
        gnorm = torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 10.0)
        self.scaler.step(self.opt_g)
        if adv:
            self.scaler.step(self.opt_d)
        self.scaler.update()
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)
        return float(gnorm)

    def set_disc_grad(self, flag: bool) -> None:
        for p in self.disc.parameters():
            p.requires_grad_(flag)

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp)

    def _loader(self, epoch: int) -> DataLoader:
        self.train_ds.set_epoch(epoch)
        gen = torch.Generator()
        gen.manual_seed(self.cfg.experiment.seed * 1000 + epoch)
        return DataLoader(self.train_ds, batch_size=self.t.batch_size, shuffle=True, generator=gen,
                          num_workers=self.cfg.data.num_workers, collate_fn=self.collate, drop_last=True,
                          pin_memory=self.device.type == "cuda", persistent_workers=False)

    def val_loader(self, dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.t.batch_size, shuffle=False,
                          num_workers=self.cfg.data.num_workers, collate_fn=self.collate)

    # ------------------------------------------------------------------ training loop
    def fit(self, max_steps: Optional[int] = None, resume: bool = True, time_limit_min: Optional[float] = None) -> Dict:
        """Train until ``max_steps`` (or epochs). ``time_limit_min``: stop cleanly (validate + save) once this
        much wall-clock time has passed, so a Colab session keeps time for evaluation; re-run to continue."""
        ensure_not_frozen(self.ckpt.dir)  # never train into the frozen baseline
        if resume:
            self.resume()
        t_start = time.time()
        stopped_by = "max_steps"
        max_steps = max_steps or self.t.max_steps
        per_epoch = len(self.train_ds) // self.t.batch_size // self.t.grad_accum_steps
        if per_epoch == 0:
            raise ValueError(f"Train split ({len(self.train_ds)}) smaller than batch_size*grad_accum_steps")
        total = max_steps or per_epoch * self.t.epochs
        self._print(self.describe_run(per_epoch, total))
        self.gen.train(); self.disc.train()
        t0, accum, acc_losses, n_steps = time.time(), self.t.grad_accum_steps, [], 0
        stop = self.step >= total
        while not stop:
            loader = self._loader(self.epoch)
            for b_idx, batch in enumerate(loader):
                if b_idx < self.batch_in_epoch:  # resume inside an epoch: skip consumed batches
                    continue
                acc_losses.append(self.micro_step(batch, accum))
                self.batch_in_epoch = b_idx + 1
                if len(acc_losses) < accum:
                    continue
                adv = self.adv_active()
                gnorm = self.optimizer_step(adv)
                self.step += 1
                n_steps += 1
                rec = {k: float(np.mean([l[k] for l in acc_losses])) for k in acc_losses[0]}
                rec.update(step=self.step, epoch=self.epoch, grad_norm=gnorm, adv=adv)
                acc_losses = []
                self.history.append(rec)
                if self.step % self.t.log_every == 0 or self.step == 1:
                    self.log.log({"kind": "train", **rec})
                    losses = " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k not in {"step", "epoch", "grad_norm", "adv"})
                    self._print(f"step {self.step}/{total} {losses} ({(time.time() - t0) / n_steps:.2f}s/step)")
                is_best = False
                if self.step % self.t.val_every == 0 or self.step == total:
                    score = self.validate()[self.score_key]
                    is_best = score > self.best
                    if is_best:
                        self.best, self.bad_evals = score, 0
                    else:
                        self.bad_evals += 1
                if self.step % self.t.checkpoint_every == 0 or self.step == total or is_best:
                    self.save(is_best)
                patience = self.t.early_stop_patience
                if patience and self.bad_evals >= patience:
                    self._print(f"early stop: no val {self.score_key} improvement in {self.bad_evals} evaluations")
                    stopped_by = "early_stop"
                    stop = True
                    break
                if self.step >= total:
                    stop = True
                    break
                if time_limit_min is not None and (time.time() - t_start) / 60 >= time_limit_min:
                    self._print(f"time budget of {time_limit_min:.0f} min reached at step {self.step}; "
                                "validating and saving. Re-run with the same settings to continue.")
                    self.validate()
                    stopped_by = "time_limit"
                    stop = True
                    break
            else:
                self.epoch += 1
                self.batch_in_epoch = 0
        self.save()
        return {"step": self.step, f"best_val_{self.score_key}": self.best, "checkpoint": str(self.ckpt.last),
                "stopped_by": stopped_by, "minutes": round((time.time() - t_start) / 60, 1)}

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
