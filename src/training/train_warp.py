"""Training loop for the warp stage (Stage B).

Usage (Colab)::

    python -m src.training.train_warp --config configs/colab_baseline.yaml --max-steps 2000

Re-running the same command after a disconnect resumes from ``checkpoints/<experiment>/warp/last.pt``,
including the exact position inside the current epoch.

Per optimiser step (``grad_accum_steps`` micro-batches):
  1. logits = G(C_seg, B_seg);  L_G = CE(logits, target) + lambda_adv * L_G_adv  (adv after warp_adv_start_step)
  2. L_D on (real one-hot target, detached softmax(logits)), both concatenated with B_seg
  3. AMP (fp16 autocast + GradScaler) only on CUDA; losses are computed in fp32.

Validation (every ``val_every`` steps): CE, pixel accuracy and mIoU over supervised pixels (background
excluded from mIoU), the swap-diagnostic ``class_fidelity``, and a PNG grid of predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import ProjectConfig, load_config, save_config
from src.data.dataset import PreparedData, prepare_dataset
from src.data.labels import get_label_set
from src.data.warp_dataset import IGNORE, SwapDiagnosticDataset, WarpTripletDataset
from src.evaluation.metrics import ConfusionMatrix
from src.models.discriminator import PatchDiscriminator
from src.models.warp_network import build_warp_generator
from src.training.checkpointing import CheckpointManager, rng_state, set_rng_state
from src.training.losses import GANLoss, masked_probs, real_maps, segmentation_ce
from src.utils.device import select_device
from src.utils.logging import JsonlLogger
from src.utils.paths import ProjectPaths
from src.utils.reproducibility import seed_everything


def class_area_agreement(input_cloth: np.ndarray, pred: np.ndarray, transfer_ids) -> float:
    """Histogram intersection of garment-class *areas*: input C_seg vs prediction (one swap pair).

    ``input_cloth`` [K, H, W] one-hot-like, ``pred`` [H, W] class ids. Each side is reduced to the
    share of garment pixels per transfer class; the score is sum(min(p_in, p_pred)) in [0, 1].
    1 = the prediction contains the same garment mix as the input; a skirt turned into trousers
    scores low even if a few trouser pixels were in the input (which ``class_fidelity`` would accept).
    Areas differ legitimately between bodies, so this is a proxy, not an accuracy.
    """
    a_in = np.array([input_cloth[k].sum() for k in transfer_ids], dtype=np.float64)
    a_pr = np.array([(pred == k).sum() for k in transfer_ids], dtype=np.float64)
    if a_in.sum() == 0 or a_pr.sum() == 0:
        return float("nan") if a_in.sum() == 0 else 0.0
    return float(np.minimum(a_in / a_in.sum(), a_pr / a_pr.sum()).sum())


def _collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("cloth", "body", "target")}
    for k in ("mode", "src", "tgt"):
        out[k] = [b[k] for b in batch]
    return out


class WarpTrainer:
    def __init__(self, cfg: ProjectConfig, prepared: Optional[PreparedData] = None, device: Optional[torch.device] = None,
                 body_source=None, verbose: bool = True):
        self.cfg, self.t = cfg, cfg.training
        self.verbose = verbose
        seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
        self.device = device or select_device(cfg.experiment.device)
        self.prepared = prepared or prepare_dataset(cfg)
        self.G_labels = get_label_set(cfg.labels.garment_label_set)
        self.train_ds = WarpTripletDataset(self.prepared, "train", cfg, train=True, body_source=body_source)
        self.val_ds = WarpTripletDataset(self.prepared, "val", cfg, train=False, body_source=self.train_ds.body)
        self.swap_ds = SwapDiagnosticDataset(self.val_ds, n=self.t.warp_swap_samples, seed=cfg.experiment.seed)
        Kg, Kb = self.G_labels.num_classes, self.train_ds.Kb
        self.gen = build_warp_generator(cfg, Kg, Kb, Kg).to(self.device)
        self.disc = PatchDiscriminator(Kg + Kb, cfg.model.discriminator_base_channels, cfg.model.disc_layers,
                                       cfg.model.spectral_norm).to(self.device)
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.t.warp_lr, betas=tuple(self.t.warp_betas))
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.t.warp_lr, betas=tuple(self.t.warp_betas))
        self.use_amp = bool(self.t.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.gan = GANLoss(self.t.warp_gan_mode)
        paths = ProjectPaths.from_config(cfg)
        self.ckpt = CheckpointManager(paths.checkpoints / cfg.experiment.name / "warp")
        self.out_dir = paths.outputs / cfg.experiment.name / "warp"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = JsonlLogger(paths.logs / cfg.experiment.name / "warp_train.jsonl")
        self.signature = {"generator": self.gen.cfg, "discriminator": self.disc.cfg,
                          "garment_label_set": cfg.labels.garment_label_set,
                          "body_label_set": cfg.labels.body_label_set, "body_backend": cfg.segmentation.body_backend,
                          "transfer_classes": list(cfg.labels.transfer_classes)}
        self.step = self.epoch = self.batch_in_epoch = 0
        self.best = -math.inf
        self.bad_evals = 0
        self.history = []

    # ---------------------------------------------------------------- checkpointing
    def state(self) -> Dict:
        return {"gen": self.gen.state_dict(), "disc": self.disc.state_dict(), "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(), "scaler": self.scaler.state_dict(), "step": self.step,
                "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch, "best": self.best,
                "bad_evals": self.bad_evals, "rng": rng_state(), "signature": self.signature,
                "config": json.loads(json.dumps(self.cfg.to_dict()))}

    def save(self, is_best: bool = False) -> None:
        self.ckpt.save(self.state(), self.step, is_best=is_best)

    def resume(self, path: Optional[Path] = None) -> bool:
        st = self.ckpt.load(path, signature=self.signature, map_location=self.device)
        if st is None:
            return False
        self.gen.load_state_dict(st["gen"])
        self.disc.load_state_dict(st["disc"])
        self.opt_g.load_state_dict(st["opt_g"])
        self.opt_d.load_state_dict(st["opt_d"])
        self.scaler.load_state_dict(st["scaler"])
        self.step, self.epoch, self.batch_in_epoch = st["step"], st["epoch"], st["batch_in_epoch"]
        self.best, self.bad_evals = st["best"], st["bad_evals"]
        set_rng_state(st["rng"])
        self._print(f"resumed from step {self.step} (epoch {self.epoch}, batch {self.batch_in_epoch})")
        return True

    # ---------------------------------------------------------------- core step
    def _adv_active(self) -> bool:
        return self.t.lambda_adv_warp > 0 and self.step >= self.t.warp_adv_start_step

    def micro_step(self, batch: Dict, accum: int) -> Dict[str, float]:
        cloth = batch["cloth"].to(self.device, non_blocking=True)
        body = batch["body"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        adv = self._adv_active()
        for p in self.disc.parameters():
            p.requires_grad_(False)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            logits = self.gen(cloth, body)
            ce = segmentation_ce(logits, target)
            g_adv = torch.zeros((), device=self.device)
            if adv:
                fake = masked_probs(logits, target)
                g_adv = self.gan.g_loss(self.disc(torch.cat([fake, body.to(fake.dtype)], 1)))
            loss_g = ce + self.t.lambda_adv_warp * g_adv
        if not torch.isfinite(loss_g):
            raise FloatingPointError(f"non-finite generator loss at step {self.step}: ce={ce.item()} adv={g_adv.item()}")
        self.scaler.scale(loss_g / accum).backward()
        d_loss = torch.zeros(())
        if adv:
            for p in self.disc.parameters():
                p.requires_grad_(True)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                real = real_maps(target, logits.shape[1], self.t.warp_real_smoothing).to(fake.dtype)
                d_loss = self.gan.d_loss(self.disc(torch.cat([real, body.to(real.dtype)], 1)),
                                         self.disc(torch.cat([fake.detach(), body.to(fake.dtype)], 1)))
            self.scaler.scale(d_loss / accum).backward()
        return {"ce": ce.item(), "g_adv": g_adv.item(), "d": d_loss.item()}

    def _optimizer_step(self, adv: bool) -> None:
        self.scaler.unscale_(self.opt_g)
        gnorm = torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 10.0)
        self.scaler.step(self.opt_g)
        if adv:
            self.scaler.step(self.opt_d)
        self.scaler.update()
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)
        return float(gnorm)

    def _loader(self, epoch: int) -> DataLoader:
        self.train_ds.set_epoch(epoch)
        gen = torch.Generator()
        gen.manual_seed(self.cfg.experiment.seed * 1000 + epoch)
        return DataLoader(self.train_ds, batch_size=self.t.batch_size, shuffle=True, generator=gen,
                          num_workers=self.cfg.data.num_workers, collate_fn=_collate, drop_last=True,
                          pin_memory=self.device.type == "cuda", persistent_workers=False)

    # ---------------------------------------------------------------- training
    def fit(self, max_steps: Optional[int] = None, resume: bool = True) -> Dict:
        if resume:
            self.resume()
        max_steps = max_steps or self.t.max_steps
        per_epoch = len(self.train_ds) // self.t.batch_size // self.t.grad_accum_steps
        if per_epoch == 0:
            raise ValueError(f"Train split ({len(self.train_ds)}) smaller than batch_size*grad_accum_steps")
        total = max_steps or per_epoch * self.t.epochs
        self._print(f"training warp: {len(self.train_ds)} triplets ({self.train_ds.n_cross_view_available()} with a "
                    f"cross-view partner), {per_epoch} steps/epoch, target {total} steps, device {self.device}, amp {self.use_amp}")
        self.gen.train(); self.disc.train()
        t0, accum, acc_losses = time.time(), self.t.grad_accum_steps, []
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
                adv = self._adv_active()
                gnorm = self._optimizer_step(adv)
                self.step += 1
                rec = {k: float(np.mean([l[k] for l in acc_losses])) for k in acc_losses[0]}
                rec.update(step=self.step, epoch=self.epoch, grad_norm=gnorm, adv=adv)
                acc_losses = []
                self.history.append(rec)
                if self.step % self.t.log_every == 0 or self.step == 1:
                    self.log.log({"kind": "train", **rec})
                    self._print(f"step {self.step}/{total} ce {rec['ce']:.4f} g_adv {rec['g_adv']:.3f} d {rec['d']:.3f} "
                                f"({(time.time() - t0) / max(1, len(self.history)):.2f}s/step)")
                is_best = False
                if self.step % self.t.val_every == 0 or self.step == total:
                    metrics = self.validate()
                    is_best = metrics["mean_iou"] > self.best
                    if is_best:
                        self.best, self.bad_evals = metrics["mean_iou"], 0
                    else:
                        self.bad_evals += 1
                if self.step % self.t.checkpoint_every == 0 or self.step == total or is_best:
                    self.save(is_best)
                if self.step >= total or (self.t.early_stop_patience and self.bad_evals >= self.t.early_stop_patience):
                    if self.bad_evals and self.t.early_stop_patience and self.bad_evals >= self.t.early_stop_patience:
                        self._print(f"early stop: no val mIoU improvement in {self.bad_evals} evaluations")
                    stop = True
                    break
            else:
                self.epoch += 1
                self.batch_in_epoch = 0
        self.save()
        return {"step": self.step, "best_val_miou": self.best, "checkpoint": str(self.ckpt.last)}

    # ---------------------------------------------------------------- validation
    @torch.no_grad()
    def predict(self, cloth: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        self.gen.eval()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            logits = self.gen(cloth.to(self.device), body.to(self.device))
        self.gen.train()
        return logits.float()

    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None, save_grid: bool = True) -> Dict:
        K = self.G_labels.num_classes
        cm = ConfusionMatrix(K, ignore_index=IGNORE)
        ces, first = [], None
        loader = DataLoader(self.val_ds, batch_size=self.t.batch_size, shuffle=False, num_workers=self.cfg.data.num_workers,
                            collate_fn=_collate)
        limit = max_batches if max_batches is not None else self.t.val_max_batches
        for i, batch in enumerate(loader):
            if limit is not None and i >= limit:
                break
            logits = self.predict(batch["cloth"], batch["body"])
            target = batch["target"].to(self.device)
            ces.append(float(segmentation_ce(logits, target)))
            pred = logits.argmax(1).cpu().numpy()
            cm.update(pred, batch["target"].numpy())
            if first is None:
                first = (batch, pred)
        summary = cm.summary(self.G_labels.names, exclude=[0])
        fidelity, area, swap_vis = self.swap_fidelity()
        metrics = {"kind": "val", "step": self.step, "ce": float(np.mean(ces)), "pixel_accuracy": summary["pixel_accuracy"],
                   "mean_iou": summary["mean_iou"], "per_class_iou": summary["per_class_iou"], "class_fidelity": fidelity,
                   "class_area_agreement": area}
        self.log.log(metrics)
        self._print(f"[val] step {self.step}: ce {metrics['ce']:.4f} acc {metrics['pixel_accuracy']:.4f} "
                    f"mIoU {metrics['mean_iou']:.4f} | swap: class_fidelity {fidelity:.3f} area_agreement {area:.3f}")
        per_class = {k: v for k, v in summary["per_class_iou"].items() if v is not None and k != "background"}
        self._print("      per-class IoU: " + " ".join(f"{k} {v:.2f}" for k, v in per_class.items()))
        if save_grid and first is not None:
            self.save_grid(first[0], first[1], swap_vis, self.out_dir / f"val_step{self.step:07d}.png")
        return metrics

    @torch.no_grad()
    def swap_fidelity(self):
        """Swap pairs (garment from one person, body from another; no ground truth).

        Returns (class_fidelity, class_area_agreement, visualisation batch):
        * class_fidelity: share of predicted garment pixels whose class occurs anywhere in the input (lenient)
        * class_area_agreement: mean histogram intersection of garment-class areas (stricter, see function)
        """
        if len(self.swap_ds) == 0:
            return float("nan"), float("nan"), None
        transfer = self.G_labels.indices(self.cfg.labels.transfer_classes)
        hits = total = 0
        areas = []
        vis = None
        loader = DataLoader(self.swap_ds, batch_size=self.t.batch_size, shuffle=False, collate_fn=_collate)
        for batch in loader:
            pred = self.predict(batch["cloth"], batch["body"]).argmax(1).cpu().numpy()
            present = batch["cloth"].flatten(2).amax(2).numpy() > 0  # [B, K] classes in the input
            for b in range(pred.shape[0]):
                garment = np.isin(pred[b], transfer)
                total += int(garment.sum())
                hits += int((garment & present[b][pred[b]]).sum())
                areas.append(class_area_agreement(batch["cloth"][b].numpy(), pred[b], transfer))
            if vis is None:
                vis = (batch, pred)
        areas = [a for a in areas if not np.isnan(a)]
        return (hits / total if total else float("nan")), (float(np.mean(areas)) if areas else float("nan")), vis

    def save_grid(self, batch, pred, swap_vis, path: Path, n: int = 4) -> Path:
        G = self.G_labels
        Bl = get_label_set(self.cfg.labels.body_label_set)

        def gcol(lab):
            lab = lab.copy()
            ign = lab == IGNORE
            lab[ign] = 0
            img = G.palette_array()[lab]
            img[ign] = (128, 128, 128)
            return img

        rows = []
        for src, prd, title in ((batch, pred, "val"), (swap_vis[0], swap_vis[1], "swap") if swap_vis else (None, None, None)):
            if src is None:
                continue
            for b in range(min(n, prd.shape[0])):
                cloth = src["cloth"][b].argmax(0).numpy()
                body = src["body"][b].argmax(0).numpy()
                tiles = [gcol(cloth), Bl.palette_array()[body], gcol(prd[b])]
                tiles.append(gcol(src["target"][b].numpy()) if title == "val" else np.full_like(tiles[0], 255))
                rows.append(np.concatenate(tiles, 1))
        Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path)
        return path

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Train the Stage-B warp network")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    paths = ProjectPaths.from_config(cfg).ensure()
    save_config(cfg, paths.experiment_dir("outputs", cfg.experiment.name) / "config_warp.yaml")
    trainer = WarpTrainer(cfg)
    print(json.dumps(trainer.fit(max_steps=args.max_steps, resume=not args.no_resume), indent=1))


if __name__ == "__main__":
    main()
