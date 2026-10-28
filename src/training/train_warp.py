"""Training loop for the warp stage (Stage B).

Usage (Colab)::

    python -m src.training.train_warp --config configs/colab_baseline.yaml --max-steps 2000

Re-running the same command after a disconnect resumes from ``checkpoints/<experiment>/warp/last.pt``,
including the exact position inside the current epoch.

The loop, checkpointing, resume and AMP live in :class:`src.training.base.GANTrainer`.
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
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from src.config import load_config, save_config
from src.data.labels import get_label_set
from src.data.warp_dataset import IGNORE, SwapDiagnosticDataset, WarpTripletDataset
from src.evaluation.metrics import ConfusionMatrix
from src.models.discriminator import PatchDiscriminator
from src.models.warp_network import build_warp_generator, warp_channels
from src.training.base import GANTrainer
from src.training.losses import GANLoss, masked_probs, real_maps, segmentation_ce
from src.utils.paths import ProjectPaths


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


def class_weights_from_counts(counts: np.ndarray, mode: str, clip=(0.2, 5.0)) -> np.ndarray:
    """CE class weights from pixel counts (Experiment B). Classes with no pixels get weight 0 (they never occur).

    median_freq: w_c = median(f) / f_c (Eigen & Fergus 2015). sqrt_median_freq: the square root of that, a milder
    version that does not almost switch off the background. Both are clipped to ``clip``.
    """
    counts = np.asarray(counts, np.float64)
    present = counts > 0
    freq = counts / counts.sum()
    w = np.zeros_like(freq)
    w[present] = np.median(freq[present]) / freq[present]
    if mode == "sqrt_median_freq":
        w[present] = np.sqrt(w[present])
    elif mode != "median_freq":
        raise ValueError(f"unknown class-weight mode {mode}")
    w[present] = np.clip(w[present], *clip)
    return w


def train_class_counts(ds, cache_dir: Path) -> np.ndarray:
    """Supervised pixel count per garment class over the training targets (at model size), cached as JSON."""
    import hashlib

    key = hashlib.sha1(("|".join(r.name for r in ds.records) + str(ds.size) + ds.G.name).encode()).hexdigest()[:12]
    path = Path(cache_dir) / "class_counts" / f"warp_train_{key}.json"
    if path.is_file():
        return np.array(json.loads(path.read_text(encoding="utf-8"))["counts"], np.float64)
    counts = np.zeros(ds.G.num_classes, np.float64)
    for rec in ds.records:
        lab = ds._garment(rec).astype(np.int64)
        lab = lab[~np.isin(lab, ds.ignore_ids)]
        counts += np.bincount(lab.ravel(), minlength=ds.G.num_classes)[: ds.G.num_classes]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"counts": counts.tolist(), "classes": ds.G.names, "n_images": len(ds.records)}), encoding="utf-8")
    return counts


METRIC_VERSION = 2  # v2: mIoU over trained classes only; untrained classes masked out of predictions


def _collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("cloth", "body", "target")}
    for k in ("mode", "src", "tgt"):
        out[k] = [b[k] for b in batch]
    return out


class WarpTrainer(GANTrainer):
    stage = "warp"
    score_key = "mean_iou"
    metric_version = METRIC_VERSION

    def build(self, body_source=None) -> None:
        cfg = self.cfg
        self.collate = _collate
        self.G_labels = get_label_set(cfg.labels.garment_label_set)
        self.train_ds = WarpTripletDataset(self.prepared, "train", cfg, train=True, body_source=body_source)
        self.val_ds = WarpTripletDataset(self.prepared, "val", cfg, train=False, body_source=self.train_ds.body)
        self.swap_ds = SwapDiagnosticDataset(self.val_ds, n=self.t.warp_swap_samples, seed=cfg.experiment.seed)
        Kg, Kb = self.G_labels.num_classes, self.train_ds.Kb
        Kc, Kb_cfg, _ = warp_channels(cfg)
        if Kb_cfg != Kb:
            raise ValueError(f"body channels: dataset {Kb} vs config {Kb_cfg}")
        self.gen = build_warp_generator(cfg, Kc, Kb, Kg).to(self.device)
        self.disc = PatchDiscriminator(Kg + Kb, cfg.model.discriminator_base_channels, cfg.model.disc_layers,
                                       cfg.model.spectral_norm).to(self.device)
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.t.warp_lr, betas=tuple(self.t.warp_betas))
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.t.warp_lr, betas=tuple(self.t.warp_betas))
        self.gan = GANLoss(self.t.warp_gan_mode)
        self.signature = {"generator": self.gen.cfg, "discriminator": self.disc.cfg,
                          "garment_label_set": cfg.labels.garment_label_set,
                          "body_label_set": cfg.labels.body_label_set, "body_backend": cfg.segmentation.body_backend,
                          "transfer_classes": list(cfg.labels.transfer_classes)}
        if cfg.model.warp_extra_body != "none" or cfg.model.warp_ref_body:
            self.signature["warp_inputs"] = {"extra_body": cfg.model.warp_extra_body, "ref_body": cfg.model.warp_ref_body}
        # Classes the warp is never trained on (IGNORE in every target): hair, face, headwear, accessory.
        self.untrained_ids = list(self.train_ds.ignore_ids)
        # mIoU = mean over the trained classes only (transfer classes + skin). Including untrained classes
        # added spurious IoU-0 entries whenever a stray pixel was predicted as one of them (metric v1 bug).
        self.miou_exclude = [0] + self.untrained_ids
        # Experiment B: class-weighted CE for training only (validation CE stays unweighted, so it is comparable)
        self.ce_weight = None
        if self.t.warp_class_weights != "none":
            counts = train_class_counts(self.train_ds, Path(cfg.paths.cache_dir))
            w = class_weights_from_counts(counts, self.t.warp_class_weights)
            self.ce_weight = torch.tensor(w, dtype=torch.float32, device=self.device)
            self.signature["class_weights"] = self.t.warp_class_weights
            self._print("CE class weights: " + " ".join(f"{n} {v:.2f}" for n, v in zip(self.G_labels.names, w) if v > 0))

    def describe_run(self, per_epoch: int, total: int) -> str:
        return (f"training warp: {len(self.train_ds)} triplets ({self.train_ds.n_cross_view_available()} with a "
                f"cross-view partner), {per_epoch} steps/epoch, target {total} steps, device {self.device}, amp {self.use_amp}")

    # ---------------------------------------------------------------- core step
    def adv_active(self) -> bool:
        return self.t.lambda_adv_warp > 0 and self.step >= self.t.warp_adv_start_step

    def micro_step(self, batch: Dict, accum: int) -> Dict[str, float]:
        cloth = batch["cloth"].to(self.device, non_blocking=True)
        body = batch["body"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        adv = self.adv_active()
        self.set_disc_grad(False)
        with self.autocast():
            logits = self.gen(cloth, body)
            ce = segmentation_ce(logits, target, weight=self.ce_weight)
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
            self.set_disc_grad(True)
            with self.autocast():
                real = real_maps(target, logits.shape[1], self.t.warp_real_smoothing).to(fake.dtype)
                d_loss = self.gan.d_loss(self.disc(torch.cat([real, body.to(real.dtype)], 1)),
                                         self.disc(torch.cat([fake.detach(), body.to(fake.dtype)], 1)))
            self.scaler.scale(d_loss / accum).backward()
        return {"ce": ce.item(), "g_adv": g_adv.item(), "d": d_loss.item()}

    # ---------------------------------------------------------------- validation
    @torch.no_grad()
    def predict(self, cloth: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        self.gen.eval()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            logits = self.gen(cloth.to(self.device), body.to(self.device))
        self.gen.train()
        return logits.float()

    def to_labels(self, logits: torch.Tensor) -> np.ndarray:
        """Argmax over trained classes only. Untrained classes (hair, face, headwear, accessory) never
        receive a training signal, and at try-on time face/hair come from the target (paper Sec. 4.2)."""
        logits = logits.clone()
        logits[:, self.untrained_ids] = -torch.inf
        return logits.argmax(1).cpu().numpy()

    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None, save_grid: bool = True, kind: str = "val") -> Dict:
        """``max_batches``: None -> training.val_max_batches; 0 -> the whole validation split."""
        K = self.G_labels.num_classes
        cm = ConfusionMatrix(K, ignore_index=IGNORE)
        ces, first = [], None
        loader = self.val_loader(self.val_ds)
        limit = self.t.val_max_batches if max_batches is None else (None if max_batches == 0 else max_batches)
        n_images = 0
        for i, batch in enumerate(loader):
            if limit is not None and i >= limit:
                break
            logits = self.predict(batch["cloth"], batch["body"])
            target = batch["target"].to(self.device)
            ces.append(float(segmentation_ce(logits, target)))
            pred = self.to_labels(logits)
            cm.update(pred, batch["target"].numpy())
            n_images += pred.shape[0]
            if first is None:
                first = (batch, pred)
        summary = cm.summary(self.G_labels.names, exclude=self.miou_exclude)
        fidelity, area, swap_vis = self.swap_fidelity()
        metrics = {"kind": kind, "step": self.step, "n_images": n_images, "metric_version": METRIC_VERSION,
                   "ce": float(np.mean(ces)), "pixel_accuracy": summary["pixel_accuracy"],
                   "mean_iou": summary["mean_iou"], "per_class_iou": summary["per_class_iou"], "class_fidelity": fidelity,
                   "class_area_agreement": area}
        self.log.log(metrics)
        self._print(f"[{kind}] step {self.step} ({n_images} images): ce {metrics['ce']:.4f} acc {metrics['pixel_accuracy']:.4f} "
                    f"mIoU {metrics['mean_iou']:.4f} | swap: class_fidelity {fidelity:.3f} area_agreement {area:.3f}")
        excluded = {self.G_labels.names[i] for i in self.miou_exclude}
        per_class = {k: v for k, v in summary["per_class_iou"].items() if v is not None and k not in excluded}
        self._print("      per-class IoU: " + " ".join(f"{k} {v:.2f}" for k, v in per_class.items()))
        if save_grid and first is not None:
            self.save_grid(first[0], first[1], swap_vis, self.out_dir / f"{kind}_step{self.step:07d}.png")
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
        loader = self.val_loader(self.swap_ds)
        for batch in loader:
            pred = self.to_labels(self.predict(batch["cloth"], batch["body"]))
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
                cloth = src["cloth"][b][: G.num_classes].argmax(0).numpy()
                body = src["body"][b][: Bl.num_classes].argmax(0).numpy()
                tiles = [gcol(cloth), Bl.palette_array()[body], gcol(prd[b])]
                tiles.append(gcol(src["target"][b].numpy()) if title == "val" else np.full_like(tiles[0], 255))
                rows.append(np.concatenate(tiles, 1))
        Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path)
        return path

def main():
    ap = argparse.ArgumentParser(description="Train the Stage-B warp network")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--evaluate", nargs="+", default=None, metavar="CKPT",
                    help="only evaluate checkpoints (last, best or a path) on the whole val split; no training")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    paths = ProjectPaths.from_config(cfg).ensure()
    save_config(cfg, paths.experiment_dir("outputs", cfg.experiment.name) / "config_warp.yaml")
    trainer = WarpTrainer(cfg)
    if args.evaluate:
        for which in args.evaluate:
            m = trainer.evaluate_checkpoint(which)
            print(json.dumps({k: m[k] for k in ("checkpoint", "step", "n_images", "mean_iou", "class_area_agreement")}, indent=1))
        return
    print(json.dumps(trainer.fit(max_steps=args.max_steps, resume=not args.no_resume), indent=1))


if __name__ == "__main__":
    main()
