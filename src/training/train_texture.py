"""Training loop for the texture stage (Stage C): alignment network + SPADE generator.

Usage (Colab)::

    python -m src.training.train_texture --config configs/colab_baseline.yaml --max-steps 2000
    python -m src.training.train_texture --config configs/colab_baseline.yaml --evaluate best last

Shared loop / checkpoints / exact resume / AMP: :class:`src.training.base.GANTrainer`
(checkpoints in ``checkpoints/<experiment>/texture/``).

Generator objective (weights in ``training.lambda_tex_*``, docs/texture_stage.md):
    hinge adversarial (Eq. 9, multi-scale)      lambda_tex_adv
  + D feature matching                           lambda_tex_fm
  + VGG-19 perceptual vs ground truth            lambda_tex_vgg
  + L1(warped exemplar I_S, ground truth at correspondence size)   lambda_tex_warp
  + cycle regularisation L1(back-warped I_S, exemplar)             lambda_tex_cycle
  + domain alignment L1(F_x(layout_t), F_y(image_t))               lambda_tex_domain
  + optional pixel L1                                              lambda_tex_l1
Discriminator: multi-scale hinge (Eq. 8) on concat(image, layout).
Validation: L1 and SSIM against the real target photo on cross-view val pairs (best.pt = highest SSIM).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import load_config, save_config
from src.data.labels import get_label_set
from src.data.preprocessing import tensor_to_image
from src.data.texture_dataset import TextureDataset, TextureSwapDataset, texture_collate
from src.evaluation.metrics import ssim
from src.models.discriminator import MultiScaleDiscriminator
from src.models.spade_generator import build_texture_generator
from src.training.base import GANTrainer
from src.training.losses import VGGPerceptualLoss, feature_matching, multiscale_hinge_d, multiscale_hinge_g
from src.utils.paths import ProjectPaths


class TextureTrainer(GANTrainer):
    stage = "texture"
    score_key = "ssim"
    metric_version = 1

    def build(self, vgg: Optional[VGGPerceptualLoss] = None) -> None:
        cfg, t = self.cfg, self.t
        self.collate = texture_collate
        self.G_labels = get_label_set(cfg.labels.garment_label_set)
        K = self.G_labels.num_classes
        self.train_ds = TextureDataset(self.prepared, "train", cfg, train=True)
        # validation: real pose change only (pseudo pairs would make SSIM look better than try-on reality)
        self.val_ds = TextureDataset(self.prepared, "val", cfg, train=False, cross_view_prob=1.0)
        self.swap_ds = TextureSwapDataset(self.val_ds, n=8, seed=cfg.experiment.seed)
        self.gen = build_texture_generator(cfg, K).to(self.device)
        self.disc = MultiScaleDiscriminator(3 + K, cfg.model.discriminator_base_channels, cfg.model.disc_layers,
                                            cfg.model.disc_scales, cfg.model.spectral_norm).to(self.device)
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=t.lr_generator, betas=tuple(t.betas))  # [P] Sec. 5.1
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=t.lr_discriminator, betas=tuple(t.betas))
        self.use_vgg = t.lambda_tex_vgg > 0
        if self.use_vgg:
            self.vgg = (vgg or VGGPerceptualLoss(t.texture_vgg_init, Path(cfg.paths.pretrained_dir) / "torch",
                                                 t.allow_vgg_download)).to(self.device)
        self.signature = {"generator": self.gen.cfg, "discriminator": self.disc.cfg,
                          "garment_label_set": cfg.labels.garment_label_set, "vgg_init": t.texture_vgg_init}

    def describe_run(self, per_epoch: int, total: int) -> str:
        return (f"training texture: {len(self.train_ds)} targets ({self.train_ds.n_cross_view_available()} with a same-item "
                f"exemplar), {per_epoch} steps/epoch, target {total} steps, device {self.device}, amp {self.use_amp}")

    def adv_active(self) -> bool:
        return self.t.lambda_tex_adv > 0 and self.step >= self.t.texture_adv_start_step

    def _batch(self, batch):
        dev = self.device
        return (batch["layout"].to(dev, non_blocking=True), batch["image"].to(dev, non_blocking=True),
                batch["exemplar"].to(dev, non_blocking=True), batch["exemplar_parsing"].to(dev, non_blocking=True))

    def micro_step(self, batch: Dict, accum: int) -> Dict[str, float]:
        t = self.t
        layout, image, ex, ex_p = self._batch(batch)
        adv = self.adv_active()
        self.set_disc_grad(False)
        with self.autocast():
            out = self.gen(layout, ex, ex_p)
            fake = out["fake"]
            losses = {}
            if adv:
                fake_outs = self.disc(torch.cat([fake, layout], 1))
                with torch.no_grad():
                    real_outs = self.disc(torch.cat([image, layout], 1))
                losses["g_adv"] = multiscale_hinge_g(fake_outs)
                if t.lambda_tex_fm > 0:
                    losses["fm"] = feature_matching(real_outs, fake_outs)
            if self.use_vgg:
                losses["vgg"] = self.vgg(fake, image)
            small = out["warped"].shape[-2:]
            if t.lambda_tex_warp > 0:
                losses["warp"] = F.l1_loss(out["warped"], F.adaptive_avg_pool2d(image.float(), small))
            if t.lambda_tex_cycle > 0:
                losses["cycle"] = F.l1_loss(out["cycle"], out["exemplar_small"])
            if t.lambda_tex_domain > 0:
                fy_t = self.gen.align.encode_y(self.gen.exemplar_input(image, layout))
                losses["domain"] = F.l1_loss(out["feat_x"].float(), fy_t.float())
            if t.lambda_tex_l1 > 0:
                losses["l1"] = F.l1_loss(fake.float(), image.float())
            weights = {"g_adv": t.lambda_tex_adv, "fm": t.lambda_tex_fm, "vgg": t.lambda_tex_vgg, "warp": t.lambda_tex_warp,
                       "cycle": t.lambda_tex_cycle, "domain": t.lambda_tex_domain, "l1": t.lambda_tex_l1}
            loss_g = sum(weights[k] * v for k, v in losses.items())
        if not torch.isfinite(loss_g):
            raise FloatingPointError(f"non-finite texture loss at step {self.step}: { {k: v.item() for k, v in losses.items()} }")
        self.scaler.scale(loss_g / accum).backward()
        rec = {k: v.item() for k, v in losses.items()}
        rec["g_total"] = loss_g.item()
        rec["d"] = 0.0
        if adv:
            self.set_disc_grad(True)
            with self.autocast():
                d_loss = multiscale_hinge_d(self.disc(torch.cat([image, layout], 1)),
                                            self.disc(torch.cat([fake.detach(), layout], 1)))
            self.scaler.scale(d_loss / accum).backward()
            rec["d"] = d_loss.item()
        for k in ("g_adv", "fm", "vgg", "warp", "cycle", "domain", "l1"):
            rec.setdefault(k, 0.0)  # constant keys so the loop can average over accumulation steps
        return rec

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def generate(self, batch: Dict) -> Dict[str, torch.Tensor]:
        layout, _, ex, ex_p = self._batch(batch)
        self.gen.eval()
        with self.autocast():
            out = self.gen(layout, ex, ex_p)
        self.gen.train()
        return {k: v.float() for k, v in out.items() if torch.is_tensor(v)}

    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None, save_grid: bool = True, kind: str = "val") -> Dict:
        limit = self.t.val_max_batches if max_batches is None else (None if max_batches == 0 else max_batches)
        l1s, ssims, wl1s, first, n = [], [], [], None, 0
        for i, batch in enumerate(self.val_loader(self.val_ds)):
            if limit is not None and i >= limit:
                break
            out = self.generate(batch)
            real = batch["image"].to(self.device)
            fake01, real01 = (out["fake"] + 1) / 2, (real + 1) / 2
            l1s += (fake01 - real01).abs().flatten(1).mean(1).tolist()
            ssims += ssim(fake01.clamp(0, 1), real01).tolist()
            small = F.adaptive_avg_pool2d(real01, out["warped"].shape[-2:])
            wl1s += (((out["warped"] + 1) / 2) - small).abs().flatten(1).mean(1).tolist()
            n += real.shape[0]
            if first is None:
                first = (batch, out)
        metrics = {"kind": kind, "step": self.step, "n_images": n, "metric_version": self.metric_version,
                   "ssim": float(np.mean(ssims)), "l1": float(np.mean(l1s)), "warp_l1": float(np.mean(wl1s))}
        self.log.log(metrics)
        self._print(f"[{kind}] step {self.step} ({n} cross-view pairs): SSIM {metrics['ssim']:.4f} L1 {metrics['l1']:.4f} "
                    f"| warped-exemplar L1 {metrics['warp_l1']:.4f}")
        if save_grid and first is not None:
            swap = None
            if len(self.swap_ds):
                sb = texture_collate([self.swap_ds[k] for k in range(min(4, len(self.swap_ds)))])
                swap = (sb, self.generate(sb))
            self.save_grid(first, swap, self.out_dir / f"{kind}_step{self.step:07d}.png")
        return metrics

    def save_grid(self, val, swap, path: Path, n: int = 4) -> Path:
        """Columns: exemplar | target layout | warped exemplar I_S (upsampled) | generated | ground truth (white for swap)."""
        G = self.G_labels
        rows = []
        for group in (val, swap):
            if group is None:
                continue
            batch, out = group
            for b in range(min(n, batch["image"].shape[0])):
                h, w = batch["image"].shape[-2:]
                warped = F.interpolate(out["warped"][b:b + 1].cpu(), size=(h, w), mode="nearest")[0]
                tiles = [tensor_to_image(batch["exemplar"][b]), G.palette_array()[batch["layout"][b].argmax(0).numpy()],
                         tensor_to_image(warped), tensor_to_image(out["fake"][b].cpu())]
                tiles.append(tensor_to_image(batch["image"][b]) if batch["mode"][b] != "swap" else np.full_like(tiles[0], 255))
                rows.append(np.concatenate(tiles, 1))
        Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path)
        return path


def main():
    ap = argparse.ArgumentParser(description="Train the Stage-C texture network")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--evaluate", nargs="+", default=None, metavar="CKPT")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    paths = ProjectPaths.from_config(cfg).ensure()
    save_config(cfg, paths.experiment_dir("outputs", cfg.experiment.name) / "config_texture.yaml")
    trainer = TextureTrainer(cfg)
    if args.evaluate:
        for which in args.evaluate:
            m = trainer.evaluate_checkpoint(which)
            print(json.dumps({k: m[k] for k in ("checkpoint", "step", "n_images", "ssim", "l1", "warp_l1")}, indent=1))
        return
    print(json.dumps(trainer.fit(max_steps=args.max_steps, resume=not args.no_resume), indent=1))


if __name__ == "__main__":
    main()
