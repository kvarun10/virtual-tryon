"""Regenerate every chart in report/figures/ from the CSV/JSON files in report/data/.

    python report/make_charts.py

Charts contain only measured numbers (no dataset images), so they can be committed and used in the report.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
CLASSES = ["skin", "top", "outer", "dress", "skirt", "pants", "footwear"]
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def rows(name):
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def warp_metric_bug():
    r = [x for x in rows("warp_val_log_10k.csv") if x["skin"]]
    steps = [int(x["step"]) for x in r]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(steps, [f(x["miou_reported_v1"]) for x in r], "o--", color="#c0392b", label="reported (v1: averaged over 7-11 classes)")
    ax.plot(steps, [f(x["miou_trained7_v2_recomputed"]) for x in r], "o-", color="#1f77b4", label="corrected (v2: 7 trained classes)")
    for x, s in zip(r, steps):
        ax.annotate(x["n_classes_in_v1_mean"], (s, f(x["miou_reported_v1"])), textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=7, color="#c0392b")
    ax.set_xlabel("training step"); ax.set_ylabel("val mIoU (400 images)")
    ax.set_title("Warp validation mIoU: metric bug vs corrected definition\n(red numbers = classes in the v1 average)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    save(fig, "warp_miou_metric_v1_vs_v2.png")


def warp_per_class_over_steps():
    r = [x for x in rows("warp_val_log_10k.csv") if x["skin"]]
    steps = [int(x["step"]) for x in r]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for c in CLASSES:
        ax.plot(steps, [f(x[c]) for x in r], "o-", ms=3, label=c)
    ax.set_xlabel("training step"); ax.set_ylabel("val IoU"); ax.set_title("Warp per-class IoU during training (400 val images)")
    ax.legend(ncol=4, fontsize=7, loc="lower right")
    save(fig, "warp_per_class_iou_over_steps.png")


def warp_eval_bars():
    r = rows("warp_full_val_eval.csv")
    fig, ax = plt.subplots(figsize=(6, 3))
    w = 0.38
    for k, (x, col) in enumerate(zip(r, ["#bbbbbb", "#1f77b4"])):
        xs = [i + (k - 0.5) * w for i in range(len(CLASSES))]
        ax.bar(xs, [f(x[c]) for c in CLASSES], w, color=col, label=f"step {x['step']} (mIoU {f(x['miou_trained7']):.3f})")
    ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_ylim(0, 1); ax.set_ylabel("IoU")
    ax.set_title("Warp per-class IoU on the full validation split (1,271 images)")
    ax.legend(fontsize=8)
    save(fig, "warp_full_val_per_class_best_vs_last.png")


def warp_losses():
    r = rows("warp_train_log_10k.csv")
    s = [int(x["step"]) for x in r]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(s, [f(x["ce"]) for x in r], color="#1f77b4")
    ax[0].axvline(500, color="k", ls=":", lw=1); ax[0].text(560, 2.2, "GAN on", fontsize=7)
    ax[0].set_title("cross-entropy (train)"); ax[0].set_xlabel("step")
    g = [x for x in r if x["source"] == "run2_gan"]
    ax[1].plot([int(x["step"]) for x in g], [f(x["g_adv"]) for x in g], label="g_adv", color="#ff7f0e")
    ax[1].plot([int(x["step"]) for x in g], [f(x["d"]) for x in g], label="d", color="#2ca02c")
    ax[1].set_title("adversarial losses (LSGAN, λ=0.1)"); ax[1].set_xlabel("step"); ax[1].legend(fontsize=8)
    save(fig, "warp_training_losses.png")


def warp_swap():
    r = [x for x in rows("warp_val_log_10k.csv") if x["swap_area_agreement"]]
    s = [int(x["step"]) for x in r]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(s, [f(x["swap_class_fidelity"]) for x in r], "x-", label="class fidelity (lenient)")
    ax.plot(s, [f(x["swap_area_agreement"]) for x in r], "s-", label="area agreement (strict)")
    ax.set_ylim(0.6, 1.0); ax.set_xlabel("training step")
    ax.set_title("Swap test: does the prediction follow the input garment?\n(garment from one person, body from another)")
    ax.legend(fontsize=8)
    save(fig, "warp_swap_test_metrics.png")


def silhouette_leak():
    r = rows("body_silhouette_leak.csv")
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [x["description"] for x in r]
    ax.barh(labels, [f(x["iou_ppp_person_vs_lip_person"]) for x in r], color=["#1f77b4"] * 3 + ["#c0392b"] + ["#1f77b4"])
    ax.set_xlim(0, 1); ax.set_xlabel("IoU of SCHP-Pascal body silhouette vs clothed silhouette (SCHP-LIP)")
    ax.set_title("SCHP-Pascal body maps follow the CLOTHED outline\n(high IoU = clothing leaks into B_seg; long dress = legs lost)")
    save(fig, "body_map_silhouette_leak.png")


def overfit():
    r = rows("warp_overfit_one_batch_cpu.csv")
    s = [int(x["step"]) for x in r]
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(s, [f(x["ce"]) for x in r], "o-", color="#1f77b4"); ax1.set_ylabel("CE", color="#1f77b4")
    ax2 = ax1.twinx(); ax2.plot(s, [f(x["pixel_acc_supervised"]) for x in r], "s--", color="#2ca02c")
    ax2.set_ylabel("pixel accuracy", color="#2ca02c"); ax2.grid(False)
    ax1.set_xlabel("step"); ax1.set_title("Overfit-one-batch sanity test (warp, CPU, 64 px)")
    save(fig, "warp_overfit_one_batch.png")


def texture_overfit():
    r = rows("texture_overfit_one_batch_cpu.csv")
    s = [int(x["step"]) for x in r]
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(s, [f(x["l1_minus1to1"]) for x in r], "o-", color="#1f77b4"); ax1.set_ylabel("L1 (images in [-1,1])", color="#1f77b4")
    ax2 = ax1.twinx(); ax2.plot(s, [f(x["ssim"]) for x in r], "s--", color="#2ca02c"); ax2.set_ylabel("SSIM", color="#2ca02c")
    ax2.grid(False); ax1.set_xlabel("step"); ax1.set_title("Overfit-one-batch sanity test (texture, CPU, 64 px)")
    save(fig, "texture_overfit_one_batch.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for fn in (warp_metric_bug, warp_per_class_over_steps, warp_eval_bars, warp_losses, warp_swap, silhouette_leak, overfit, texture_overfit):
        fn()
