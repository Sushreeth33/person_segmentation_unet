"""Generate publication-style comparison charts and test-mask grids for S.No. 5 & 6."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import PennFudanDataset, load_split_manifest, resize_pair
from .model import UNet
from .train import resolve_device


MODES = ("none", "zoom", "flip", "both", "copypaste", "both_copypaste")
MODE_LABELS = {
    "none": "Baseline (none)",
    "zoom": "Zoom (1.0-1.15)",
    "flip": "Horizontal Flip",
    "both": "Flip + Zoom",
    "copypaste": "Copy-Paste only",
    "both_copypaste": "Flip + Zoom + Copy-Paste",
}
MODE_SHORT = {
    "none": "none",
    "zoom": "zoom",
    "flip": "flip",
    "both": "both",
    "copypaste": "copy-\npaste",
    "both_copypaste": "both +\ncopy-paste",
}
MODE_COLORS = {
    "none": "#64748b",
    "zoom": "#f59e0b",
    "flip": "#0ea5e9",
    "both": "#10b981",
    "copypaste": "#a855f7",
    "both_copypaste": "#e11d48",
}
SEEDS = (42, 43, 44)


def _load_split_summary(results_dir: Path, split: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Return per-mode mean and sample SD for precision, recall, dice, iou."""
    metrics_keys = ("precision", "recall", "dice", "iou")
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for mode in MODES:
        vals: dict[str, list[float]] = {k: [] for k in metrics_keys}
        for seed in SEEDS:
            payload = json.loads((results_dir / f"tuned_{mode}_seed{seed}" / f"{split}_metrics.json").read_text())
            for k in metrics_keys:
                vals[k].append(float(payload["metrics"][k]))
        summary[mode] = {
            k: (statistics.mean(vals[k]), statistics.stdev(vals[k]))
            for k in metrics_keys
        }
    return summary


def _load_seed_values(results_dir: Path, split: str, metric: str) -> dict[str, list[float]]:
    """Return raw list of 3-seed metric values for each mode."""
    out: dict[str, list[float]] = {}
    for mode in MODES:
        vals = []
        for seed in SEEDS:
            payload = json.loads((results_dir / f"tuned_{mode}_seed{seed}" / f"{split}_metrics.json").read_text())
            vals.append(float(payload["metrics"][metric]))
        out[mode] = vals
    return out


def plot_metric_bars(results_dir: Path, output_dir: Path) -> Path:
    """Plot 3-seed mean +- SD grouped bar chart for Validation and Held-out Test."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5), sharey=True)
    metric_keys = ("precision", "recall", "dice", "iou")
    metric_labels = ("Precision", "Recall", "Dice / F1", "IoU")
    x = np.arange(len(metric_keys))
    width = 0.13

    for ax, split, title in zip(
        axes,
        ("validation", "test"),
        ("Validation Split (25 images, 3-seed mean ± SD)", "Held-out Test Split (26 images, 3-seed mean ± SD)"),
    ):
        stats = _load_split_summary(results_dir, split)
        for idx, mode in enumerate(MODES):
            means = [stats[mode][k][0] for k in metric_keys]
            sds = [stats[mode][k][1] for k in metric_keys]
            offset = (idx - (len(MODES) - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=sds,
                capsize=2.5,
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                edgecolor="#1e293b",
                linewidth=0.45,
                alpha=0.92,
            )
            if mode in ("none", "both_copypaste"):
                for bar, mean_val, sd_val in zip(bars, means, sds):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        mean_val + sd_val + 0.006,
                        f"{mean_val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        fontweight="bold" if mode == "both_copypaste" else "normal",
                        color="#0f172a",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0.55, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if split == "validation":
            ax.set_ylabel("Score (threshold = 0.5)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "U-Net Augmentation Comparison & Copy-Paste Ablation (50 epochs, cosine LR, 16 base channels)",
        fontsize=13,
        fontweight="bold",
        y=1.08,
    )
    fig.tight_layout()
    out_path = output_dir / "augmentation_ablation_bar.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_history_arrays(results_dir: Path, mode: str, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return epochs, mean curve, and SD curve across the 3 seeds."""
    series = []
    epochs = []
    for seed in SEEDS:
        with (results_dir / f"tuned_{mode}_seed{seed}" / "history.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not epochs:
            epochs = [int(r["epoch"]) for r in rows]
        series.append([float(r[column]) for r in rows])
    arr = np.asarray(series, dtype=np.float64)
    return np.asarray(epochs), arr.mean(axis=0), arr.std(axis=0, ddof=1)


def plot_training_curves(results_dir: Path, output_dir: Path) -> Path:
    """Plot 50-epoch train/validation loss and validation Dice curves across all 6 variants."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))

    ax_loss = axes[0]
    for mode in ("none", "both", "both_copypaste"):
        ep, tr_m, _ = _load_history_arrays(results_dir, mode, "train_loss")
        _, va_m, va_sd = _load_history_arrays(results_dir, mode, "validation_loss")
        color = MODE_COLORS[mode]
        ax_loss.plot(ep, tr_m, linestyle="--", linewidth=1.4, color=color, alpha=0.65, label=f"{MODE_LABELS[mode]} Train")
        ax_loss.plot(ep, va_m, linestyle="-", linewidth=2.1, color=color, label=f"{MODE_LABELS[mode]} Val")
        ax_loss.fill_between(ep, va_m - va_sd, va_m + va_sd, color=color, alpha=0.11)
    ax_loss.set_title("Training & Validation Loss (Mean ± SD across 3 seeds)", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Dice + BCE Loss")
    ax_loss.set_ylim(0.0, 1.35)
    ax_loss.grid(True, linestyle="--", alpha=0.35)
    ax_loss.legend(fontsize=8.0, loc="upper right")

    ax_dice = axes[1]
    for mode in MODES:
        ep, va_m, va_sd = _load_history_arrays(results_dir, mode, "validation_dice")
        color = MODE_COLORS[mode]
        ax_dice.plot(ep, va_m, linewidth=2.1, color=color, label=MODE_LABELS[mode])
        ax_dice.fill_between(ep, va_m - va_sd, va_m + va_sd, color=color, alpha=0.10)
    ax_dice.set_title("Validation Dice over 50 Epochs (All 6 Variants)", fontsize=11, fontweight="bold")
    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Pooled Validation Dice")
    ax_dice.set_ylim(0.50, 0.87)
    ax_dice.grid(True, linestyle="--", alpha=0.35)
    ax_dice.legend(fontsize=8.2, loc="lower right")

    fig.tight_layout()
    out_path = output_dir / "training_curves_none_vs_both.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_seed_variability(results_dir: Path, output_dir: Path) -> Path:
    """Plot individual seed points and Mean +- SD intervals for Validation & Test IoU."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), sharey=True)
    x = np.arange(len(MODES))

    for ax, split, title in zip(
        axes,
        ("validation", "test"),
        ("Validation IoU across Seeds 42, 43, 44", "Held-out Test IoU across Seeds 42, 43, 44"),
    ):
        raw = _load_seed_values(results_dir, split, "iou")
        for idx, mode in enumerate(MODES):
            vals = raw[mode]
            mean_v = statistics.mean(vals)
            sd_v = statistics.stdev(vals)
            jitter = np.array([-0.09, 0.0, 0.09])
            ax.scatter(
                np.full(3, idx) + jitter,
                vals,
                color=MODE_COLORS[mode],
                s=52,
                zorder=4,
                edgecolors="#0f172a",
                linewidths=0.7,
            )
            ax.errorbar(
                idx,
                mean_v,
                yerr=sd_v,
                fmt="_",
                markersize=16,
                markeredgewidth=2.3,
                elinewidth=1.8,
                capsize=5,
                color="#0f172a",
                zorder=5,
            )
            ax.text(
                idx,
                max(vals) + 0.008,
                f"{mean_v:.4f}\n(±{sd_v:.4f})",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#0f172a",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_SHORT[m] for m in MODES], fontsize=9)
        ax.set_ylim(0.65, 0.815)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if split == "validation":
            ax.set_ylabel("Pooled IoU Score", fontsize=10)

    fig.suptitle(
        "Seed-to-Seed Consistency across All 6 Variants (Best Test Mean: 0.7740 ± 0.0070)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path = output_dir / "seed_variability_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _per_image_test_arrays(
    data_dir: Path,
    results_dir: Path,
    split_manifest: Path,
    device: torch.device,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], np.ndarray, list[np.ndarray]]:
    """Compute per-image test IoU arrays across seeds and collect seed-42 binary predictions."""
    ds = PennFudanDataset(data_dir, augmentation="none", image_size=256, resize_mode="stretch")
    splits = load_split_manifest(split_manifest, [p.name for p in ds.images])
    test_indices = splits["test"]
    filenames = [ds.images[i].name for i in test_indices]

    iou_by_mode: dict[str, np.ndarray] = {}
    preds_s42: dict[str, dict[int, np.ndarray]] = {"none": {}, "both": {}, "both_copypaste": {}}
    targets: list[np.ndarray] = []

    for ds_idx in test_indices:
        _, msk = ds[ds_idx]
        targets.append(msk[0].numpy() >= 0.5)

    for mode in ("none", "both", "both_copypaste"):
        seed_ious = []
        for seed in SEEDS:
            ckpt = torch.load(results_dir / f"tuned_{mode}_seed{seed}" / "best.pt", map_location="cpu", weights_only=False)
            model = UNet(**ckpt["model_config"]).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            ious = []
            with torch.no_grad():
                for pos, ds_idx in enumerate(test_indices):
                    img, _ = ds[ds_idx]
                    pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy() >= 0.5
                    if seed == 42:
                        preds_s42[mode][ds_idx] = pred
                    tgt = targets[pos]
                    tp = np.count_nonzero(pred & tgt)
                    fp = np.count_nonzero(pred & ~tgt)
                    fn = np.count_nonzero(~pred & tgt)
                    ious.append(tp / (tp + fp + fn) if (tp + fp + fn) else 1.0)
            seed_ious.append(ious)
        iou_by_mode[mode] = np.mean(seed_ious, axis=0)

    return filenames, iou_by_mode, preds_s42, np.asarray(test_indices), targets


def plot_per_image_test_deltas(
    filenames: list[str],
    iou_by_mode: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Plot paired per-image 3-seed test IoU change (both_copypaste - none) for all 26 test images."""
    diffs = iou_by_mode["both_copypaste"] - iou_by_mode["none"]
    order = np.argsort(diffs)
    sorted_names = [filenames[i] for i in order]
    sorted_diffs = diffs[order]
    colors = ["#e11d48" if d < 0 else "#10b981" for d in sorted_diffs]

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    y = np.arange(len(sorted_names))
    ax.barh(y, sorted_diffs * 100.0, color=colors, edgecolor="#1e293b", linewidth=0.4, alpha=0.9)
    ax.axvline(0.0, color="#0f172a", linewidth=1.0)
    mean_gain = float(diffs.mean() * 100.0)
    ax.axvline(mean_gain, color="#0284c7", linestyle="--", linewidth=1.4, label=f"Mean gain: {mean_gain:+.2f} pp")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names, fontsize=8.5)
    ax.set_xlabel("3-Seed Mean Test IoU Change: Flip+Zoom+Copy-Paste − Baseline (none) [percentage points]", fontsize=10)
    wins = int(np.sum(diffs > 0))
    ax.set_title(
        f"Per-Image Test IoU Gain across All 26 Held-out Images ({wins}/26 improved, paired p = 0.00049)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    out_path = output_dir / "per_image_test_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _error_map(target: np.ndarray, prediction: np.ndarray) -> Image.Image:
    """Create RGB error map: TP green, FP red, FN blue."""
    err = np.zeros((*target.shape, 3), dtype=np.uint8)
    err[prediction & target] = (40, 200, 70)
    err[prediction & ~target] = (240, 55, 55)
    err[~prediction & target] = (55, 120, 255)
    return Image.fromarray(err, mode="RGB")


def _mask_iou(pred: np.ndarray, tgt: np.ndarray) -> float:
    tp = np.count_nonzero(pred & tgt)
    fp = np.count_nonzero(pred & ~tgt)
    fn = np.count_nonzero(~pred & tgt)
    return float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0


def plot_test_mask_comparison_grid(
    data_dir: Path,
    test_indices: np.ndarray,
    targets: list[np.ndarray],
    preds_s42: dict[str, dict[int, np.ndarray]],
    iou_by_mode: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Create a 5-row visual comparison grid across Baseline, Flip+Zoom, and +Copy-Paste."""
    ds = PennFudanDataset(data_dir, augmentation="none", image_size=256, resize_mode="stretch")
    diffs = iou_by_mode["both_copypaste"] - iou_by_mode["none"]
    cp_mean = iou_by_mode["both_copypaste"]

    selected_positions: list[int] = []
    for pos in [
        int(np.argmax(diffs)),
        int(np.argsort(diffs)[-2]),
        int(np.argmin(cp_mean)),
        int(np.argmax(cp_mean)),
        int(np.argmin(diffs)),
    ]:
        if pos not in selected_positions:
            selected_positions.append(pos)

    size = 256
    header_h = 34
    row_title_h = 26
    pad = 8
    col_labels = (
        "Input Image",
        "Ground Truth",
        "Baseline (none)",
        "Flip + Zoom (both)",
        "+ Copy-Paste",
    )
    canvas_w = size * 5 + pad * 6
    canvas_h = header_h + len(selected_positions) * (size + row_title_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for col_idx, label in enumerate(col_labels):
        draw.text((pad + col_idx * (size + pad) + 6, 9), label, fill="black")

    y_cursor = header_h
    for pos in selected_positions:
        ds_idx = int(test_indices[pos])
        image_path = ds.images[ds_idx]
        raw_img = Image.open(image_path).convert("RGB")
        resized_img, _, _ = resize_pair(raw_img, None, size, "stretch")
        tgt = targets[pos]
        p_none = preds_s42["none"][ds_idx]
        p_both = preds_s42["both"][ds_idx]
        p_cp = preds_s42["both_copypaste"][ds_idx]
        iou_n = _mask_iou(p_none, tgt)
        iou_b = _mask_iou(p_both, tgt)
        iou_c = _mask_iou(p_cp, tgt)

        row_label = (
            f"{image_path.name}  |  Seed-42 IoU: none {iou_n:.3f} -> both {iou_b:.3f} -> +CopyPaste {iou_c:.3f} "
            f"({iou_c - iou_n:+.3f})  |  TP green / FP red / FN blue"
        )
        draw.text((pad + 4, y_cursor + 5), row_label, fill="black")
        y_row = y_cursor + row_title_h

        gt_rgb = Image.fromarray((tgt.astype(np.uint8) * 255), mode="L").convert("RGB")
        panels = (
            resized_img,
            gt_rgb,
            _error_map(tgt, p_none),
            _error_map(tgt, p_both),
            _error_map(tgt, p_cp),
        )
        for col_idx, panel in enumerate(panels):
            canvas.paste(panel, (pad + col_idx * (size + pad), y_row))
        y_cursor += row_title_h + size + pad

    out_path = output_dir / "test_baseline_vs_proposed_masks.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/PennFudanPed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--split-manifest", type=Path, default=Path("report/dataset_analysis/split_seed42.json"))
    parser.add_argument("--output", type=Path, default=Path("report/figures"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    saved = [
        plot_metric_bars(args.results_dir, args.output),
        plot_training_curves(args.results_dir, args.output),
        plot_seed_variability(args.results_dir, args.output),
    ]
    filenames, iou_by_mode, preds_s42, test_indices, targets = _per_image_test_arrays(
        args.data, args.results_dir, args.split_manifest, device
    )
    saved.append(plot_per_image_test_deltas(filenames, iou_by_mode, args.output))
    saved.append(
        plot_test_mask_comparison_grid(
            args.data, test_indices, targets, preds_s42, iou_by_mode, args.output
        )
    )
    print("Generated figures:")
    for path in saved:
        print(f" - {path}")


if __name__ == "__main__":
    main()
