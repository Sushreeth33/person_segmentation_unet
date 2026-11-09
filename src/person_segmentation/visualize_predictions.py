"""Create validation overlays and per-image scores for a saved checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import PennFudanDataset, load_split_manifest, resize_pair
from .model import UNet
from .train import resolve_device


PANEL_LABELS = ("Image", "Ground truth", "Prediction", "Errors: TP green / FP red / FN blue")


def tint(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    """Blend a color over pixels selected by a boolean mask."""
    base = np.asarray(image, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    color_array = np.asarray(color, dtype=np.float32)
    base[selected] = base[selected] * (1.0 - alpha) + color_array * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def make_card(image: Image.Image, target: np.ndarray, prediction: np.ndarray,
              title: str) -> Image.Image:
    """Create a four-panel card with truth, prediction, and pixel errors."""
    tp = prediction & target
    fp = prediction & ~target
    fn = ~prediction & target
    error_colors = np.zeros((*target.shape, 3), dtype=np.uint8)
    error_colors[tp] = (40, 200, 70)
    error_colors[fp] = (240, 55, 55)
    error_colors[fn] = (55, 120, 255)

    panels = (
        image,
        tint(image, target, (40, 230, 70)),
        tint(image, prediction, (220, 40, 230)),
        Image.fromarray(error_colors, mode="RGB"),
    )
    width, height = image.size
    header = 54
    card = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(card)
    draw.text((8, 5), title, fill="black")
    for panel_index, (label, panel) in enumerate(zip(PANEL_LABELS, panels)):
        x = panel_index * width
        draw.text((x + 5, 30), label, fill="black")
        card.paste(panel, (x, header))
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("report/dataset_analysis/split_seed42.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("report/validation_error_analysis"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = int(checkpoint["image_size"])
    base_dataset = PennFudanDataset(args.data, image_size=size)
    splits = load_split_manifest(args.split_manifest, [path.name for path in base_dataset.images])
    resize_mode = checkpoint.get("resize_mode", checkpoint.get("config", {}).get("resize_mode", "stretch"))
    dataset = PennFudanDataset(args.data, splits[args.split], "none", size, resize_mode)

    device = resolve_device(args.device)
    model = UNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    overlay_dir = args.output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    total_tp = total_fp = total_fn = total_tn = 0
    cards: dict[str, Image.Image] = {}

    for item in range(len(dataset)):
        image_tensor, mask_tensor = dataset[item]
        original_index = dataset.indices[item]
        image_path = dataset.images[original_index]
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))[0, 0]
            probability = torch.sigmoid(logits).cpu().numpy()

        target = mask_tensor[0].numpy() >= 0.5
        prediction = probability >= args.threshold
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        tn = int(np.count_nonzero(~prediction & ~target))
        dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        image = Image.open(image_path).convert("RGB")
        image, _, _ = resize_pair(image, None, size, resize_mode)
        title = f"{image_path.name}  Dice {dice:.3f}  IoU {iou:.3f}"
        card = make_card(image, target, prediction, title)
        cards[image_path.name] = card
        card.save(overlay_dir / f"{image_path.stem}_overlay.png")
        rows.append({
            "image": image_path.name,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "true_positive_pixels": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "foreground_pixels": int(np.count_nonzero(target)),
        })

    rows.sort(key=lambda row: (float(row["dice"]), str(row["image"])))
    with (args.output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sorted_cards = [cards[str(row["image"])] for row in rows]
    columns = 2
    rows_count = (len(sorted_cards) + columns - 1) // columns
    card_width, card_height = sorted_cards[0].size
    worst_sheet = Image.new("RGB", (columns * card_width, rows_count * card_height), "white")
    for index, card in enumerate(sorted_cards):
        x = (index % columns) * card_width
        y = (index // columns) * card_height
        worst_sheet.paste(card, (x, y))
    worst_sheet.save(args.output / "validation_overlays_sorted_by_dice.png")

    total_pixels = total_tp + total_fp + total_fn + total_tn
    micro_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn)
    micro_iou = total_tp / (total_tp + total_fp + total_fn)
    summary = {
        "split": args.split,
        "images": len(rows),
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint),
        "micro_dice": micro_dice,
        "micro_iou": micro_iou,
        "macro_image_dice": float(np.mean([row["dice"] for row in rows])),
        "macro_image_iou": float(np.mean([row["iou"] for row in rows])),
        "pixel_accuracy": (total_tp + total_tn) / total_pixels if total_pixels else 1.0,
        "lowest_dice_images": [row["image"] for row in rows[:5]],
        "highest_false_negative_images": [
            row["image"] for row in sorted(rows, key=lambda row: int(row["false_negative_pixels"]), reverse=True)[:5]
        ],
        "highest_false_positive_images": [
            row["image"] for row in sorted(rows, key=lambda row: int(row["false_positive_pixels"]), reverse=True)[:5]
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
