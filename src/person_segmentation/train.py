"""Train a U-Net on Penn-Fudan and save the best validation checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PennFudanDataset, SMALL_ZOOM_SCALE_RANGE, ZOOM_PROBABILITY, load_split_manifest
from .model import DiceBCELoss, UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="PennFudanPed dataset directory")
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("report/dataset_analysis/split_seed42.json"))
    parser.add_argument("--augmentation", choices=("none", "flip", "zoom", "both"), default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=("stretch", "letterbox"), default="stretch")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--cache", action="store_true", help="Cache decoded images/masks in RAM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model: UNet, loader: DataLoader, criterion: DiceBCELoss,
              device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = true_positive = false_positive = false_negative = true_negative = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks >= 0.5
            true_positive += (predictions & targets).sum().item()
            false_positive += (predictions & ~targets).sum().item()
            false_negative += (~predictions & targets).sum().item()
            true_negative += (~predictions & ~targets).sum().item()
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    dice = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) \
        if 2 * true_positive + false_positive + false_negative else 1.0
    iou = true_positive / (true_positive + false_positive + false_negative) \
        if true_positive + false_positive + false_negative else 1.0
    total_pixels = true_positive + false_positive + false_negative + true_negative
    return {
        "loss": loss_sum / max(1, sample_count),
        "pixel_accuracy": (true_positive + true_negative) / total_pixels if total_pixels else 1.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "dice": dice,
        "iou": iou,
    }


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.image_size < 32
            or args.image_size % 16 != 0 or args.base_channels < 1 or args.learning_rate <= 0):
        raise SystemExit(
            "epochs, batch-size, and base-channels must be positive; learning-rate must be "
            "positive; image-size must be at least 32 and divisible by 16"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_data = PennFudanDataset(args.data, image_size=args.image_size)
    splits = load_split_manifest(args.split_manifest, [path.name for path in base_data.images])
    train_data = PennFudanDataset(
        args.data, splits["train"], args.augmentation, args.image_size, args.resize_mode,
        cache=args.cache,
    )
    validation_data = PennFudanDataset(
        args.data, splits["validation"], "none", args.image_size, args.resize_mode,
        cache=args.cache,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    device = resolve_device(args.device)
    model = UNet(base_channels=args.base_channels).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        if args.lr_scheduler == "cosine" else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["data"] = str(args.data)
    config["output"] = str(args.output)
    config["split_manifest"] = str(args.split_manifest)
    config["device"] = str(device)
    if args.augmentation in {"zoom", "both"}:
        config["zoom_scale_range"] = list(SMALL_ZOOM_SCALE_RANGE)
        config["zoom_probability"] = ZOOM_PROBABILITY
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_dice = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"validation_{k}": v for k, v in validation_metrics.items()}}
        history.append(row)
        print("Epoch {}/{} | train loss {:.4f}, Dice {:.4f} | val loss {:.4f}, Dice {:.4f}, IoU {:.4f}".format(
            epoch, args.epochs, train_metrics["loss"], train_metrics["dice"],
            validation_metrics["loss"], validation_metrics["dice"], validation_metrics["iou"],
        ))
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"in_channels": 3, "out_channels": 1,
                                 "base_channels": args.base_channels},
                "image_size": args.image_size,
                "resize_mode": args.resize_mode,
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "config": config,
            }, args.output / "best.pt")

    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Best checkpoint: {args.output / 'best.pt'} (validation Dice {best_dice:.4f})")


if __name__ == "__main__":
    main()
