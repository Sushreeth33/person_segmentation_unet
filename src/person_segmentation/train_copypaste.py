"""Train a U-Net with Copy-Paste person instance augmentation without altering core train.py."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .copypaste import COPY_PASTE_SCALE_RANGE, CopyPastePennFudanDataset
from .data import PennFudanDataset, SMALL_ZOOM_SCALE_RANGE, ZOOM_PROBABILITY, load_split_manifest
from .model import DiceBCELoss, UNet
from .train import resolve_device, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=Path("report/dataset_analysis/split_seed42.json"))
    parser.add_argument("--base-augmentation", choices=("none", "flip", "zoom", "both"), default="both")
    parser.add_argument("--copy-paste-prob", type=float, default=0.5)
    parser.add_argument("--max-paste-instances", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=("stretch", "letterbox"), default="stretch")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_data = PennFudanDataset(args.data, image_size=args.image_size)
    splits = load_split_manifest(args.split_manifest, [path.name for path in base_data.images])

    train_data = CopyPastePennFudanDataset(
        args.data,
        indices=splits["train"],
        base_augmentation=args.base_augmentation,
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        cache=args.cache,
        copy_paste_prob=args.copy_paste_prob,
        max_paste_instances=args.max_paste_instances,
    )
    validation_data = PennFudanDataset(
        args.data,
        splits["validation"],
        augmentation="none",
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        cache=args.cache,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
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
    config["augmentation"] = f"{args.base_augmentation}+copypaste"
    config["training_cutouts"] = len(train_data.cutouts)
    config["copy_paste_scale_range"] = list(COPY_PASTE_SCALE_RANGE)
    if args.base_augmentation in {"zoom", "both"}:
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
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"validation_{k}": v for k, v in validation_metrics.items()},
        }
        history.append(row)
        print(
            "Epoch {}/{} | train loss {:.4f}, Dice {:.4f} | val loss {:.4f}, Dice {:.4f}, IoU {:.4f}".format(
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["dice"],
                validation_metrics["loss"],
                validation_metrics["dice"],
                validation_metrics["iou"],
            )
        )
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {"in_channels": 3, "out_channels": 1, "base_channels": args.base_channels},
                    "image_size": args.image_size,
                    "resize_mode": args.resize_mode,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "config": config,
                },
                args.output / "best.pt",
            )

    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Best checkpoint: {args.output / 'best.pt'} (validation Dice {best_dice:.4f})")


if __name__ == "__main__":
    main()
