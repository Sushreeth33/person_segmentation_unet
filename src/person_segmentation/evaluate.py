"""Evaluate a saved checkpoint on a dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from .data import PennFudanDataset, load_split_manifest
from .model import DiceBCELoss, UNet
from .train import resolve_device, run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("report/dataset_analysis/split_seed42.json"))
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resize-mode", choices=("stretch", "letterbox"),
                        help="Override the checkpoint preprocessing mode")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Optional path to save the evaluation JSON")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = checkpoint["image_size"]
    resize_mode = args.resize_mode or checkpoint.get(
        "resize_mode", checkpoint.get("config", {}).get("resize_mode", "stretch")
    )
    dataset = PennFudanDataset(
        args.data, augmentation="none", image_size=image_size, resize_mode=resize_mode
    )
    if args.split == "all":
        dataset.indices = list(range(len(dataset.images)))
    else:
        splits = load_split_manifest(args.split_manifest, [path.name for path in dataset.images])
        dataset.indices = splits[args.split]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    model = UNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    model.to(device)
    metrics = run_epoch(model, loader, DiceBCELoss(), device)
    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "images": len(dataset),
        "threshold": 0.5,
        "metrics": metrics,
    }
    serialized = json.dumps(results, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
