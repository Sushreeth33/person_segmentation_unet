"""Predict and save a binary person mask for one image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .train import resolve_device
from .model import UNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = resolve_device(args.device)
    model = UNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    size = checkpoint["image_size"]
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask.resize(original_size, Image.Resampling.NEAREST).save(args.output)
    print(f"Saved binary mask to {args.output}")


if __name__ == "__main__":
    main()
