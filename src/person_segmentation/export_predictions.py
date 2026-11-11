"""Export standalone black-and-white prediction masks for a dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import PennFudanDataset, load_split_manifest, resize_pair, restore_mask
from .model import UNet
from .train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("report/dataset_analysis/split_seed42.json"))
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("report/validation_binary_predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(checkpoint["image_size"])
    resize_mode = checkpoint.get("resize_mode", checkpoint.get("config", {}).get("resize_mode", "stretch"))
    source = PennFudanDataset(args.data, image_size=image_size, resize_mode=resize_mode)
    split_indices = load_split_manifest(
        args.split_manifest, [path.name for path in source.images]
    )
    indices = list(range(len(source.images))) if args.split == "all" else split_indices[args.split]

    device = resolve_device(args.device)
    model = UNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, object]] = []
    for index in indices:
        image_path = source.images[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        original_size = image.size
        model_image, _, transform = resize_pair(image, None, image_size, resize_mode)
        image_array = np.asarray(model_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

        binary = Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255, mode="L")
        binary = restore_mask(binary, transform)
        if binary.size != original_size:
            raise RuntimeError(f"Prediction size mismatch for {image_path.name}")
        output_path = args.output / f"{image_path.stem}_prediction.png"
        binary.save(output_path)
        exported.append({
            "source_image": image_path.name,
            "prediction_mask": output_path.name,
            "width": binary.width,
            "height": binary.height,
            "black_background": 0,
            "white_person": 255,
        })

    manifest = {
        "split": args.split,
        "images": len(exported),
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "resize_mode": resize_mode,
        "mask_encoding": "8-bit grayscale: 0=background (black), 255=person (white)",
        "predictions": exported,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "exported_masks": len(exported),
        "output": str(args.output),
        "mask_encoding": manifest["mask_encoding"],
    }, indent=2))


if __name__ == "__main__":
    main()
