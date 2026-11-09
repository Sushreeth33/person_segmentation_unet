"""Export resized RGB images and binary masks as PNG files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def export_dataset(data_root: Path, output_root: Path, image_size: int, overwrite: bool) -> None:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    image_dir = data_root / "PNGImages"
    mask_dir = data_root / "PedMasks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected PNGImages/ and PedMasks/ under {data_root}")
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError("Output directory must not be inside the source dataset")
    if image_size < 1:
        raise ValueError("image_size must be positive")

    images = sorted(image_dir.glob("*.png"))
    missing = [p.name for p in images if not (mask_dir / f"{p.stem}_mask.png").is_file()]
    if not images or missing:
        raise FileNotFoundError(f"Source dataset is incomplete; images={len(images)}, missing masks={len(missing)}")

    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_root} is not empty. Choose another output path or pass --overwrite to replace files."
        )
    output_images = output_root / "PNGImages"
    output_masks = output_root / "PedMasks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        with Image.open(image_path) as source_image, Image.open(mask_path) as source_mask:
            image = source_image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            instance_ids = np.asarray(source_mask)
            binary_mask = (instance_ids > 0).astype(np.uint8)
            resized_mask = Image.fromarray(binary_mask).resize(
                (image_size, image_size), Image.Resampling.NEAREST
            )
            image.save(output_images / image_path.name, format="PNG")
            resized_mask.save(output_masks / f"{image_path.stem}_mask.png", format="PNG")

    exported_images = sorted(output_images.glob("*.png"))
    exported_masks = sorted(output_masks.glob("*.png"))
    if len(exported_images) != len(images) or len(exported_masks) != len(images):
        raise RuntimeError("Exported image or mask count does not match the source dataset")

    mask_values: set[int] = set()
    for image_path in exported_images:
        mask_path = output_masks / f"{image_path.stem}_mask.png"
        if not mask_path.is_file():
            raise RuntimeError(f"Exported mask missing for {image_path.name}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != (image_size, image_size) or mask.size != (image_size, image_size):
                raise RuntimeError(f"Exported dimensions are incorrect for {image_path.name}")
            if image.mode != "RGB":
                raise RuntimeError(f"Exported image is not RGB: {image_path.name}")
            mask_values.update(int(value) for value in np.unique(np.asarray(mask)))
    if not mask_values.issubset({0, 1}):
        raise RuntimeError(f"Exported masks are not binary: {sorted(mask_values)}")

    metadata = {
        "source_dataset": str(data_root),
        "image_count": len(exported_images),
        "image_size_width_height": [image_size, image_size],
        "image_format": "RGB PNG (8-bit per channel)",
        "mask_format": "grayscale PNG with values 0=background, 1=person",
        "rgb_resize_interpolation": "bilinear",
        "mask_resize_interpolation": "nearest",
        "rgb_float_normalization": "Divide 8-bit RGB values by 255 when loading for a model",
        "mask_values": sorted(mask_values),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Exported images: {output_images}")
    print(f"Exported masks:  {output_masks}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/PennFudanPed"))
    parser.add_argument("--output", type=Path, default=Path("data/PennFudanPed_256"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_dataset(args.data, args.output, args.image_size, args.overwrite)


if __name__ == "__main__":
    main()
