"""Penn-Fudan dataset loading and paired image/mask augmentations."""

from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SMALL_ZOOM_SCALE_RANGE = (1.0, 1.15)
ZOOM_PROBABILITY = 0.5


class PennFudanDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load Penn-Fudan images and merge instance IDs into a person mask."""

    def __init__(self, root: str | Path, indices: list[int] | None = None,
                 augmentation: str = "none", image_size: int = 256,
                 resize_mode: str = "stretch", cache: bool = False) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "PNGImages"
        self.mask_dir = self.root / "PedMasks"
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected PNGImages/ and PedMasks/ under {self.root}. "
                "See the README for dataset setup."
            )
        self.images = sorted(self.image_dir.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No PNG images found in {self.image_dir}")
        if image_size < 1:
            raise ValueError("image_size must be positive")
        if augmentation not in {"none", "flip", "zoom", "both"}:
            raise ValueError("augmentation must be one of: none, flip, zoom, both")
        if resize_mode not in {"stretch", "letterbox"}:
            raise ValueError("resize_mode must be one of: stretch, letterbox")
        missing = [p.name for p in self.images if not self.mask_path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"Masks missing for {len(missing)} image(s), e.g. {missing[0]}")
        self.indices = list(range(len(self.images))) if indices is None else indices
        self.augmentation = augmentation
        self.image_size = image_size
        self.resize_mode = resize_mode
        self._cached_pairs: dict[int, tuple[Image.Image, Image.Image]] = {}
        if cache:
            for idx in self.indices:
                self._cached_pairs[idx] = self._load_pair(idx)

    def _load_pair(self, dataset_index: int) -> tuple[Image.Image, Image.Image]:
        image_path = self.images[dataset_index]
        with Image.open(image_path) as image_file, Image.open(self.mask_path(image_path)) as mask_file:
            image = image_file.convert("RGB")
            mask_array = np.asarray(mask_file)
        # Penn-Fudan masks use 0 for background and distinct IDs for people.
        mask = Image.fromarray((mask_array > 0).astype(np.uint8) * 255)
        return image, mask

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_index = self.indices[item]
        if dataset_index in self._cached_pairs:
            image, mask = self._cached_pairs[dataset_index]
        else:
            image, mask = self._load_pair(dataset_index)

        apply_zoom = self.augmentation in {"zoom", "both"} and random.random() < ZOOM_PROBABILITY
        if apply_zoom and self.resize_mode == "stretch":
            # Resize directly from raw image to zoomed size to avoid double-resampling blur.
            image, mask = _random_zoom(image, mask, self.image_size)
        else:
            image, mask, _ = resize_pair(image, mask, self.image_size, self.resize_mode)
            if apply_zoom:
                image, mask = _random_zoom(image, mask, self.image_size)

        if self.augmentation in {"flip", "both"} and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None, ...].copy())
        return image_tensor, mask_tensor

    def mask_path(self, image_path: Path) -> Path:
        """Penn-Fudan mask names append ``_mask`` to the image stem."""
        return self.mask_dir / f"{image_path.stem}_mask.png"


def _random_zoom(image: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    """Apply the same random zoom and crop/pad to an image and its mask."""
    scale = random.uniform(*SMALL_ZOOM_SCALE_RANGE)
    scaled_size = max(1, round(size * scale))
    image = image.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
    mask = mask.resize((scaled_size, scaled_size), Image.Resampling.NEAREST)
    if scaled_size >= size:
        left = random.randint(0, scaled_size - size)
        top = random.randint(0, scaled_size - size)
        box = (left, top, left + size, top + size)
        return image.crop(box), mask.crop(box)

    left = random.randint(0, size - scaled_size)
    top = random.randint(0, size - scaled_size)
    image_canvas = Image.new("RGB", (size, size), (0, 0, 0))
    mask_canvas = Image.new("L", (size, size), 0)
    image_canvas.paste(image, (left, top))
    mask_canvas.paste(mask, (left, top))
    return image_canvas, mask_canvas


def resize_pair(image: Image.Image, mask: Image.Image | None, size: int,
                mode: str = "stretch") -> tuple[Image.Image, Image.Image | None, dict[str, object]]:
    """Resize an image/mask pair and return geometry for optional inverse mapping."""
    original_size = image.size
    if mode == "stretch":
        resized_image = image.resize((size, size), Image.Resampling.BILINEAR)
        resized_mask = mask.resize((size, size), Image.Resampling.NEAREST) if mask is not None else None
        transform: dict[str, object] = {
            "mode": "stretch",
            "original_size": list(original_size),
            "resized_size": [size, size],
            "padding": [0, 0, 0, 0],
        }
        return resized_image, resized_mask, transform

    if mode != "letterbox":
        raise ValueError("mode must be one of: stretch, letterbox")

    original_width, original_height = original_size
    scale = min(size / original_width, size / original_height)
    resized_width = max(1, min(size, round(original_width * scale)))
    resized_height = max(1, min(size, round(original_height * scale)))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    right = size - resized_width - left
    bottom = size - resized_height - top

    resized_image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    image_canvas = Image.new("RGB", (size, size), (128, 128, 128))
    image_canvas.paste(resized_image, (left, top))
    if mask is None:
        mask_canvas = None
    else:
        resized_mask = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
        mask_canvas = Image.new("L", (size, size), 0)
        mask_canvas.paste(resized_mask, (left, top))

    transform = {
        "mode": "letterbox",
        "original_size": list(original_size),
        "resized_size": [resized_width, resized_height],
        "padding": [left, top, right, bottom],
    }
    return image_canvas, mask_canvas, transform


def restore_mask(mask: Image.Image, transform: dict[str, object]) -> Image.Image:
    """Map a square prediction mask back to its original image dimensions."""
    original_width, original_height = transform["original_size"]
    if transform["mode"] == "stretch":
        return mask.resize((int(original_width), int(original_height)), Image.Resampling.NEAREST)
    left, top, right, bottom = transform["padding"]
    size = mask.size[0]
    cropped = mask.crop((int(left), int(top), size - int(right), size - int(bottom)))
    return cropped.resize((int(original_width), int(original_height)), Image.Resampling.NEAREST)


def load_split_manifest(path: str | Path, image_names: list[str]) -> dict[str, list[int]]:
    """Load and validate a filename-based train/validation/test split manifest."""
    manifest_path = Path(path)
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_names = ("train", "validation", "test")
    if any(not isinstance(payload.get(name), list) for name in split_names):
        raise ValueError(f"{manifest_path} must contain train, validation, and test filename lists")
    positions = {name: index for index, name in enumerate(image_names)}
    seen: set[str] = set()
    result: dict[str, list[int]] = {}
    for split_name in split_names:
        names = payload[split_name]
        if not all(isinstance(name, str) for name in names):
            raise ValueError(f"All entries in split '{split_name}' must be filenames")
        if len(names) != len(set(names)):
            raise ValueError(f"Split '{split_name}' contains duplicate filenames")
        duplicates = seen.intersection(names)
        if duplicates:
            raise ValueError(f"Split manifest contains overlapping images, e.g. {sorted(duplicates)[0]}")
        unknown = [name for name in names if name not in positions]
        if unknown:
            raise ValueError(f"Split manifest contains unknown image filename: {unknown[0]}")
        seen.update(names)
        result[split_name] = [positions[name] for name in names]
    missing = sorted(set(image_names) - seen)
    if missing:
        raise ValueError(f"Split manifest does not assign all images; first missing: {missing[0]}")
    if any(not result[name] for name in split_names):
        raise ValueError("Train, validation, and test splits must all be non-empty")
    return result
