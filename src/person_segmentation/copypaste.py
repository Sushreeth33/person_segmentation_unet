"""Copy-Paste person instance augmentation built strictly from training-split masks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from .data import PennFudanDataset, ZOOM_PROBABILITY, _random_zoom, resize_pair

COPY_PASTE_SCALE_RANGE = (0.85, 1.15)


@dataclass(frozen=True)
class PersonCutout:
    """Single segmented pedestrian cutout extracted from a training image."""

    source_index: int
    rgb: Image.Image
    mask: Image.Image
    bottom_y: int


def _extract_training_cutouts(
    dataset: PennFudanDataset,
    image_size: int,
    min_area: int = 150,
    min_height: int = 20,
    min_width: int = 8,
) -> list[PersonCutout]:
    """Build an instance bank strictly from ``dataset.indices`` (training split only)."""
    cutouts: list[PersonCutout] = []
    for ds_idx in dataset.indices:
        image_path = dataset.images[ds_idx]
        with Image.open(image_path) as img_f, Image.open(dataset.mask_path(image_path)) as msk_f:
            rgb = img_f.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            inst_arr = np.asarray(
                msk_f.resize((image_size, image_size), Image.Resampling.NEAREST),
                dtype=np.int32,
            )
        rgb_arr = np.asarray(rgb, dtype=np.uint8)
        for inst_id in np.unique(inst_arr):
            if inst_id <= 0:
                continue
            inst_mask = inst_arr == inst_id
            area = int(np.count_nonzero(inst_mask))
            if area < min_area:
                continue
            ys, xs = np.nonzero(inst_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            h, w = y1 - y0, x1 - x0
            if h < min_height or w < min_width:
                continue
            crop_rgb = Image.fromarray(rgb_arr[y0:y1, x0:x1], mode="RGB")
            crop_mask = Image.fromarray((inst_mask[y0:y1, x0:x1].astype(np.uint8)) * 255, mode="L")
            cutouts.append(
                PersonCutout(
                    source_index=ds_idx,
                    rgb=crop_rgb,
                    mask=crop_mask,
                    bottom_y=y1,
                )
            )
    return cutouts


class CopyPastePennFudanDataset(PennFudanDataset):
    """Penn-Fudan dataset supporting optional Copy-Paste of training-split person instances."""

    def __init__(
        self,
        root: str | Path,
        indices: list[int] | None = None,
        base_augmentation: str = "both",
        image_size: int = 256,
        resize_mode: str = "stretch",
        cache: bool = True,
        copy_paste_prob: float = 0.5,
        max_paste_instances: int = 2,
        scale_range: tuple[float, float] = COPY_PASTE_SCALE_RANGE,
        max_overlap_ratio: float = 0.45,
        feather_edges: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            indices=indices,
            augmentation=base_augmentation,
            image_size=image_size,
            resize_mode=resize_mode,
            cache=cache,
        )
        if not 0.0 <= copy_paste_prob <= 1.0:
            raise ValueError("copy_paste_prob must be in [0, 1]")
        if max_paste_instances < 1:
            raise ValueError("max_paste_instances must be >= 1")
        self.copy_paste_prob = copy_paste_prob
        self.max_paste_instances = max_paste_instances
        self.scale_range = scale_range
        self.max_overlap_ratio = max_overlap_ratio
        self.feather_edges = feather_edges
        self.cutouts = _extract_training_cutouts(self, image_size=image_size)
        if copy_paste_prob > 0 and not self.cutouts:
            raise RuntimeError("No valid training cutouts extracted for Copy-Paste augmentation")

    def _paste_instances(
        self,
        base_image: Image.Image,
        base_mask: Image.Image,
        target_ds_index: int,
    ) -> tuple[Image.Image, Image.Image]:
        """Paste 1..max_paste_instances cutouts onto (base_image, base_mask)."""
        canvas_rgb = np.asarray(base_image, dtype=np.float32).copy()
        canvas_mask = (np.asarray(base_mask, dtype=np.uint8) > 127).copy()
        size = self.image_size

        candidates = [c for c in self.cutouts if c.source_index != target_ds_index] or self.cutouts
        num_to_paste = random.randint(1, self.max_paste_instances)
        for _ in range(num_to_paste):
            cutout = random.choice(candidates)
            patch_rgb = cutout.rgb
            patch_mask = cutout.mask
            if random.random() < 0.5:
                patch_rgb = patch_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                patch_mask = patch_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            scale = random.uniform(*self.scale_range)
            new_w = max(8, min(size, round(patch_rgb.width * scale)))
            new_h = max(16, min(size, round(patch_rgb.height * scale)))
            patch_rgb = patch_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
            patch_mask = patch_mask.resize((new_w, new_h), Image.Resampling.NEAREST)

            x0 = random.randint(0, max(0, size - new_w))
            bottom_target = cutout.bottom_y + random.randint(-28, 28)
            bottom_target = max(new_h, min(size, bottom_target))
            y0 = bottom_target - new_h

            patch_bool = np.asarray(patch_mask, dtype=np.uint8) > 127
            patch_area = int(np.count_nonzero(patch_bool))
            if patch_area == 0:
                continue

            roi_existing = canvas_mask[y0 : y0 + new_h, x0 : x0 + new_w]
            overlap = int(np.count_nonzero(roi_existing & patch_bool))
            if (overlap / patch_area) > self.max_overlap_ratio:
                continue

            if self.feather_edges:
                alpha_img = patch_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
                alpha = (np.asarray(alpha_img, dtype=np.float32) / 255.0)[..., None]
                alpha = np.where(patch_bool[..., None], np.maximum(alpha, 0.75), 0.0)
            else:
                alpha = patch_bool[..., None].astype(np.float32)

            patch_rgb_arr = np.asarray(patch_rgb, dtype=np.float32)
            roi_rgb = canvas_rgb[y0 : y0 + new_h, x0 : x0 + new_w]
            canvas_rgb[y0 : y0 + new_h, x0 : x0 + new_w] = (
                alpha * patch_rgb_arr + (1.0 - alpha) * roi_rgb
            )
            canvas_mask[y0 : y0 + new_h, x0 : x0 + new_w] |= patch_bool

        out_img = Image.fromarray(np.clip(canvas_rgb, 0, 255).astype(np.uint8), mode="RGB")
        out_msk = Image.fromarray(canvas_mask.astype(np.uint8) * 255, mode="L")
        return out_img, out_msk

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_index = self.indices[item]
        if dataset_index in self._cached_pairs:
            image, mask = self._cached_pairs[dataset_index]
        else:
            image, mask = self._load_pair(dataset_index)

        apply_zoom = self.augmentation in {"zoom", "both"} and random.random() < ZOOM_PROBABILITY
        if apply_zoom and self.resize_mode == "stretch":
            image, mask = _random_zoom(image, mask, self.image_size)
        else:
            image, mask, _ = resize_pair(image, mask, self.image_size, self.resize_mode)
            if apply_zoom:
                image, mask = _random_zoom(image, mask, self.image_size)

        if self.augmentation in {"flip", "both"} and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if random.random() < self.copy_paste_prob:
            image, mask = self._paste_instances(image, mask, dataset_index)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None, ...].copy())
        return image_tensor, mask_tensor
