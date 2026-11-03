"""Audit Penn-Fudan files and write dataset statistics and preview figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
]


def mask_for(image_path: Path, mask_dir: Path) -> Path:
    return mask_dir / f"{image_path.stem}_mask.png"


def fit_panel(image: Image.Image, size: int = 280) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), "white")
    panel.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return panel


def save_visuals(rows: list[dict], output_dir: Path, target_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    selected = rows if len(rows) <= 6 else [rows[i] for i in np.linspace(0, len(rows) - 1, 6, dtype=int)]
    panel_size, label_height, gap = 280, 28, 10
    canvas = Image.new("RGB", (3 * panel_size + 4 * gap, len(selected) * (panel_size + label_height + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ("Original image", "Instance IDs", "Binary person mask")
    for column, title in enumerate(headers):
        draw.text((gap + column * (panel_size + gap), 2), title, fill="black", font=font)
    for row_index, entry in enumerate(selected):
        y = label_height + gap + row_index * (panel_size + label_height + gap)
        image = Image.open(entry["image_path"]).convert("RGB")
        original_mask = np.asarray(Image.open(entry["mask_path"]))
        instance_view = np.zeros((*original_mask.shape, 3), dtype=np.uint8)
        for instance_id in np.unique(original_mask):
            if instance_id == 0:
                continue
            instance_view[original_mask == instance_id] = COLORS[(int(instance_id) - 1) % len(COLORS)]
        binary = (original_mask > 0).astype(np.uint8) * 255
        views = [image, Image.fromarray(instance_view), Image.fromarray(binary).convert("RGB")]
        for column, view in enumerate(views):
            panel = fit_panel(view, panel_size)
            canvas.paste(panel, (gap + column * (panel_size + gap), y))
        draw.text((gap, y + panel_size + 3), entry["image"], fill="black", font=font)
    canvas.save(output_dir / "sample_images_masks.png")

    distribution = Counter(entry["person_instances"] for entry in rows)
    chart_width, chart_height = 900, 500
    chart = Image.new("RGB", (chart_width, chart_height), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((30, 20), "Penn-Fudan: images by number of person instances", fill="black", font=font)
    left, top, bottom, right = 80, 70, 420, 850
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    max_count = max(distribution.values())
    ticks = sorted(distribution)
    bar_slot = (right - left) / max(1, len(ticks))
    for i, instances in enumerate(ticks):
        count = distribution[instances]
        bar_height = (bottom - top - 35) * count / max_count
        x0 = left + i * bar_slot + bar_slot * 0.2
        x1 = left + (i + 1) * bar_slot - bar_slot * 0.2
        draw.rectangle((x0, bottom - bar_height, x1, bottom), fill=(55, 120, 190))
        draw.text((x0, bottom + 8), str(instances), fill="black", font=font)
        draw.text((x0, bottom - bar_height - 18), str(count), fill="black", font=font)
    draw.text((left, 455), "Person instances per image", fill="black", font=font)
    draw.text((15, 60), "Images", fill="black", font=font)
    chart.save(output_dir / "instances_per_image.png")


def analyze(data_root: Path, output_dir: Path, expected_count: int, target_size: int, seed: int) -> None:
    image_dir, mask_dir = data_root / "PNGImages", data_root / "PedMasks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected PNGImages/ and PedMasks/ under {data_root}")
    images = sorted(image_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    image_stems = {path.stem for path in images}
    mask_stems = {path.stem.removesuffix("_mask") for path in masks}
    missing_masks = sorted(image_stems - mask_stems)
    orphan_masks = sorted(mask_stems - image_stems)

    image_sizes: Counter[str] = Counter()
    mask_sizes: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    instance_distribution: Counter[int] = Counter()
    raw_values: set[int] = set()
    rows: list[dict] = []
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    foreground_fractions: list[float] = []
    binary_values: set[int] = set()
    normalized_min = 1.0
    normalized_max = 0.0
    normalized_finite = True
    dimension_mismatches: list[str] = []

    for image_path in images:
        mask_path = mask_for(image_path, mask_dir)
        if not mask_path.is_file():
            continue
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            image = image_file.convert("RGB")
            mask = np.asarray(mask_file)
            image_array = np.asarray(image, dtype=np.uint8)
            normalized_image = image_array.astype(np.float32) / 255.0
            normalized_min = min(normalized_min, float(normalized_image.min()))
            normalized_max = max(normalized_max, float(normalized_image.max()))
            normalized_finite = normalized_finite and bool(np.isfinite(normalized_image).all())
            width, height = image.size
            mask_height, mask_width = mask.shape[:2]
            image_sizes[f"{width}x{height}"] += 1
            mask_sizes[f"{mask_width}x{mask_height}"] += 1
            mask_modes[mask_file.mode] += 1
            if (width, height) != (mask_width, mask_height):
                dimension_mismatches.append(image_path.name)
            unique_values = np.unique(mask)
            raw_values.update(int(value) for value in unique_values)
            instance_ids = [int(value) for value in unique_values if value != 0]
            instance_count = len(instance_ids)
            instance_distribution[instance_count] += 1
            foreground = (mask > 0)
            foreground_pixels = int(foreground.sum())
            foreground_fraction = foreground_pixels / foreground.size
            foreground_fractions.append(foreground_fraction)
            rgb_sum += image_array.sum(axis=(0, 1), dtype=np.float64)
            rgb_square_sum += np.square(image_array.astype(np.float64)).sum(axis=(0, 1))
            pixel_count += width * height
            resized_mask = Image.fromarray(foreground.astype(np.uint8)).resize(
                (target_size, target_size), Image.Resampling.NEAREST
            )
            binary_values.update(int(value) for value in np.unique(np.asarray(resized_mask)))
            rows.append({
                "image": image_path.name,
                "image_path": image_path,
                "mask_path": mask_path,
                "width": width,
                "height": height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_mode": mask_file.mode,
                "person_instances": instance_count,
                "instance_ids": ";".join(map(str, instance_ids)),
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": foreground_fraction,
            })

    if not rows:
        raise ValueError("No valid image/mask pairs were found")
    channel_mean = rgb_sum / pixel_count
    channel_variance = np.maximum(rgb_square_sum / pixel_count - channel_mean ** 2, 0)
    channel_std = np.sqrt(channel_variance)
    counts = {str(key): value for key, value in sorted(instance_distribution.items())}

    # Stable 70/15/15 split with largest-remainder allocation, recorded by filename.
    ordered_names = [row["image"] for row in rows]
    shuffled = ordered_names.copy()
    random.Random(seed).shuffle(shuffled)
    train_count = math.floor(len(shuffled) * 0.70 + 1e-9)
    validation_count = math.floor(len(shuffled) * 0.15 + 1e-9)
    splits = {
        "train": sorted(shuffled[:train_count]),
        "validation": sorted(shuffled[train_count:train_count + validation_count]),
        "test": sorted(shuffled[train_count + validation_count:]),
    }
    manifest = {
        "dataset": "Penn-Fudan Pedestrian",
        "seed": seed,
        "strategy": "seeded random shuffle, 70/15/15 split",
        "counts": {name: len(names) for name, names in splits.items()},
        **splits,
    }

    summary = {
        "dataset": "Penn-Fudan Pedestrian",
        "source_url": "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
        "expected_image_count": expected_count,
        "image_count": len(images),
        "mask_count": len(masks),
        "matched_pair_count": len(rows),
        "missing_masks": missing_masks,
        "orphan_masks": orphan_masks,
        "dimension_mismatch_images": dimension_mismatches,
        "distinct_original_image_sizes": len(image_sizes),
        "image_width_range": [min(row["width"] for row in rows), max(row["width"] for row in rows)],
        "image_height_range": [min(row["height"] for row in rows), max(row["height"] for row in rows)],
        "image_size_distribution_width_x_height": dict(sorted(image_sizes.items())),
        "mask_size_distribution_width_x_height": dict(sorted(mask_sizes.items())),
        "mask_mode_distribution": dict(mask_modes),
        "raw_mask_values_including_background": sorted(raw_values),
        "person_instance_count_distribution": counts,
        "foreground_fraction_mean_per_image": float(np.mean(foreground_fractions)),
        "foreground_fraction_std_per_image": float(np.std(foreground_fractions)),
        "rgb_mean_0_to_1": (channel_mean / 255).round(6).tolist(),
        "rgb_std_0_to_1": (channel_std / 255).round(6).tolist(),
        "normalized_rgb_range_0_to_1": [normalized_min, normalized_max],
        "normalized_rgb_all_finite": normalized_finite,
        "binary_mask_values_after_nearest_resize": sorted(binary_values),
        "resize_target_width_height": [target_size, target_size],
        "split_seed": seed,
        "split_counts": manifest["counts"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in rows[0] if key not in {"image_path", "mask_path"}]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "split_seed42.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(output_dir / "summary.md", summary, manifest)
    save_visuals(rows, output_dir, target_size)

    failures = []
    if expected_count and len(images) != expected_count:
        failures.append(f"expected {expected_count} images, found {len(images)}")
    if len(images) != len(masks) or missing_masks or orphan_masks:
        failures.append(f"image/mask pairs incomplete ({len(rows)} matched)")
    if dimension_mismatches:
        failures.append(f"{len(dimension_mismatches)} image/mask dimensions differ")
    if binary_values != {0, 1}:
        failures.append(f"resized binary masks contain values {sorted(binary_values)}")
    if not normalized_finite or normalized_min < 0.0 or normalized_max > 1.0:
        failures.append(f"normalized RGB range is [{normalized_min}, {normalized_max}]")
    if len(set(manifest[name][0] for name in ("train", "validation", "test"))) != 3:
        failures.append("split manifest has a duplicate first sample")
    if failures:
        raise RuntimeError("Dataset audit failed: " + "; ".join(failures))
    print(json.dumps(summary, indent=2))
    print(f"Saved dataset audit and visualizations under {output_dir}")


def write_report(path: Path, summary: dict, manifest: dict) -> None:
    image_sizes = summary["image_size_distribution_width_x_height"]
    size_lines = "\n".join(
        f"| {size} | {count} |" for size, count in list(image_sizes.items())[:10]
    )
    instance_lines = "\n".join(
        f"| {count} | {images} |" for count, images in summary["person_instance_count_distribution"].items()
    )
    report = f"""# Penn-Fudan dataset audit

Source: [Penn-Fudan Pedestrian dataset](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)

## Integrity

- Images: {summary['image_count']}; masks: {summary['mask_count']}; matched pairs: {summary['matched_pair_count']}.
- Missing masks: {len(summary['missing_masks'])}; orphan masks: {len(summary['orphan_masks'])}.
- Image/mask dimension mismatches: {len(summary['dimension_mismatch_images'])}.
- Original masks use pixel value 0 for background and positive integer values for separate person instances. Observed values are {summary['raw_mask_values_including_background']}; person instance IDs are {summary['raw_mask_values_including_background'][1:]}.
- Merging all positive IDs and resizing with nearest-neighbor interpolation produced binary values {summary['binary_mask_values_after_nearest_resize']}.

## Image sizes

There are {summary['distinct_original_image_sizes']} distinct original image sizes. Widths range from {summary['image_width_range'][0]} to {summary['image_width_range'][1]} pixels; heights range from {summary['image_height_range'][0]} to {summary['image_height_range'][1]} pixels. All image/mask pairs have matching dimensions. The table shows the first ten width × height pairs; the complete per-image data is in `per_image.csv`.

| Width × height | Images |
|---|---:|
{size_lines}

## Dataset statistics

- RGB channel means, scaled to [0, 1]: {summary['rgb_mean_0_to_1']}.
- RGB channel standard deviations, scaled to [0, 1]: {summary['rgb_std_0_to_1']}.
- Verified normalized RGB values are finite and range from {summary['normalized_rgb_range_0_to_1'][0]:.4f} to {summary['normalized_rgb_range_0_to_1'][1]:.4f}.
- Mean per-image foreground fraction before resize: {summary['foreground_fraction_mean_per_image']:.4f} ± {summary['foreground_fraction_std_per_image']:.4f}.
- Images were resized to {summary['resize_target_width_height'][0]} × {summary['resize_target_width_height'][1]}; RGB used bilinear interpolation and masks used nearest-neighbor interpolation.

## Person instances per image

| Person instances | Images |
|---:|---:|
{instance_lines}

## Reproducible split

Seed {manifest['seed']}; seeded random shuffle; 70/15/15 split. Counts: {manifest['counts']}. The filename manifest is stored in `split_seed42.json`.

See `sample_images_masks.png` for original images, color-coded instance IDs, and merged binary masks. See `instances_per_image.png` for the instance-count distribution. `per_image.csv` and `summary.json` contain the full audit results.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/PennFudanPed"))
    parser.add_argument("--output", type=Path, default=Path("report/dataset_analysis"))
    parser.add_argument("--expected-count", type=int, default=170)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.data, args.output, args.expected_count, args.image_size, args.seed)


if __name__ == "__main__":
    main()
