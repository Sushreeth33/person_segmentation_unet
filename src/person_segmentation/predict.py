"""Predict, evaluate, and visualize a binary person mask for an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import label

from .data import resize_pair, restore_mask
from .train import resolve_device
from .model import UNet


def tint(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    """Blend a color over pixels selected by a boolean mask."""
    base = np.asarray(image, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    color_array = np.asarray(color, dtype=np.float32)
    base[selected] = base[selected] * (1.0 - alpha) + color_array * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def create_visualization(image: Image.Image, pred_binary: np.ndarray,
                         target_binary: np.ndarray | None = None) -> Image.Image:
    """Create a side-by-side comparison card."""
    width, height = image.size
    header = 40
    
    if target_binary is not None:
        tp = pred_binary & target_binary
        fp = pred_binary & ~target_binary
        fn = ~pred_binary & target_binary
        error_colors = np.zeros((*target_binary.shape, 3), dtype=np.uint8)
        error_colors[tp] = (40, 200, 70)   # TP: Green
        error_colors[fp] = (240, 55, 55)   # FP: Red
        error_colors[fn] = (55, 120, 255)  # FN: Blue

        panels = (
            ("1. Input RGB", image),
            ("2. Ground Truth", tint(image, target_binary, (40, 230, 70))),
            ("3. Prediction", tint(image, pred_binary, (220, 40, 230))),
            ("4. Errors (TP:Grn, FP:Red, FN:Blu)", Image.fromarray(error_colors, mode="RGB")),
        )
    else:
        # Prediction only
        mask_gray = Image.fromarray((pred_binary.astype(np.uint8) * 255)).convert("RGB")
        overlay = tint(image, pred_binary, (220, 40, 230), alpha=0.45)
        panels = (
            ("1. Input RGB Image", image),
            ("2. Predicted Binary Mask", mask_gray),
            ("3. Person Overlay (Magenta)", overlay),
        )

    card = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(card)
    for idx, (lbl, panel) in enumerate(panels):
        x = idx * width
        draw.text((x + 10, 12), lbl, fill="black")
        card.paste(panel, (x, header))
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to input test image (.png, .jpg)")
    parser.add_argument("--mask", type=Path, default=None, help="Optional ground truth mask to compute scores")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/unet_proposed_both_seed42.pt"),
                        help="Path to trained U-Net checkpoint (.pt)")
    parser.add_argument("--output", type=Path, default=Path("results/prediction.png"),
                        help="Path to save predicted binary mask")
    parser.add_argument("--visualize", type=Path, default=Path("results/prediction_visual.png"),
                        help="Path to save side-by-side visual card")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification threshold")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Fallback checkpoint check
    ckpt_path = args.checkpoint
    if not ckpt_path.exists() and Path("results/tuned_both_seed42/best.pt").exists():
        ckpt_path = Path("results/tuned_both_seed42/best.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please specify a valid checkpoint.")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = resolve_device(args.device)
    model = UNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    size = checkpoint["image_size"]
    resize_mode = checkpoint.get("resize_mode", checkpoint.get("config", {}).get("resize_mode", "stretch"))
    image_resized, _, transform = resize_pair(image, None, size, resize_mode)
    array = np.asarray(image_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)

    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    mask_resized = (probability >= args.threshold).astype(np.uint8) * 255
    mask = Image.fromarray(mask_resized)
    mask = restore_mask(mask, transform)

    if mask.size != original_size:
        raise RuntimeError("Restored prediction mask dimensions do not match the input image")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(args.output)
    
    pred_binary = np.asarray(mask) > 0
    pred_count = np.count_nonzero(pred_binary)
    total_pixels = pred_binary.size
    pred_pct = (pred_count / total_pixels) * 100.0

    # Count estimated distinct pedestrians using connected components
    _, num_instances = label(pred_binary)

    # 1. If ground-truth mask is provided, compute rigorous evaluation scores
    if args.mask is not None and args.mask.exists():
        target_img = Image.open(args.mask).convert("L")
        if target_img.size != original_size:
            target_img = target_img.resize(original_size, Image.Resampling.NEAREST)
        target_binary = np.asarray(target_img) > 0
        target_count = np.count_nonzero(target_binary)
        target_pct = (target_count / total_pixels) * 100.0

        tp = int(np.logical_and(pred_binary, target_binary).sum())
        fp = int(np.logical_and(pred_binary, ~target_binary).sum())
        fn = int(np.logical_and(~pred_binary, target_binary).sum())
        tn = int(np.logical_and(~pred_binary, ~target_binary).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 1.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 1.0

        print("\n" + "=" * 65)
        print("          SINGLE-IMAGE EVALUATION METRICS (VS. GROUND TRUTH)")
        print("=" * 65)
        print(f"  Test Image:          {args.image} ({original_size[0]}x{original_size[1]})")
        print(f"  Ground Truth Mask:   {args.mask}")
        print(f"  Model Checkpoint:    {ckpt_path.name}")
        print("-" * 65)
        print(f"  {'Metric':<24} {'Score':<12} {'Percentage':<12}")
        print("-" * 65)
        print(f"  {'Dice / F1 Score':<24} {dice:<12.4f} {dice*100:<10.2f}%")
        print(f"  {'IoU (Jaccard Index)':<24} {iou:<12.4f} {iou*100:<10.2f}%")
        print(f"  {'Pixel Accuracy':<24} {accuracy:<12.4f} {accuracy*100:<10.2f}%")
        print(f"  {'Precision':<24} {precision:<12.4f} {precision*100:<10.2f}%")
        print(f"  {'Recall (Sensitivity)':<24} {recall:<12.4f} {recall*100:<10.2f}%")
        print(f"  {'Specificity':<24} {specificity:<12.4f} {specificity*100:<10.2f}%")
        print("-" * 65)
        print(f"  Ground Truth Area:   {target_count:,} px ({target_pct:.2f}% of scene)")
        print(f"  Predicted Area:      {pred_count:,} px ({pred_pct:.2f}% of scene)")
        print(f"  Detected Silhouettes:{num_instances} distinct pedestrian instance(s)")
        print("=" * 65)

        # Generate 4-panel visual comparison card
        if args.visualize:
            card = create_visualization(image, pred_binary, target_binary)
            args.visualize.parent.mkdir(parents=True, exist_ok=True)
            card.save(args.visualize)
            print(f"  Saved 4-panel comparison card to: {args.visualize}")
        print(f"  Saved predicted binary mask to:   {args.output}\n")

    else:
        # 2. Inference-only (no ground-truth mask provided)
        print("\n" + "=" * 65)
        print("          SINGLE-IMAGE PREDICTION SUMMARY (INFERENCE MODE)")
        print("=" * 65)
        print(f"  Input Image:         {args.image} ({original_size[0]}x{original_size[1]} RGB)")
        print(f"  Model Input Size:    {size}x{size} ({resize_mode})")
        print(f"  Model Checkpoint:    {ckpt_path.name}")
        print("-" * 65)
        print(f"  Foreground Area:     {pred_count:,} pixels ({pred_pct:.2f}% of scene)")
        print(f"  Estimated Persons:   {num_instances} distinct pedestrian silhouette(s)")
        print(f"  Decision Threshold:  tau = {args.threshold}")
        print("=" * 65)

        # Generate 3-panel visual overlay card
        if args.visualize:
            card = create_visualization(image, pred_binary, None)
            args.visualize.parent.mkdir(parents=True, exist_ok=True)
            card.save(args.visualize)
            print(f"  Saved visual overlay card to:    {args.visualize}")
        print(f"  Saved predicted binary mask to:  {args.output}\n")


if __name__ == "__main__":
    main()
