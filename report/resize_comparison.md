# Resize preprocessing comparison

## Experiment

We compared the existing direct stretch to 256 × 256 with aspect-ratio-preserving resize and centered padding (letterbox). Both runs used the seed-42 horizontal-flip U-Net, the same train/validation split, 30 epochs, batch size 4, 16 base channels, Adam at 0.001, and CPU. The stretch model was evaluated from the original images with stretch preprocessing; this reproduced its saved validation metrics exactly. The letterbox run trained and validated directly from original Penn-Fudan images and masks so the original aspect ratios were retained.

For letterbox, each image was resized to fit within 256 × 256, centered on a gray RGB canvas, and paired with a zero-valued mask pad. Checkpoints were selected by best validation Dice. The test split was not used.

## Validation results

| Resize method | Best epoch | Loss | Pixel accuracy | Precision | Recall | Specificity | Micro Dice | Micro IoU | Mean image Dice | Mean image IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stretch | 29 | 0.1837 | 0.9387 | 0.8104 | 0.8460 | 0.9583 | 0.8278 | 0.7062 | 0.8331 | 0.7225 |
| Letterbox | 24 | 0.1743 | 0.9523 | 0.8109 | 0.8266 | 0.9711 | 0.8187 | 0.6930 | 0.8257 | 0.7121 |

On this fixed validation split, letterbox was lower by 0.91 percentage points in micro Dice and 1.32 points in micro IoU. Pixel accuracy increased, but it is less informative here because letterbox adds padded background pixels. Precision was nearly unchanged, while recall fell by about 1.94 points.

The 256 × 256 letterbox approach did not improve the target overlap scores, so retain stretch preprocessing for the current model. This is a one-seed comparison on one validation split; it is evidence for the current project choice, not a universal claim that stretching is better. The test split was left untouched.

## Visual and reproducibility files

- Reproduce the letterbox training run (from the repository root):

```powershell
python -m person_segmentation.train --data data/PennFudanPed --output results/resize_letterbox_flip_seed42 --split-manifest report/dataset_analysis/split_seed42.json --augmentation flip --resize-mode letterbox --epochs 30 --batch-size 4 --image-size 256 --base-channels 16 --learning-rate 0.001 --seed 42 --device cpu
```

- Letterbox validation overlays and per-image metrics: `report/resize_letterbox_validation_analysis/`.
- Letterbox checkpoint, settings, and training history: `results/resize_letterbox_flip_seed42/` (ignored by Git).
- Stretch comparison checkpoint: `results/augmentation_flip/best.pt` (ignored by Git).
- The loader, train/evaluate commands, prediction, and visualization paths now support the checkpoint's saved `resize_mode` (`stretch` or `letterbox`).
