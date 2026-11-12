# Person segmentation with U-Net

Binary person segmentation on the Penn-Fudan Pedestrian dataset. This scaffold includes a compact U-Net, paired image/mask augmentations, training, evaluation, and single-image prediction. It is set up to compare horizontal flipping and zooming while keeping the validation split fixed.

## Project layout

```text
src/person_segmentation/  dataset, model, training, evaluation, prediction
notebooks/                space for exploratory analysis
report/                   space for experiment notes and results
results/                  generated checkpoints and training history (git-ignored)
slides/                   space for presentation material
```

## Setup

Use Python 3.10 or newer. Install the project and its dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
```

Download and extract the Penn-Fudan Pedestrian dataset from the [official Penn-Fudan archive](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip). The data directory passed to the commands below should contain `PNGImages/` and `PedMasks/`, for example:

```text
data/PennFudanPed/PNGImages/FudanPed00001.png
data/PennFudanPed/PedMasks/FudanPed00001_mask.png
```

Run the dataset audit after extraction:

```bash
python -m person_segmentation.analyze_dataset --data data/PennFudanPed --output report/dataset_analysis --seed 42
```

The audit checks all 170 image/mask pairs, dimensions, mask IDs, normalized RGB range, and binary-mask values. It saves per-image statistics, a summary, visualizations, and a filename-based split manifest under `report/dataset_analysis/`. The current split uses seed 42 and assigns 119 images to train, 25 to validation, and 26 to test. The dataset itself is local and excluded from Git.

To save physical 256 × 256 PNG copies (instead of resizing only when loading), run:

```bash
python -m person_segmentation.export_preprocessed --data data/PennFudanPed --output data/PennFudanPed_256 --image-size 256
```

This creates `data/PennFudanPed_256/PNGImages/` and `data/PennFudanPed_256/PedMasks/`, with 170 files in each. The RGB PNGs remain 8-bit; the loader converts them to `[0, 1]` floats by dividing by 255. The mask PNGs contain exactly `0` for background and `1` for person. Export settings and counts are recorded in `data/PennFudanPed_256/metadata.json`.

Training resizes images to a square input size (default 256 × 256). Penn-Fudan mask filenames append `_mask` to the image stem. Original masks use 0 for background and positive integer IDs for person instances; preprocessing merges every positive ID into one person foreground class. RGB values are scaled to [0, 1]. Images are resized with bilinear interpolation and masks with nearest-neighbor interpolation.

## Train

```bash
python -m person_segmentation.train --data data/PennFudanPed_256 --output results/baseline --augmentation none --seed 42 --epochs 30 --batch-size 4 --image-size 256 --base-channels 16 --learning-rate 0.001 --device cpu
```

Available augmentation modes are `none`, `flip`, `zoom`, and `both`. Horizontal flips and random zooms are applied to images and masks together. For a controlled comparison, keep the seed and all training options fixed and use a separate output directory for each run:

```bash
python -m person_segmentation.train --data data/PennFudanPed_256 --output results/flip --augmentation flip --seed 42 --epochs 30 --batch-size 4 --image-size 256 --base-channels 16 --learning-rate 0.001 --device cpu
python -m person_segmentation.train --data data/PennFudanPed_256 --output results/zoom --augmentation zoom --seed 42 --epochs 30 --batch-size 4 --image-size 256 --base-channels 16 --learning-rate 0.001 --device cpu
python -m person_segmentation.train --data data/PennFudanPed_256 --output results/flip_zoom --augmentation both --seed 42 --epochs 30 --batch-size 4 --image-size 256 --base-channels 16 --learning-rate 0.001 --device cpu
```

Training reads the fixed split from `report/dataset_analysis/split_seed42.json`; only train and validation images are used during fitting and checkpoint selection. The test split remains untouched for final evaluation. The best validation Dice checkpoint is saved as `best.pt`; `history.csv` records per-epoch loss, Dice, and IoU, and `config.json` records run settings. Training defaults to CUDA when available, then Apple Metal, then CPU. Common options include `--epochs`, `--batch-size`, `--image-size`, `--learning-rate`, `--lr-scheduler` (`none` or `cosine`), `--base-channels`, `--cache`, and `--device`.

The initial 30-epoch CPU no-augmentation baseline (`base-channels 16`) reached validation Dice 0.8131 / IoU 0.6851 and test Dice 0.8471 / IoU 0.7348. After enabling CUDA GPU training, updating small zoom to zoom-in `(1.0, 1.15)` at `p = 0.5` from raw images, and training for 50 epochs with `--lr-scheduler cosine`, the 3-seed `none` → `both` (flip + small zoom) comparison improved mean validation Dice/IoU from `0.8078/0.6776` to `0.8298/0.7091` and mean test Dice/IoU from `0.8495/0.7385` to `0.8668/0.7649`. Adding training-split-only Copy-Paste augmentation (`person_segmentation.train_copypaste`, `290` cutouts, `p = 0.5`) further raised 3-seed mean validation Dice/IoU to `0.8514/0.7414` and test Dice/IoU to `0.8726/0.7740`. See [the baseline results note](report/baseline_results.md), [the augmentation comparison note](report/augmentation_comparison.md), and [the results discussion](report/results_discussion.md). Checkpoints and epoch histories are saved under `results/` (and consolidated under `results/all_project_results/`).

## Evaluate and predict

Evaluate a checkpoint on the held-out test split by default:

```bash
python -m person_segmentation.evaluate --data data/PennFudanPed_256 --checkpoint results/baseline/best.pt
```

Use `--split validation` to report validation metrics; use `--split all` only for whole-dataset diagnostics, not as an unbiased final score.

Generate a binary PNG mask for a single image:

```bash
python -m person_segmentation.predict --image path/to/image.png --checkpoint results/no_augmentation/best.pt --output results/prediction.png
```

## Metrics and experiment notes

Evaluation reports pixel accuracy, precision, recall, specificity, Dice/F1, and IoU from predictions thresholded at 0.5. Dice equals F1 for this binary segmentation setup. Pixel accuracy includes background pixels, so read it with Dice, IoU, precision, and recall. Evaluation JSON can be saved with `--output path/to/metrics.json`. The model uses a BCE + soft Dice training loss. Record the split seed, augmentation mode, image size, epochs, and validation scores when reporting results. The dataset is small, so results can vary with split choice and training settings; compare runs on the same split and reserve test metrics for final comparisons.
