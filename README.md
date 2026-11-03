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

Training resizes images to a square input size (default 256 × 256). Penn-Fudan mask filenames append `_mask` to the image stem. Original masks use 0 for background and positive integer IDs for person instances; preprocessing merges every positive ID into one person foreground class. RGB values are scaled to [0, 1]. Images are resized with bilinear interpolation and masks with nearest-neighbor interpolation.

## Train

```bash
python -m person_segmentation.train --data data/PennFudanPed --output results/no_augmentation --augmentation none
```

Available augmentation modes are `none`, `flip`, `zoom`, and `both`. Horizontal flips and random zooms are applied to images and masks together. For a controlled comparison, keep the seed and all training options fixed and use a separate output directory for each run:

```bash
python -m person_segmentation.train --data data/PennFudanPed --output results/flip --augmentation flip --seed 42
python -m person_segmentation.train --data data/PennFudanPed --output results/zoom --augmentation zoom --seed 42
python -m person_segmentation.train --data data/PennFudanPed --output results/flip_zoom --augmentation both --seed 42
```

Training reads the fixed split from `report/dataset_analysis/split_seed42.json`; only train and validation images are used during fitting and checkpoint selection. The test split remains untouched for final evaluation. The best validation Dice checkpoint is saved as `best.pt`; `history.csv` records per-epoch loss, Dice, and IoU, and `config.json` records run settings. Training defaults to CUDA when available, then Apple Metal, then CPU. Common options include `--epochs`, `--batch-size`, `--image-size`, `--learning-rate`, `--base-channels`, and `--device`.

## Evaluate and predict

Evaluate a checkpoint on the held-out test split by default:

```bash
python -m person_segmentation.evaluate --data data/PennFudanPed --checkpoint results/no_augmentation/best.pt
```

Use `--split validation` to report validation metrics; use `--split all` only for whole-dataset diagnostics, not as an unbiased final score.

Generate a binary PNG mask for a single image:

```bash
python -m person_segmentation.predict --image path/to/image.png --checkpoint results/no_augmentation/best.pt --output results/prediction.png
```

## Metrics and experiment notes

Dice and intersection-over-union (IoU) are calculated from thresholded predictions at 0.5. The model uses a BCE + soft Dice training loss. Record the split seed, augmentation mode, image size, epochs, and validation scores when reporting results. The dataset is small, so results can vary with split choice and training settings; compare runs on the same split and avoid treating validation scores as independent test results.
