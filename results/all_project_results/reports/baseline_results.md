# Baseline U-Net results

## Model and training setup

- Task: binary person segmentation; one output logit per pixel.
- Architecture: five-level U-Net with two convolution, batch-normalization, and ReLU operations per block; max-pooling encoder; transposed-convolution decoder with skip connections; 16 base channels.
- Input: 256 × 256 RGB images; no augmentation.
- Loss: equal-weight BCE-with-logits and soft Dice loss.
- Optimizer: Adam, learning rate 0.001.
- Run: 30 epochs, batch size 4, random seed 42, CPU.
- Split: fixed seed-42 manifest; 119 train, 25 validation, 26 test.
- Checkpoint selection: highest validation Dice.

## Results

| Evaluation set | Images | Loss | Dice | IoU |
|---|---:|---:|---:|---:|
| Best validation checkpoint (epoch 22) | 25 | 0.1995 | 0.8131 | 0.6851 |
| Held-out test | 26 | 0.1623 | 0.8471 | 0.7348 |

## Additional pixel-level scores

All scores below use a 0.5 probability threshold and aggregate pixel counts across the complete split.

| Evaluation set | Pixel accuracy | Precision | Recall / sensitivity | Specificity | F1 / Dice |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9346 | 0.8086 | 0.8177 | 0.9592 | 0.8131 |
| Test | 0.9509 | 0.8143 | 0.8828 | 0.9633 | 0.8471 |

For this binary task, F1 and Dice are the same score. Pixel accuracy is the share of all pixels classified correctly, including both person and background pixels. It can look high when background occupies most of an image; Penn-Fudan averages about 17.3% person pixels, so accuracy should be read alongside foreground metrics such as Dice, IoU, precision, and recall.

Training finished at epoch 30 with training Dice 0.9283. The selected validation checkpoint is from epoch 22. Because Penn-Fudan is small and this is one fixed split and one seed, treat these as initial CPU baseline results for comparison, not as a robust estimate across multiple runs.

## Updated 50-epoch GPU cosine-LR baseline (`tuned_none`)

In the subsequent tuned comparison (50 epochs, `--lr-scheduler cosine` with `eta_min = 1e-5`, `--cache`, single-step stretch resize from raw `data/PennFudanPed/`, RTX 4060 GPU), the no-augmentation baseline was trained across seeds `42`, `43`, and `44`:

| Seed | Best epoch | Val Precision | Val Recall | Val Dice | Val IoU | Test Precision | Test Recall | Test Dice | Test IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 28 | 0.8390 | 0.7752 | 0.8058 | 0.6748 | 0.8572 | 0.8693 | 0.8632 | 0.7593 |
| 43 | 29 | 0.8134 | 0.7926 | 0.8029 | 0.6707 | 0.8223 | 0.8754 | 0.8480 | 0.7361 |
| 44 | 43 | 0.8445 | 0.7870 | 0.8147 | 0.6874 | 0.8297 | 0.8450 | 0.8373 | 0.7201 |
| **Mean ± SD** | — | 0.8323 | 0.7849 | **0.8078 ± 0.0062** | **0.6776 ± 0.0087** | 0.8364 | 0.8632 | **0.8495 ± 0.0130** | **0.7385 ± 0.0197** |

See [the augmentation comparison note](augmentation_comparison.md) for the full 4-way comparison (`none`, `zoom`, `flip`, `both`).

## Files

- Training and evaluation code: `src/person_segmentation/train.py`, `src/person_segmentation/evaluate.py`, `src/person_segmentation/model.py`.
- Per-epoch metrics and run settings: `results/baseline/history.csv`, `results/baseline/config.json`.
- Expanded validation and test metrics: `results/baseline/validation_metrics.json`, `results/baseline/test_metrics.json`.
- Best model weights: `results/baseline/best.pt`.

The `results/` directory is ignored by Git, so checkpoints and per-epoch output remain local unless intentionally shared separately.
