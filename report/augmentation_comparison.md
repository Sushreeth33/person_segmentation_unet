# Augmentation comparison

## Experiment

All runs used the same fixed 119/25/26 train/validation/test split, 16-channel U-Net, 256 × 256 inputs, Adam at 0.001, batch size 4, CPU, and 30 epochs. We compared no augmentation with horizontal flips at seeds 42, 43, and 44. The only difference within each seed-matched pair was training augmentation. Flips were applied to training image/mask pairs; validation images were not augmented. Each checkpoint was selected independently by best validation Dice.

The test split was not evaluated for this comparison.

## Validation results across seeds

Each seed used the same train/validation split and 30-epoch budget. A checkpoint was selected independently for each run by its highest validation Dice.

| Seed | No-augmentation best epoch | Dice | IoU | Flip best epoch | Dice | IoU | Dice change | IoU change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 22 | 0.8131 | 0.6851 | 29 | 0.8278 | 0.7062 | +0.0147 | +0.0211 |
| 43 | 25 | 0.8050 | 0.6737 | 27 | 0.8135 | 0.6856 | +0.0084 | +0.0119 |
| 44 | 24 | 0.8071 | 0.6765 | 28 | 0.8346 | 0.7162 | +0.0276 | +0.0396 |
| **Mean** | — | **0.8084** | **0.6784** | — | **0.8253** | **0.7027** | **+0.0169** | **+0.0242** |

Horizontal flips improved validation Dice and IoU in all three seed-matched comparisons. The mean gain was 1.69 percentage points in Dice and 2.42 points in IoU. Precision/recall tradeoffs varied by seed; on average the flip runs gained recall. This supports keeping horizontal flips for the next stage, but three seeds on one fixed validation split are still a limited check and do not establish performance on unseen data.

## One-time held-out test evaluation

After selecting horizontal flips from validation results, we evaluated the seed-42 flip checkpoint once on the held-out 26-image test split. At threshold 0.5, the flip model improved over the seed-42 no-augmentation baseline on the reported metrics:

| Test setup | Loss | Pixel accuracy | Precision | Recall | Specificity | Dice / F1 | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| No augmentation | 0.1623 | 0.9509 | 0.8143 | 0.8828 | 0.9633 | 0.8471 | 0.7348 |
| Horizontal flip | 0.1563 | 0.9542 | 0.8243 | 0.8931 | 0.9653 | 0.8573 | 0.7503 |

The flip checkpoint gained 1.02 percentage points in Dice and 1.55 points in IoU on this test split. This is a favorable result for the selected configuration, but it is based on one small test set and one training seed. The test scores are reported as final evaluation; future model choices should continue to use validation data rather than these test results.

The zoom-only and combined flip-plus-zoom runs were deferred because CPU training was slow. The next step is qualitative error review using validation predictions, then decide whether to try zoom augmentation or a pretrained encoder.

## Reproducibility files

- Flip model checkpoint: `results/augmentation_flip/best.pt` (ignored by Git).
- Flip training settings and epoch history: `results/augmentation_flip/config.json` and `results/augmentation_flip/history.csv` (ignored by Git).
- Re-evaluated validation metrics: `results/augmentation_flip/validation_metrics.json` (ignored by Git).
- One-time held-out test metrics: `results/augmentation_flip/test_metrics.json` (ignored by Git).
- Repeat-seed checkpoints, settings, and histories: `results/augmentation_repeats/seed{42,43,44}_{none,flip}/` (ignored by Git; seed 42 artifacts are in the baseline and initial flip folders).
- Training implementation: `src/person_segmentation/data.py` and `src/person_segmentation/train.py`.
