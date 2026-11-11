# Augmentation comparison

## Experiment

All runs used the same fixed 119/25/26 train/validation/test split, 16-channel U-Net, 256 × 256 inputs, Adam at 0.001, batch size 4, CPU, and 30 epochs. We compared no augmentation with horizontal flips at seeds 42, 43, and 44. The only difference within each seed-matched pair was training augmentation. Flips were applied to training image/mask pairs; validation images were not augmented. Each checkpoint was selected independently by best validation Dice.

The section below reports the additional required proposed-model run: horizontal flips plus small zoom. This run uses the same fixed split and model/training settings. The zoom scale was set to 0.9–1.1 to represent a small zoom.

## Validation results across seeds

Each seed used the same train/validation split and 30-epoch budget. A checkpoint was selected independently for each run by its highest validation Dice.

| Seed | No-augmentation best epoch | Dice | IoU | Flip best epoch | Dice | IoU | Dice change | IoU change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 22 | 0.8131 | 0.6851 | 29 | 0.8278 | 0.7062 | +0.0147 | +0.0211 |
| 43 | 25 | 0.8050 | 0.6737 | 27 | 0.8135 | 0.6856 | +0.0084 | +0.0119 |
| 44 | 24 | 0.8071 | 0.6765 | 28 | 0.8346 | 0.7162 | +0.0276 | +0.0396 |
| **Mean** | — | **0.8084** | **0.6784** | — | **0.8253** | **0.7027** | **+0.0169** | **+0.0242** |

Horizontal flips improved validation Dice and IoU in all three seed-matched comparisons. The mean gain was 1.69 percentage points in Dice and 2.42 points in IoU. Precision/recall tradeoffs varied by seed; on average the flip runs gained recall. This supports keeping horizontal flips for the next stage, but three seeds on one fixed validation split are still a limited check and do not establish performance on unseen data.

## Required proposed model: horizontal flip plus small zoom

The proposed model used both random horizontal flips (50% chance) and random scale in the range 0.9–1.1 during training. All other settings matched the seed-42 baseline: the same 119/25/26 split, compact 16-channel U-Net, BCE-plus-soft-Dice loss, Adam at 0.001, batch size 4, 256 × 256 inputs, 30 epochs, and seed 42. Checkpoint selection used validation Dice. Evaluation uses threshold 0.5.

| Seed-42 setup | Best epoch | Validation pooled Dice | Validation pooled IoU | Validation mean image Dice ± sample SD | Test pooled Dice | Test pooled IoU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, no augmentation | 22 | 0.8131 | 0.6851 | 0.8171 ± 0.0915 | 0.8471 | 0.7348 |
| Horizontal flip only | 29 | 0.8278 | 0.7062 | 0.8331 ± 0.0864 | 0.8573 | 0.7503 |
| Horizontal flip + small zoom | 29 | 0.8104 | 0.6812 | 0.8146 ± 0.0994 | 0.8563 | 0.7487 |

The combined augmentation's validation Dice is 0.27 percentage points below the baseline and 1.74 points below flip-only. On the test split it is 0.91 points above the baseline and 0.10 points below flip-only in Dice. It is 1.39 points above baseline and 0.15 points below flip-only in IoU. The difference in validation and test ordering is a reminder that the test set is small; do not tune based on these test scores. The mean-image Dice and pooled Dice are both shown because the former weights each image equally, while the latter pools pixel counts.

The formal baseline-versus-proposed comparison is now complete. On this run, adding small zoom to horizontal flips did not improve validation Dice over the no-augmentation baseline; the flip-only model remains the strongest validation result. The problem statement's comparison should still report baseline and flip-plus-zoom as the main pair, with flip-only as a useful intermediate ablation.

### Proposed-model repeat runs across training seeds

To check whether the seed-42 result was unusually favorable, the same flip-plus-small-zoom experiment was repeated with seeds 43 and 44. The filename split remained fixed at seed 42 (119/25/26 images); only model initialization, training shuffle, and augmentation randomness changed. Each run trained for 30 epochs and selected its checkpoint using validation Dice. The test split was evaluated after checkpoint selection for each run.

| Training seed | Best epoch | Validation Dice | Validation IoU | Test Dice | Test IoU |
|---:|---:|---:|---:|---:|---:|
| 42 | 29 | 0.8104 | 0.6812 | 0.8563 | 0.7487 |
| 43 | 20 | 0.8061 | 0.6752 | 0.8296 | 0.7087 |
| 44 | 29 | 0.8123 | 0.6839 | 0.8440 | 0.7300 |
| **Mean across seeds** | — | **0.8096** | **0.6801** | **0.8433** | **0.7292** |

Across these three training seeds, validation Dice ranged by about 0.0062 from lowest to highest, while test Dice ranged by about 0.0267. This indicates meaningful run-to-run variation on the small test set. The proposed model's three-seed mean validation Dice/IoU (0.8096/0.6801) is only 0.0012/0.0017 above the corresponding no-augmentation three-seed mean (0.8084/0.6784). These results do not support claiming that flip-plus-small-zoom reliably improves performance. The test results are descriptive checks of seed variability, not a basis for selecting a seed or tuning the model. The data split stayed the same, so this is not a measure of split-to-split variability.

Per-run artifacts are saved under `results/augmentation_flip_small_zoom_seed42/`, `results/augmentation_flip_small_zoom_seed43/`, and `results/augmentation_flip_small_zoom_seed44/` (local, Git-ignored). Each folder contains its configuration, training history, selected checkpoint, and validation/test metric JSON files.

## One-time held-out test evaluation

After selecting horizontal flips from validation results, we evaluated the seed-42 flip checkpoint once on the held-out 26-image test split. At threshold 0.5, the flip model improved over the seed-42 no-augmentation baseline on the reported metrics:

| Test setup | Loss | Pixel accuracy | Precision | Recall | Specificity | Dice / F1 | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| No augmentation | 0.1623 | 0.9509 | 0.8143 | 0.8828 | 0.9633 | 0.8471 | 0.7348 |
| Horizontal flip | 0.1563 | 0.9542 | 0.8243 | 0.8931 | 0.9653 | 0.8573 | 0.7503 |

The flip checkpoint gained 1.02 percentage points in Dice and 1.55 points in IoU on this test split. This is a favorable result for the selected configuration, but it is based on one small test set and one training seed. The test scores are reported as final evaluation; future model choices should continue to use validation data rather than these test results.

Zoom-only was not run as a separate ablation; it is not needed to complete the stated baseline-versus-proposed comparison. Further experiments such as Copy-Paste remain optional extensions and should be selected using validation data.

## Reproducibility files

- Flip model checkpoint: `results/augmentation_flip/best.pt` (ignored by Git).
- Flip training settings and epoch history: `results/augmentation_flip/config.json` and `results/augmentation_flip/history.csv` (ignored by Git).
- Re-evaluated validation metrics: `results/augmentation_flip/validation_metrics.json` (ignored by Git).
- One-time held-out test metrics: `results/augmentation_flip/test_metrics.json` (ignored by Git).
- Proposed flip-plus-small-zoom checkpoint, settings, histories, and validation/test metrics: `results/augmentation_flip_small_zoom_seed42/` (ignored by Git).
- Repeat-seed checkpoints, settings, and histories: `results/augmentation_repeats/seed{42,43,44}_{none,flip}/` (ignored by Git; seed 42 artifacts are in the baseline and initial flip folders).
- Training implementation: `src/person_segmentation/data.py` and `src/person_segmentation/train.py`.
