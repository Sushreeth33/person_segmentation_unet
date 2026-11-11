# Consolidated project results

This folder gathers the model outputs and analysis artifacts in one place. It is a local results bundle under the Git-ignored `results/` directory; copies here do not replace the original run folders.

## Main score summary

Dice and IoU are the primary overlap scores. Validation is used for checkpoint comparison; test results are recorded for the baseline, selected seed-42 horizontal-flip model, and the required flip-plus-small-zoom model across three training seeds. Test scores are descriptive final evaluations and should not be used for tuning.

| Run | Seed | Resize | Best epoch | Validation Dice | Validation IoU | Test Dice | Test IoU | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Baseline, no augmentation | 42 | Stretch | 22 | 0.8131 | 0.6851 | 0.8471 | 0.7348 | Complete |
| Horizontal flip | 42 | Stretch | 29 | 0.8278 | 0.7062 | 0.8573 | 0.7503 | Complete; selected test evaluation |
| No augmentation | 43 | Stretch | 25 | 0.8050 | 0.6737 | — | — | Validation repeat complete |
| Horizontal flip | 43 | Stretch | 27 | 0.8135 | 0.6856 | — | — | Validation repeat complete |
| No augmentation | 44 | Stretch | 24 | 0.8071 | 0.6765 | — | — | Validation repeat complete |
| Horizontal flip | 44 | Stretch | 28 | 0.8346 | 0.7162 | — | — | Validation repeat complete |
| Horizontal flip, letterbox | 42 | Letterbox | 24 | 0.8187 | 0.6930 | — | — | Validation comparison complete |
| Zoom only | 42 | Stretch | — | — | — | — | — | Incomplete: config only; no checkpoint, history, or metrics |
| Flip + small zoom | 42 | Stretch | 29 | 0.8104 | 0.6812 | 0.8563 | 0.7487 | Proposed model |
| Flip + small zoom | 43 | Stretch | 20 | 0.8061 | 0.6752 | 0.8296 | 0.7087 | Proposed model repeat |
| Flip + small zoom | 44 | Stretch | 29 | 0.8123 | 0.6839 | 0.8440 | 0.7300 | Proposed model repeat |

The three-seed validation mean for no augmentation was Dice 0.8084 / IoU 0.6784. With horizontal flips it was Dice 0.8253 / IoU 0.7027. The mean gains were 0.0169 Dice and 0.0242 IoU. The same fixed train/validation split was used across seeds.

For the seed-42 stretch model, standalone test metrics were:

| Run | Pixel accuracy | Precision | Recall | Specificity | Dice / F1 | IoU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, no augmentation | 0.9509 | 0.8143 | 0.8828 | 0.9633 | 0.8471 | 0.7348 |
| Horizontal flip | 0.9542 | 0.8243 | 0.8931 | 0.9653 | 0.8573 | 0.7503 |

The required flip-plus-small-zoom model was repeated with training seeds 42, 43, and 44 on the same fixed data split. Mean validation Dice/IoU were 0.8096/0.6801; mean test Dice/IoU were 0.8433/0.7292. Its validation mean is only 0.0012 Dice and 0.0017 IoU above the three-seed no-augmentation mean. These are across-seed means on one fixed split; see `reports/augmentation_comparison.md` for interpretation.

## Folder contents

- `models/`: copied configs, epoch histories, best checkpoints, and saved metric JSON for all model runs. `augmentation_zoom/` explicitly records the unfinished run.
- `models/augmentation_flip_small_zoom_seed{42,43,44}/`: selected checkpoints and metrics for the proposed model across three training seeds.
- `predictions/validation_binary_masks/`: 25 standalone 8-bit grayscale PNG predictions at original image dimensions. Pixel 0 is black background; pixel 255 is white person.
- `visualizations/validation_error_analysis/`: original stretch model's validation overlays, per-image metrics, and summary.
- `visualizations/resize_letterbox_validation_analysis/`: letterbox experiment's validation overlays and per-image metrics.
- `dataset_analysis/`: dataset audit summary, split manifest, per-image statistics, and visualizations.
- `reports/`: copies of the model results, augmentation comparison, resize comparison, and U-Net/loss/metrics explanations.
- `model_metrics.csv`: one-row-per-run comparison including validation/test metric fields where available.

## Source locations

- Original per-run artifacts remain in the sibling folders under `results/`.
- Code is in `src/person_segmentation/`.
- Tracked project reports remain in `report/`.

Because `/results/` is excluded by `.gitignore`, this consolidated bundle is currently local to this computer and is not part of the GitHub repository.

Required proposed model (flip + small 0.9–1.1 zoom) is complete. Its run files are under `models/augmentation_flip_small_zoom_seed42/`; validation pooled Dice/IoU are 0.8104/0.6812 and test pooled Dice/IoU are 0.8563/0.7487. See `reports/augmentation_comparison.md` for comparison and interpretation.
