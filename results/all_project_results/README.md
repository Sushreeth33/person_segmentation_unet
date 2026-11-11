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

Because `/results/` is listed in `.gitignore`, raw training folders remain local unless force-added; this consolidated `results/all_project_results/` bundle is tracked in Git so checkpoints, metrics, and reports are preserved on GitHub.

## Phase 2: Tuned GPU 4-way comparison (`50` epochs, cosine LR, `1.0–1.15` zoom at `p = 0.5`)

After diagnosing zoom-out black padding (`0.9–1.1` at `p = 1.0`), double bilinear resampling, and 30-epoch under-convergence in Phase 1, CUDA GPU training was enabled (RTX 4060 Laptop GPU) and all four modes (`none`, `zoom`, `flip`, `both`) were trained across seeds `42`, `43`, and `44` from raw `data/PennFudanPed/` using `--lr-scheduler cosine` (`eta_min = 1e-5`) and `--cache`:

| Run (`models/tuned_*`) | Seed | Best epoch | Val Precision | Val Recall | Val Dice | Val IoU | Test Precision | Test Recall | Test Dice | Test IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tuned_none_seed42` | 42 | 28 | 0.8390 | 0.7752 | 0.8058 | 0.6748 | 0.8572 | 0.8693 | 0.8632 | 0.7593 |
| `tuned_none_seed43` | 43 | 29 | 0.8134 | 0.7926 | 0.8029 | 0.6707 | 0.8223 | 0.8754 | 0.8480 | 0.7361 |
| `tuned_none_seed44` | 44 | 43 | 0.8445 | 0.7870 | 0.8147 | 0.6874 | 0.8297 | 0.8450 | 0.8373 | 0.7201 |
| `tuned_zoom_seed42` | 42 | 28 | 0.8348 | 0.8131 | 0.8238 | 0.7004 | 0.8388 | 0.8798 | 0.8588 | 0.7525 |
| `tuned_zoom_seed43` | 43 | 37 | 0.8441 | 0.7854 | 0.8137 | 0.6859 | 0.8247 | 0.8804 | 0.8517 | 0.7417 |
| `tuned_zoom_seed44` | 44 | 43 | 0.8226 | 0.8032 | 0.8128 | 0.6846 | 0.8173 | 0.8816 | 0.8483 | 0.7365 |
| `tuned_flip_seed42` | 42 | 40 | 0.8235 | 0.8308 | 0.8271 | 0.7052 | 0.8520 | 0.9091 | 0.8796 | 0.7851 |
| `tuned_flip_seed43` | 43 | 42 | 0.8286 | 0.8223 | 0.8254 | 0.7027 | 0.8417 | 0.8909 | 0.8656 | 0.7630 |
| `tuned_flip_seed44` | 44 | 32 | 0.8047 | 0.8584 | 0.8307 | 0.7104 | 0.7709 | 0.9315 | 0.8436 | 0.7295 |
| `tuned_both_seed42` | 42 | 37 | 0.8048 | 0.8769 | 0.8393 | 0.7231 | 0.8177 | 0.9240 | 0.8676 | 0.7662 |
| `tuned_both_seed43` | 43 | 42 | 0.8220 | 0.8225 | 0.8223 | 0.6982 | 0.8369 | 0.8949 | 0.8649 | 0.7620 |
| `tuned_both_seed44` | 44 | 43 | 0.8205 | 0.8351 | 0.8277 | 0.7061 | 0.8252 | 0.9151 | 0.8678 | 0.7665 |

Three-seed `Mean ± sample SD` summary (`none` → `both`):
- **Validation:** `none` `0.8078 ± 0.0062` Dice / `0.6776 ± 0.0087` IoU → `both` **`0.8298 ± 0.0087` Dice / `0.7091 ± 0.0127` IoU** (`+2.20` Dice pp / `+3.15` IoU pp).
- **Held-out test (`26` images):** `none` `0.8495 ± 0.0130` Dice / `0.7385 ± 0.0197` IoU → `both` **`0.8668 ± 0.0016` Dice / `0.7649 ± 0.0025` IoU** (`+1.73` Dice pp / `+2.64` IoU pp; `20/26` test images improved, per-image IoU `p = 0.00130`). See `reports/augmentation_comparison.md` for details.
