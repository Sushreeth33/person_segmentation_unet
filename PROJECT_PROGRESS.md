# Person Segmentation Project Progress

Checklist status based on the repository, the dataset audit, and the user's update that S.No. 1 and S.No. 2 are complete.

## Initial setup

- [x] Create GitHub repository
- [x] Initialize README and Python `.gitignore`
- [x] Clone repository in VS Code
- [x] Create Python virtual environment
- [x] Install initial Python dependencies

## S.No. 1 — Title, Introduction and Contribution

- [x] Finalize project title
- [x] Write project problem statement
- [x] Explain image and person segmentation
- [x] Write project motivation
- [x] Define project objectives
- [x] Document proposed contribution

**Progress: 6/6 complete (per user update).**

## S.No. 2 — Literature Review

- [x] Study FCN paper (Long et al., 2015)
- [x] Study U-Net paper (Ronneberger et al., 2015)
- [x] Study DeepLabv3+ paper (Chen et al., 2018)
- [x] Study data augmentation survey (Shorten & Khoshgoftaar, 2019)
- [x] Summarize existing methods and limitations
- [x] Document research motivation and proposed approach

**Progress: 6/6 complete (per user update).**

## S.No. 3 — Dataset and Preprocessing

- [x] Download Penn-Fudan Pedestrian dataset
- [x] Verify all 170 images and corresponding masks
- [x] Inspect image and mask dimensions
- [x] Visualize original images and instance masks
- [x] Understand instance-mask pixel values
- [x] Merge individual person masks into binary masks
- [x] Resize images and masks to 256 × 256
- [x] Normalize RGB images
- [x] Verify binary masks contain only 0 and 1
- [x] Create reproducible train/validation/test split
- [x] Calculate dataset statistics
- [x] Generate dataset visualization and analysis

**Progress: 12/12 complete.** The 170 image/mask pairs were audited with no missing files or dimension mismatches. Original masks are grayscale with background value 0 and person instance IDs 1–8. All positive IDs are merged to foreground; resized masks contain only 0 and 1. Images are resized to 256 × 256 and RGB values are scaled to [0, 1]. A saved 256 × 256 PNG copy of all images and binary masks is in the ignored `data/PennFudanPed_256/` folder. The seed-42 split contains 119 train, 25 validation, and 26 test images. Dataset files remain local in the ignored `data/` folder.

Audit artifacts: `report/dataset_analysis/summary.md`, `summary.json`, `per_image.csv`, `split_seed42.json`, `sample_images_masks.png`, and `instances_per_image.png`. The audit and split implementation is in `src/person_segmentation/analyze_dataset.py`; the 256 × 256 file export is in `src/person_segmentation/export_preprocessed.py`; the filename-aware loader and split validation are in `src/person_segmentation/data.py`.

## S.No. 4 — Model Implementation and Augmentation Comparison

- [x] Implement the baseline U-Net architecture
- [x] Configure binary segmentation loss and optimizer
- [x] Train a no-augmentation baseline on the fixed training split
- [x] Select the checkpoint using validation Dice
- [x] Evaluate the selected checkpoint on the held-out test split
- [x] Record baseline settings and metrics

**Progress: baseline and required augmentation comparison complete (Phase 1 CPU + Phase 2 tuned GPU).** The 16-channel U-Net was initially trained for 30 epochs on CPU using 256 × 256 inputs (`baseline` best val epoch 22: Dice 0.8131, IoU 0.6851, test Dice 0.8471, IoU 0.7348; `flip` best val epoch 29: Dice 0.8278, IoU 0.7062, test Dice 0.8573, IoU 0.7503; `both` with 0.9–1.1 zoom at `p = 1.0` averaged val Dice 0.8096 / IoU 0.6801 across seeds 42–44). After diagnosing zoom-out black padding, double resampling, and 30-epoch under-convergence, CUDA support (`torch 2.14.1+cu130`) was enabled on the RTX 4060 Laptop GPU and the zoom rule was updated to zoom-in `(1.0, 1.15)` at `p = 0.5` with single-step stretch resizing, 50 epochs, and cosine LR scheduling (`eta_min = 1e-5`). All four variants (`none`, `zoom`, `flip`, `both`) were then run across seeds 42–44 (`results/tuned_*`): `both` improved over `none` from val Dice/IoU `0.8078/0.6776` to `0.8298/0.7091` (+2.20 / +3.15 pp) and test Dice/IoU `0.8495/0.7385` to `0.8668/0.7649` (+1.73 / +2.64 pp, cutting test IoU seed SD from `±0.0197` to `±0.0025`). See `report/augmentation_comparison.md` for full tables and paired test statistics.

### Next

- [x] Repeat the no-augmentation and horizontal-flip runs with additional seeds to check whether the validation gain is stable.
- [x] Evaluate the selected horizontal-flip configuration once on the held-out test split.
- [x] Review validation prediction overlays and common errors.
- [x] Compare stretched resizing with aspect-preserving letterbox resizing on validation.
- [x] Document the U-Net, training loss, Dice, and IoU formulas.
- [x] Export standalone black-and-white prediction masks for the validation split.
- [x] Gather model runs, metrics, checkpoints, histories, prediction masks, and analysis artifacts in one results folder.
- [x] Complete the required horizontal-flip plus small-zoom proposed model comparison.
- [x] Repeat the required proposed model with training seeds 43 and 44 and record each validation/test result.
- [x] Run the full 4-way tuned GPU comparison (`none`, `zoom`, `flip`, `both`) across seeds 42, 43, and 44 with cosine LR annealing and zoom-in `1.0–1.15` (`p = 0.5`).
- [x] Generate comparison graphs and held-out test mask overlay visualizations for S.No. 6.
- [x] Draft complete editable project report following the prescribed structure (`report/Person_Segmentation_UNet_Report.docx`, plus PDF preview `report/Person_Segmentation_UNet_Report.pdf`).
- [ ] Finalize Title Page metadata / export final PDF, create 10-slide presentation, and assemble `MLproject_rollno_topic` submission folder (S.No. 8).

### Optional extensions (not requested by the project statement)

- [ ] Try a higher-resolution input or a pretrained encoder.
- [x] Implement and compare person-instance Copy-Paste augmentation using validation metrics.

Across seeds 42, 43, and 44, horizontal flips improved validation Dice in each matched run. Mean Dice rose from 0.8084 to 0.8253 (+1.69 points), and mean IoU rose from 0.6784 to 0.7027 (+2.42 points). These runs used one fixed split, so they measure training-seed variability rather than split-to-split variability. Detailed results are in `report/augmentation_comparison.md`.

The seed-42 flip checkpoint was evaluated once on the 26-image test split: Dice 0.8573 and IoU 0.7503, versus 0.8471 and 0.7348 for the no-augmentation baseline. The test scores are now final reporting metrics and should not guide further model tuning.

Validation error review is saved in `report/validation_error_analysis/summary.md`, with per-image overlays, a Dice-sorted contact sheet, and pixel-level counts. The main observed issues are background false positives in cluttered scenes, missed person pixels in some images, and errors around thin boundaries/limbs.

The aspect-preserving 256 × 256 letterbox run scored validation Dice 0.8187 and IoU 0.6930, below the matched stretch model's 0.8278 and 0.7062. Pixel accuracy rose because padding adds background pixels. Keep stretch as the current preprocessing choice; details and overlays are in `report/resize_comparison.md` and `report/resize_letterbox_validation_analysis/`.

The U-Net architecture, exact 50/50 BCE-plus-soft-Dice training loss, pooled thresholded Dice/IoU evaluation formulas, code locations, and related papers are documented in `report/model_loss_and_metrics.md`.

The seed-42 horizontal-flip checkpoint's 25 validation predictions are exported as original-size 8-bit grayscale masks in `report/validation_binary_predictions/` (0=black background, 255=white person). A ZIP copy is at `report/validation_binary_predictions.zip`.

The Phase 1 30-epoch CPU `both` runs (0.9–1.1 scale at `p = 1.0`, seeds 42–44) averaged validation Dice/IoU `0.8096/0.6801` and test Dice/IoU `0.8433/0.7292`.

In Phase 2, after diagnosing zoom-out black padding, double resampling, and 30-epoch under-convergence, all four variants (`none`, `zoom`, `flip`, `both`) were trained cleanly on GPU (RTX 4060, `50` epochs, `--lr-scheduler cosine` with `eta_min = 1e-5`, single-step raw stretch resizing, zoom-in `1.0–1.15` at `p = 0.5`) across seeds 42, 43, and 44 (`results/tuned_*`):
- Validation 3-seed means (`Dice / IoU`): `none` `0.8078 / 0.6776` → `zoom` `0.8168 / 0.6903` → `flip` `0.8277 / 0.7061` → `both` **`0.8298 / 0.7091`** (+2.20 / +3.15 pp).
- Held-out test 3-seed means (`Dice / IoU`): `none` `0.8495 ± 0.0130 / 0.7385 ± 0.0197` → `zoom` `0.8529 / 0.7436` → `flip` `0.8629 / 0.7592` → `both` **`0.8668 ± 0.0016 / 0.7649 ± 0.0025`** (+1.73 / +2.64 pp; `20/26` test images improved, per-image IoU `p = 0.00130`).

In Phase 3, training-only person-instance Copy-Paste augmentation (`src/person_segmentation/copypaste.py`, `train_copypaste.py`, `290` cutouts from the `119` training images) was evaluated across seeds 42, 43, and 44:
- `copypaste` only (`p = 0.5`): validation Dice/IoU `0.8491 / 0.7379`, held-out test Dice/IoU `0.8628 / 0.7588`.
- `both + copypaste` (`p = 0.5`): validation Dice/IoU **`0.8514 ± 0.0091 / 0.7414 ± 0.0138`**, held-out test Dice/IoU **`0.8726 ± 0.0045 / 0.7740 ± 0.0070`** (+2.31 / +3.55 pp over baseline; `22/26` test images improved, per-image IoU `p = 0.00049`).

All model run folders (Phase 1, Phase 2, and Phase 3 `tuned_*`), metrics, checkpoints, histories, prediction masks, validation/test visualizations, S.No. 5 & 6 comparison figures (`report/figures/`, `report/results_discussion.md`, `report/test_error_analysis/`), dataset audit outputs, the editable structured report (`report/Person_Segmentation_UNet_Report.docx`), and report copies are consolidated in `results/all_project_results/` (tracked explicitly in Git alongside `report/`).
