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

## S.No. 4 — Baseline Model Implementation

- [x] Implement the baseline U-Net architecture
- [x] Configure binary segmentation loss and optimizer
- [x] Train a no-augmentation baseline on the fixed training split
- [x] Select the checkpoint using validation Dice
- [x] Evaluate the selected checkpoint on the held-out test split
- [x] Record baseline settings and metrics

**Progress: baseline and required augmentation comparison complete.** The 16-channel U-Net was trained for 30 epochs on CPU using 256 × 256 inputs. The no-augmentation baseline's best validation checkpoint was from epoch 22 (Dice 0.8131, IoU 0.6851); its held-out test scores were Dice 0.8471 and IoU 0.7348. A controlled horizontal-flip run used the same split, seed, model size, optimizer, and epoch budget. Its best validation checkpoint was from epoch 29 (Dice 0.8278, IoU 0.7062), gains of 1.47 and 2.11 percentage points. Test data was not used for checkpoint selection. Horizontal flip plus small zoom was subsequently trained and repeated with seeds 43 and 44; see `report/augmentation_comparison.md` for details. The zoom-only variant remains unrun because it is not required by the project statement.

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

### Optional extensions (not requested by the project statement)

- [ ] Try a higher-resolution input or a pretrained encoder.
- [ ] Implement and compare person-instance Copy-Paste augmentation using validation metrics.

Across seeds 42, 43, and 44, horizontal flips improved validation Dice in each matched run. Mean Dice rose from 0.8084 to 0.8253 (+1.69 points), and mean IoU rose from 0.6784 to 0.7027 (+2.42 points). These runs used one fixed split, so they measure training-seed variability rather than split-to-split variability. Detailed results are in `report/augmentation_comparison.md`.

The seed-42 flip checkpoint was evaluated once on the 26-image test split: Dice 0.8573 and IoU 0.7503, versus 0.8471 and 0.7348 for the no-augmentation baseline. The test scores are now final reporting metrics and should not guide further model tuning.

Validation error review is saved in `report/validation_error_analysis/summary.md`, with per-image overlays, a Dice-sorted contact sheet, and pixel-level counts. The main observed issues are background false positives in cluttered scenes, missed person pixels in some images, and errors around thin boundaries/limbs.

The aspect-preserving 256 × 256 letterbox run scored validation Dice 0.8187 and IoU 0.6930, below the matched stretch model's 0.8278 and 0.7062. Pixel accuracy rose because padding adds background pixels. Keep stretch as the current preprocessing choice; details and overlays are in `report/resize_comparison.md` and `report/resize_letterbox_validation_analysis/`.

The U-Net architecture, exact 50/50 BCE-plus-soft-Dice training loss, pooled thresholded Dice/IoU evaluation formulas, code locations, and related papers are documented in `report/model_loss_and_metrics.md`.

The seed-42 horizontal-flip checkpoint's 25 validation predictions are exported as original-size 8-bit grayscale masks in `report/validation_binary_predictions/` (0=black background, 255=white person). A ZIP copy is at `report/validation_binary_predictions.zip`.

The required proposed model was trained with horizontal flips and a small 0.9–1.1 scale zoom, keeping the baseline setup fixed. Its best validation checkpoint was epoch 29 (pooled Dice 0.8104, IoU 0.6812); test Dice/IoU were 0.8563/0.7487. The combined model did not beat the baseline on validation Dice, so the results do not show a validation improvement from the proposed augmentation in this run. Full comparison is in `report/augmentation_comparison.md`; run outputs are in `results/augmentation_flip_small_zoom_seed42/` (local, Git-ignored).

The required flip-plus-small-zoom model was repeated with training seeds 43 and 44 on the same fixed split and 30-epoch budget. The best checkpoints were epoch 20 (seed 43) and epoch 29 (seed 44). Validation Dice/IoU were 0.8061/0.6752 and 0.8123/0.6839; test Dice/IoU were 0.8296/0.7087 and 0.8440/0.7300. Across seeds 42–44, mean validation Dice/IoU were 0.8096/0.6801 and mean test Dice/IoU were 0.8433/0.7292. These runs measure training-seed variability only; detailed metrics and artifacts are in `report/augmentation_comparison.md` and the three seed-specific folders under `results/` (Git-ignored).

All model run folders and supporting outputs have also been copied into the local consolidated bundle `results/all_project_results/`, with a score table, model artifacts, prediction masks, validation visualizations, dataset audit outputs, and report copies. The bundle is under the Git-ignored `results/` folder, so it is not pushed to GitHub by default.
