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

**Progress: 12/12 complete.** The 170 image/mask pairs were audited with no missing files or dimension mismatches. Original masks are grayscale with background value 0 and person instance IDs 1–8. All positive IDs are merged to foreground; resized masks contain only 0 and 1. Images are resized to 256 × 256 and RGB values are scaled to [0, 1]. The seed-42 split contains 119 train, 25 validation, and 26 test images. Dataset files remain local in the ignored `data/` folder.

Audit artifacts: `report/dataset_analysis/summary.md`, `summary.json`, `per_image.csv`, `split_seed42.json`, `sample_images_masks.png`, and `instances_per_image.png`. The audit and split implementation is in `src/person_segmentation/analyze_dataset.py`; the filename-aware loader and split validation are in `src/person_segmentation/data.py`.
