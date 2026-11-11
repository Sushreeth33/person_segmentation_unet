# Held-out test prediction review

## Setup and aggregate scores

- Split: held-out test, `26` images (evaluated once after checkpoint selection; never used for tuning).
- Primary checkpoint inspected: `results/tuned_both_seed42/best.pt` (`flip + small zoom`, best epoch `37`, 16-channel U-Net, 50 epochs, cosine LR, `256 × 256` stretch input, probability threshold `0.5`).
- Pooled (micro) test scores: Dice **`0.8676`**, IoU **`0.7662`**, Precision `0.8177`, Recall `0.9240`, Pixel Accuracy `0.9565`.
- Macro (per-image) test means: Dice **`0.8602`**, IoU **`0.7637`**.
- Compared with `results/tuned_none_seed42/best.pt` (`0.8632` micro Dice / `0.7593` micro IoU), `tuned_both_seed42` preserves silhouette continuity and recovers missed limbs/torsos, while across all three seeds (`42–44`) `both` improves mean test Dice/IoU from `0.8495 / 0.7385` to **`0.8668 / 0.7649`** (`20/26` test images improved in 3-seed per-image IoU, `p = 0.0013`).

## Observed error patterns in test masks

Visual inspection of the four-panel error overlays (`report/test_error_analysis/overlays/` and `report/figures/test_baseline_vs_proposed_masks.png`) shows four clear, concrete patterns:

1. **Unlabeled non-pedestrian or cropped people (annotation boundary cases).**
   - `FudanPed00029.png` has the lowest test score (`Dice 0.607`, `IoU 0.436`, `Precision 0.448`, `Recall 0.942`, `5,724` FP vs. `285` FN). Visually, the model cleanly segments both the annotated walking pedestrian on the right and an *unannotated person riding a bicycle* on the left; because Penn-Fudan ground truth omits the cyclist, that region counts as false positives. Compared with `tuned_none_seed42` (`Dice 0.539`, `IoU 0.369`), `tuned_both_seed42` also removes a spurious middle-background blob and fills in the pedestrian's head.
   - `FudanPed00008.png` (`Dice 0.860`, `IoU 0.754`, vs. `none` `0.880` / `0.786`) shows a similar effect on the lower-left frame edge, where partially cropped walking figures were not labeled in ground truth.
2. **Carried objects, pets, and cluttered shadows.**
   - `PennPed00094.png` (`Dice 0.714`, `IoU 0.555`, `Recall 0.941`, `Precision 0.575`, `4,936` FP) captures nearly the entire woman accurately, with false positives concentrated on the orange rolling suitcase, small leashed dog, and a shadowed tree/bin region behind her.
3. **Low-contrast bright/dark boundaries and crowded overlap.**
   - `FudanPed00036.png` (`Dice 0.685`, `IoU 0.521`, `3,122` FP, `1,367` FN) exhibits false negatives on a white torso blending into a pale wall on the right, paired with false positives spilling into dark glass-door shadows behind overlapping figures.
4. **Distant small figures at `256 × 256` resolution vs. strong foreground recovery.**
   - `FudanPed00058.png` (`Dice 0.866`, `IoU 0.764`, up sharply from `tuned_none_seed42` `Dice 0.734` / `IoU 0.580`, `+0.184` IoU) demonstrates how flip + small zoom recovers an entire missing lower-leg region on the main subject (`FN` drops from `3,703` to `1,338`), while remaining misses are mostly tiny background pedestrians (`< 15 px` wide) near the far edges.
   - Clean uncluttered pedestrians such as `FudanPed00072.png` (`Dice 0.965`, `IoU 0.932`) and `FudanPed00023.png` (`Dice 0.955`, `IoU 0.914`) show tight silhouettes with only `1–2 px` edge uncertainty around shoes and hair.

## Files

- `test_overlays_sorted_by_dice.png`: contact sheet of all `26` test overlays (`tuned_both_seed42`), ordered from lowest to highest Dice. Legend: green = TP, red = FP, blue = FN.
- `overlays/`: full-resolution four-panel overlay cards for each test image.
- `per_image_metrics.csv` and `summary.json`: per-image and aggregate test statistics.
- Standalone binary masks (original image dimensions, `0` = black background, `255` = white person): `report/test_binary_predictions/` and `report/test_binary_predictions.zip`.
- Baseline-vs-proposed visual comparison grid: `report/figures/test_baseline_vs_proposed_masks.png`.
