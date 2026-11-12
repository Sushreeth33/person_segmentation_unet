# Held-out test prediction review

## Setup and aggregate scores

- Split: held-out test, `26` images (evaluated once after checkpoint selection; never used for tuning).
- Primary checkpoint inspected: `results/tuned_both_copypaste_seed42/best.pt` (`Flip + Small Zoom + Copy-Paste`, best epoch `40`, 16-channel U-Net, 50 epochs, cosine LR, `256 × 256` stretch input, probability threshold `0.5`).
- Pooled (micro) test scores: Dice **`0.8773`**, IoU **`0.7815`**, Precision `0.8372`, Recall `0.9215`, Pixel Accuracy `0.9603`.
- Macro (per-image) test means: Dice **`0.8737`**, IoU **`0.7838`**.
- Progression across 3 seeds (`42–44`): `none` (`0.8495 / 0.7385`) → `both` (`0.8668 / 0.7649`) → **`both + copypaste` (`0.8726 / 0.7740`)**, improving **`22/26`** held-out test images over baseline (`+3.61 pp` per-image IoU, `p = 0.00049`).

## Observed error patterns in test masks

Visual inspection of the four-panel error overlays (`report/test_error_analysis/overlays/` and `report/figures/test_baseline_vs_proposed_masks.png`) shows:

1. **Precision recovery on cluttered backgrounds (`PennPed00094.png`, `FudanPed00023.png`, `FudanPed00056.png`).**
   - In `PennPed00094.png`, `both + copypaste` improves Dice from `0.714` (`both`) to **`0.821`** (`IoU 0.696`), cutting false positives from `4,936` px (`both`) down to `2,613` px by suppressing both the shadowed tree/trash-bin false positive and the leashed dog (with remaining FPs concentrated on the orange rolling suitcase).
   - `FudanPed00056.png` rises from `0.778` Dice (`both`) to **`0.903`** Dice (`IoU 0.823`), cutting FPs from `3,656` px to `446` px.
2. **Unlabeled non-pedestrian people (annotation boundary cases).**
   - `FudanPed00029.png` remains the lowest-scoring image (`Dice 0.595`, `IoU 0.423`, `Precision 0.434`, `Recall 0.948`, `6,109` FP vs. `257` FN) because the model segments the *unannotated cyclist* on the left alongside the labeled walking pedestrian on the right.
3. **Low-contrast bright/dark boundaries and crowded overlap.**
   - `FudanPed00036.png` (`Dice 0.707`, `IoU 0.546`, `2,232` FP, `1,619` FN) retains some false positives in dark glass-door shadows and missed pixels where a white shirt meets a pale building wall, though FP pixels drop by nearly `900` px compared with `both` (`3,122 → 2,232`).
4. **Distant small figures at `256 × 256` resolution vs. clean silhouettes.**
   - `FudanPed00058.png` (`Dice 0.842`, `IoU 0.727`, up `+0.147` IoU over `none`'s `0.580`) recovers the foreground woman's legs and mid-ground pedestrians, while tiny background pedestrians near the far edges remain challenging at `256 × 256`.
   - Clean pedestrian scenes such as `FudanPed00072.png` (`Dice 0.961`, `IoU 0.925`) and `FudanPed00023.png` (`Dice 0.957`, `IoU 0.917`) maintain sharp boundaries with minimal edge error.

## Files

- `test_overlays_sorted_by_dice.png`: contact sheet of all `26` test overlays (`tuned_both_copypaste_seed42`), ordered from lowest to highest Dice. Legend: green = TP, red = FP, blue = FN.
- `overlays/`: full-resolution four-panel overlay cards for each test image.
- `per_image_metrics.csv` and `summary.json`: per-image and aggregate test statistics.
- Standalone binary masks (original image dimensions, `0` = black background, `255` = white person): `report/test_binary_predictions/` and `report/test_binary_predictions.zip`.
- Multi-model visual comparison grid (`none` vs. `both` vs. `both + copypaste`): `report/figures/test_baseline_vs_proposed_masks.png`.
