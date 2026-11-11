# Methodology, Ablation, and Comparative Analysis (S.No. 5 & S.No. 6 Summary)

This note consolidates the proposed methodology, exact rules/formulas, 4-way ablation table, visual figures, and held-out test-mask interpretation for the finalized U-Net (`none`, `zoom`, `flip`, `both`) pipeline.

## 1. Proposed Methodology and Rules (S.No. 5)

- **Backbone & Objective (kept fixed across all runs):** Compact 5-level U-Net (`base_channels = 16`, `256 × 256` stretch RGB inputs scaled to `[0, 1]`, merged binary foreground person mask), trained with equal-weight `0.5 * BCEWithLogits + 0.5 * SoftDice` loss and Adam (`lr = 0.001`, batch size `4`) on fixed split `split_seed42.json` (`119` train / `25` val / `26` test). See [model_loss_and_metrics.md](model_loss_and_metrics.md) for full equations.
- **Explored Change (`both` = Horizontal Flip + Small Zoom + Cosine LR Stabilization):**
  1. **Random Horizontal Flip (`p = 0.5`):** Applies identical left-right reflection to `(image, mask)` pairs during training, removing left/right walking-direction bias without introducing padding artifacts.
  2. **Small Random Zoom-In (`scale ∈ [1.0, 1.15]`, `p = 0.5`):** Crops a `1 / s` central/jittered window from the raw image and resizes in a single step directly to `256 × 256` (bilinear for RGB, nearest-neighbor for mask). Restricting `scale ≥ 1.0` eliminates the black border padding and double-resampling blur present in Phase 1 (`[0.9, 1.1]` at `p = 1.0`).
  3. **Cosine Annealing LR (`CosineAnnealingLR`, `T_max = 50`, `eta_min = 1e-5`):** Smoothly decays the learning rate from `10^-3` to `10^-5` over `50` epochs, preventing late-epoch oscillation under stochastic augmentation (`lr_t = eta_min + 0.5 * (lr_0 - eta_min) * (1 + cos(pi * t / T_max))`).

## 2. Comparative Ablation Table (3-Seed Mean ± Sample SD, `seeds 42, 43, 44`)

| Variant | Val Precision | Val Recall | Val Dice / F1 | Val IoU | Test Precision | Test Recall | Test Dice / F1 | Test IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` (Baseline) | 0.8323 | 0.7849 | 0.8078 ± 0.0062 | 0.6776 ± 0.0087 | 0.8364 | 0.8632 | 0.8495 ± 0.0130 | 0.7385 ± 0.0197 |
| `zoom` only (`1.0–1.15`, `p = 0.5`) | 0.8339 | 0.8006 | 0.8168 ± 0.0061 | 0.6903 ± 0.0088 | 0.8270 | 0.8806 | 0.8529 ± 0.0054 | 0.7436 ± 0.0082 |
| `flip` only (`p = 0.5`) | 0.8189 | 0.8372 | 0.8277 ± 0.0027 | 0.7061 ± 0.0039 | 0.8215 | 0.9105 | 0.8629 ± 0.0181 | 0.7592 ± 0.0280 |
| **`both` (Proposed: Flip + Zoom)** | **0.8158** | **0.8448** | **0.8298 ± 0.0087** | **0.7091 ± 0.0127** | **0.8266** | **0.9113** | **0.8668 ± 0.0016** | **0.7649 ± 0.0025** |

### Ablation Takeaways
- **When both augmentations are present (`both` vs. `none`):** Validation overlap increases by **+2.20 pp Dice** / **+3.15 pp IoU** (`>3.6×` baseline seed SD); held-out test overlap increases by **+1.73 pp Dice** / **+2.64 pp IoU**, winning on **20/26** test images (`+2.55 pp` per-image IoU gain, two-sided sign-permutation `p = 0.00130`).
- **Removing zoom (`both → flip`):** Keeps most of the mean overlap gain (`+2.07 pp` test IoU over `none`, `19/26` test wins), confirming horizontal flip is the primary driver of higher foreground Recall (`0.8632 → 0.9105`). However, removing zoom sharply increases test seed variance (`IoU SD` grows from `±0.0025` to `±0.0280`).
- **Removing flip (`both → zoom`):** Retains variance tightening relative to baseline (`±0.0082` test IoU SD), with a smaller mean overlap gain (`+1.27 pp` val IoU, `+0.51 pp` test IoU over `none`).

## 3. Generated Figures & Test Mask Interpretation (S.No. 6)

- [report/figures/augmentation_ablation_bar.png](figures/augmentation_ablation_bar.png): Grouped Precision, Recall, Dice/F1, and IoU comparison (`Mean ± SD`) across Validation and Held-out Test splits.
- [report/figures/training_curves_none_vs_both.png](figures/training_curves_none_vs_both.png): 50-epoch train/validation loss trajectories (`none` vs. `both`) and 4-variant validation Dice curves.
- [report/figures/seed_variability_comparison.png](figures/seed_variability_comparison.png): Per-seed scatter + `Mean ± SD` plot highlighting run-to-run stabilization under `both` (`±0.0025` test IoU SD).
- [report/figures/per_image_test_comparison.png](figures/per_image_test_comparison.png): Paired per-image test IoU deltas across all `26` held-out images (`20/26` improved).
- [report/figures/test_baseline_vs_proposed_masks.png](figures/test_baseline_vs_proposed_masks.png): Side-by-side held-out test mask error comparison (`tuned_none_seed42` vs. `tuned_both_seed42`).
- [report/test_error_analysis/summary.md](test_error_analysis/summary.md): Full qualitative and quantitative interpretation of held-out test masks, including silhouette/limb recovery (`FudanPed00058.png`), unlabeled cyclist/edge-crop cases (`FudanPed00029.png`, `FudanPed00008.png`), and suitcase/pet false positives (`PennPed00094.png`).
