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

The combined augmentation's validation Dice is 0.27 percentage points below the baseline and 1.74 points below flip-only in this initial 30-epoch CPU setup. On the test split it is 0.91 points above the baseline and 0.10 points below flip-only in Dice. It is 1.39 points above baseline and 0.15 points below flip-only in IoU. The difference in validation and test ordering is a reminder that the test set is small; do not tune based on these test scores. The mean-image Dice and pooled Dice are both shown because the former weights each image equally, while the latter pools pixel counts.

The initial 30-epoch CPU baseline-versus-proposed comparison showed that adding bidirectional zoom (0.9–1.1 at 100% probability) with a constant 0.001 learning rate did not improve validation Dice over no augmentation; see the tuned GPU section below for the diagnosed fixes and updated full ablation. The problem statement's comparison reports baseline (`none`) and flip-plus-zoom (`both`) as the main pair, with `flip` and `zoom` as single-augmentation ablations.

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

Zoom-only (`0.9–1.1` on CPU) was left unrun in Phase 1; instead, after diagnosing the causes of Phase 1 under-convergence, all four variants (`none`, `zoom`, `flip`, `both`) were trained cleanly across three seeds under the tuned GPU pipeline below.

## Tuned GPU pipeline: full 4-way comparison and ablation (50 epochs, cosine LR, `1.0–1.15` zoom at `p = 0.5`)

### Diagnosis of Phase 1 limitations and targeted fixes

Reviewing Phase 1 histories and zoom behavior identified three training bottlenecks that masked the benefit of `flip + small zoom`:

1. **Zoom-out black padding (`0.9–1.1` at `p = 1.0`):** Zooming out below `1.0` shrank images ~50% of the time and padded borders with solid black `(0, 0, 0)`, reducing foreground recall. Replacing this with zoom-in only (`SMALL_ZOOM_SCALE_RANGE = (1.0, 1.15)`) applied at `ZOOM_PROBABILITY = 0.5` preserves person framing and mixes zoomed and unzoomed views.
2. **Double bilinear resampling:** Applying zoom to pre-resized `256 × 256` copies (`data/PennFudanPed_256/`) blurred edges twice. Loading from raw `data/PennFudanPed/` and resizing directly to the zoomed crop when `resize_mode == "stretch"` performs a single resampling step (`--cache` keeps decoded RGB/mask pairs in RAM).
3. **Under-convergence and late-epoch LR oscillation:** With batch size `4` and constant `lr = 0.001` over `30` epochs, augmented runs had not stabilized cleanly. Switching to `--epochs 50` with `--lr-scheduler cosine` (`CosineAnnealingLR`, `eta_min = 1e-5`) on the RTX 4060 Laptop GPU (`device = cuda`) resolved late-epoch instability.

### Tuned validation and one-time test results across seeds (`42`, `43`, `44`)

All 12 runs share the fixed `119/25/26` `split_seed42.json` manifest, compact 16-channel U-Net, equal-weight `DiceBCELoss`, Adam (`lr = 0.001`), cosine schedule (`50` epochs, `eta_min = 1e-5`), batch size `4`, raw-source `256 × 256` stretch resizing, and RTX 4060 GPU training. Checkpoints were selected strictly by validation Dice; held-out test metrics (`26` images) were evaluated once per checkpoint afterward.

| Variant | Seed | Best epoch | Val Precision | Val Recall | Val Dice | Val IoU | Test Precision | Test Recall | Test Dice | Test IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` (baseline) | 42 | 28 | 0.8390 | 0.7752 | 0.8058 | 0.6748 | 0.8572 | 0.8693 | 0.8632 | 0.7593 |
| `none` (baseline) | 43 | 29 | 0.8134 | 0.7926 | 0.8029 | 0.6707 | 0.8223 | 0.8754 | 0.8480 | 0.7361 |
| `none` (baseline) | 44 | 43 | 0.8445 | 0.7870 | 0.8147 | 0.6874 | 0.8297 | 0.8450 | 0.8373 | 0.7201 |
| `zoom` (ablation) | 42 | 28 | 0.8348 | 0.8131 | 0.8238 | 0.7004 | 0.8388 | 0.8798 | 0.8588 | 0.7525 |
| `zoom` (ablation) | 43 | 37 | 0.8441 | 0.7854 | 0.8137 | 0.6859 | 0.8247 | 0.8804 | 0.8517 | 0.7417 |
| `zoom` (ablation) | 44 | 43 | 0.8226 | 0.8032 | 0.8128 | 0.6846 | 0.8173 | 0.8816 | 0.8483 | 0.7365 |
| `flip` (ablation) | 42 | 40 | 0.8235 | 0.8308 | 0.8271 | 0.7052 | 0.8520 | 0.9091 | 0.8796 | 0.7851 |
| `flip` (ablation) | 43 | 42 | 0.8286 | 0.8223 | 0.8254 | 0.7027 | 0.8417 | 0.8909 | 0.8656 | 0.7630 |
| `flip` (ablation) | 44 | 32 | 0.8047 | 0.8584 | 0.8307 | 0.7104 | 0.7709 | 0.9315 | 0.8436 | 0.7295 |
| `both` (proposed) | 42 | 37 | 0.8048 | 0.8769 | 0.8393 | 0.7231 | 0.8177 | 0.9240 | 0.8676 | 0.7662 |
| `both` (proposed) | 43 | 42 | 0.8220 | 0.8225 | 0.8223 | 0.6982 | 0.8369 | 0.8949 | 0.8649 | 0.7620 |
| `both` (proposed) | 44 | 43 | 0.8205 | 0.8351 | 0.8277 | 0.7061 | 0.8252 | 0.9151 | 0.8678 | 0.7665 |

### Three-seed summary (`Mean ± sample SD`)

| Variant | Val Precision | Val Recall | Val Dice (`Mean ± SD`) | Val IoU (`Mean ± SD`) | Test Precision | Test Recall | Test Dice (`Mean ± SD`) | Test IoU (`Mean ± SD`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` (baseline) | 0.8323 | 0.7849 | 0.8078 ± 0.0062 | 0.6776 ± 0.0087 | 0.8364 | 0.8632 | 0.8495 ± 0.0130 | 0.7385 ± 0.0197 |
| `zoom` only | 0.8339 | 0.8006 | 0.8168 ± 0.0061 | 0.6903 ± 0.0088 | 0.8270 | 0.8806 | 0.8529 ± 0.0054 | 0.7436 ± 0.0082 |
| `flip` only | 0.8189 | 0.8372 | 0.8277 ± 0.0027 | 0.7061 ± 0.0039 | 0.8215 | 0.9105 | 0.8629 ± 0.0181 | 0.7592 ± 0.0280 |
| **`both` (proposed)** | **0.8158** | **0.8448** | **0.8298 ± 0.0087** | **0.7091 ± 0.0127** | **0.8266** | **0.9113** | **0.8668 ± 0.0016** | **0.7649 ± 0.0025** |

### Statistical check and ablation interpretation

- **Baseline (`none`) vs. Proposed (`both`):** Validation overlap rises by **+0.0220 Dice (+2.20 pp)** and **+0.0315 IoU (+3.15 pp)** — over `3.6×` the baseline seed SD (`±0.0087` IoU). Held-out test overlap rises by **+0.0173 Dice (+1.73 pp)** and **+0.0264 IoU (+2.64 pp)**, driven by a `+4.81 pp` increase in test Recall (`0.8632 → 0.9113`) with a small `-0.98 pp` Precision shift (`0.8364 → 0.8266`). Every `both` seed (`0.8649–0.8678` test Dice) beats every `none` seed (`0.8373–0.8632` test Dice).
- **Per-image paired test check (`n = 26` test images, 3-seed mean per image):** Comparing `none` vs. `both`, the proposed model improves on **20/26** images (`+0.0164` Dice, `SD = 0.0257`, `t = 3.26`, two-sided sign-permutation `p = 0.00256`; `+0.0255` IoU, `SD = 0.0370`, `t = 3.51`, `p = 0.00130`).
- **Contribution of each augmentation (ablation):** Horizontal flip accounts for the majority of the mean gain (`none → flip`: `+0.0199` val Dice / `+0.0285` val IoU; `+0.0134` test Dice / `+0.0207` test IoU; `19/26` test images improved, IoU `p = 0.01956`). Adding small zoom (`1.0–1.15×` at `p = 0.5`) yields a smaller incremental mean gain (`flip → both`: `+0.0021` val Dice / `+0.0030` val IoU; `+0.0039` test Dice / `+0.0057` test IoU; `14/26` test images improved, IoU `p = 0.77602`) while substantially reducing run-to-run variance on test (`±0.0280` `flip` IoU SD → `±0.0025` `both` IoU SD).
- Further experiments such as Copy-Paste were evaluated in Phase 3 below as an optional extension.

## Phase 3: Copy-Paste person-instance augmentation (optional extension)

Using raw per-person instance masks strictly from the `119` training images (`290` pedestrian cutouts with `area >= 150 px`, `h >= 20`, `w >= 8`; never touching validation or test), `src/person_segmentation/copypaste.py` and `src/person_segmentation/train_copypaste.py` apply Copy-Paste at `p = 0.5` (`1–2` donor instances per image, horizontal flip `p = 0.5`, scale jitter `0.85–1.15×`, grounded vertical placement `±28 px`, `max_overlap <= 0.45`, and `0.8 px` soft RGB boundary feathering with crisp binary mask union). Both standalone `copypaste` and `both + copypaste` (`Flip + Small Zoom + Copy-Paste`) were trained under the exact same 50-epoch cosine-LR GPU setup across seeds `42, 43, 44`:

| Variant | Seed | Val Precision | Val Recall | Val Dice | Val IoU | Test Precision | Test Recall | Test Dice | Test IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `copypaste` only | 42 | 0.8300 | 0.8958 | 0.8616 | 0.7569 | 0.8101 | 0.9318 | 0.8667 | 0.7648 |
| `copypaste` only | 43 | 0.8264 | 0.8682 | 0.8468 | 0.7343 | 0.8125 | 0.9124 | 0.8595 | 0.7537 |
| `copypaste` only | 44 | 0.8348 | 0.8433 | 0.8390 | 0.7227 | 0.8321 | 0.8947 | 0.8623 | 0.7579 |
| **`copypaste` Mean ± SD** | — | 0.8304 | 0.8691 | **0.8491 ± 0.0115** | **0.7379 ± 0.0174** | 0.8182 | 0.9130 | **0.8628 ± 0.0036** | **0.7588 ± 0.0056** |
| `both + copypaste` | 42 | 0.8281 | 0.8953 | 0.8604 | 0.7551 | 0.8372 | 0.9215 | 0.8773 | 0.7815 |
| `both + copypaste` | 43 | 0.8240 | 0.8614 | 0.8422 | 0.7275 | 0.8260 | 0.9154 | 0.8684 | 0.7675 |
| `both + copypaste` | 44 | 0.8510 | 0.8522 | 0.8516 | 0.7416 | 0.8434 | 0.9028 | 0.8721 | 0.7731 |
| **`both + copypaste` Mean ± SD** | — | **0.8344** | **0.8696** | **0.8514 ± 0.0091** | **0.7414 ± 0.0138** | **0.8355** | **0.9132** | **0.8726 ± 0.0045** | **0.7740 ± 0.0070** |

### Copy-Paste takeaways
- **Over Baseline (`none → both + copypaste`):** Validation overlap rises by **+4.36 pp Dice / +6.38 pp IoU** (`0.8078/0.6776 → 0.8514/0.7414`); held-out test overlap rises by **+2.31 pp Dice / +3.55 pp IoU** (`0.8495/0.7385 → 0.8726/0.7740`), improving **22/26** test images (`+0.0361` per-image IoU, `t = 3.76`, `p = 0.00049`; `+0.0233` Dice, `t = 3.38`, `p = 0.00137`).
- **Over Flip + Small Zoom (`both → both + copypaste`):** Validation overlap rises by **+2.16 pp Dice / +3.23 pp IoU** (`0.8298/0.7091 → 0.8514/0.7414`), recovering Precision (`0.8158 → 0.8344`) alongside Recall (`0.8448 → 0.8696`). Test overlap rises by **+0.58 pp Dice / +0.91 pp IoU** (`15/26` test images improved, `p = 0.152`).

## Reproducibility files

- **Phase 1 (30-epoch CPU) artifacts:**
  - Flip model checkpoint & metrics: `results/augmentation_flip/` (copied to `results/all_project_results/models/augmentation_flip/`).
  - Proposed `0.9–1.1` flip-plus-small-zoom runs: `results/augmentation_flip_small_zoom_seed{42,43,44}/` (copied to `results/all_project_results/models/`).
  - Repeat-seed `none`/`flip` runs: `results/augmentation_repeats/seed{42,43,44}_{none,flip}/` (copied to `results/all_project_results/models/`).
- **Phase 2 (50-epoch GPU, cosine LR, `1.0–1.15` zoom at `p = 0.5`) artifacts:**
  - All 12 core runs (`none`, `zoom`, `flip`, `both` × seeds `42, 43, 44`): `results/tuned_{none,zoom,flip,both}_seed{42,43,44}/` (copied to `results/all_project_results/models/`).
  - Phase 3 Copy-Paste runs (`copypaste`, `both_copypaste` × seeds `42, 43, 44`): `results/tuned_{copypaste,both_copypaste}_seed{42,43,44}/` (copied to `results/all_project_results/models/`).
  - Training and dataset code: `src/person_segmentation/data.py`, `src/person_segmentation/train.py`, `src/person_segmentation/copypaste.py`, and `src/person_segmentation/train_copypaste.py`.
