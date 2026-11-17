# Trained Model Artifacts

This directory houses the primary trained U-Net checkpoint for easy single-command evaluation and inference.

## Primary Model Checkpoint
- **`unet_proposed_both_seed42.pt`**:
  - **Architecture:** 5-Level U-Net with skip connections (7.76M parameters).
  - **Input Size:** 256 × 256 × 3 RGB.
  - **Trained With:** Proposed Controlled Augmentations (Horizontal Flip $p=0.5$ + Bounded Zoom-in $[1.0, 1.15\times]$ $p=0.5$).
  - **Validation Dice / IoU:** 82.98% / 70.91%
  - **Held-Out Test Dice / IoU:** 86.68% / 76.49% (vs. Baseline 73.85% IoU; $+2.64$ pp gain, $8\times$ variance reduction).

## Reproduction Command
To evaluate this checkpoint on the untouched test split:
```bash
python -m person_segmentation.evaluate --data data/PennFudanPed_256 --checkpoint models/unet_proposed_both_seed42.pt --split test
```

To run inference on a single test photograph:
```bash
python -m person_segmentation.predict --image data/PennFudanPed/PNGImages/PennPed00034.png --checkpoint models/unet_proposed_both_seed42.pt --output results/test_inference.png
```

*Note: All 18 complete experimental run checkpoints (across seeds 42, 43, 44 and all ablation variants) are systematically archived under `results/`.*
