# Validation prediction review

## Setup and aggregate scores

- Split: validation, 25 images; test images were not included.
- Checkpoint: seed-42 U-Net with horizontal-flip training augmentation.
- Input and threshold: 256 × 256, probability threshold 0.5.
- Pixel-aggregated Dice: 0.8278; IoU: 0.7062; pixel accuracy: 0.9387.
- Mean of per-image Dice: 0.8331; mean per-image IoU: 0.7225.

## Observed error patterns

The overlays show two main kinds of difficult cases:

1. **Extra foreground on cluttered backgrounds.** FudanPed00040 has the lowest Dice (0.612); it has 5,074 false-positive pixels and 1,813 false-negative pixels. The prediction spills beyond the two people into parts of the doorway/glass area. FudanPed00041 also has many false-positive pixels (4,357) in a busy scene.
2. **Missed person pixels.** PennPed00014 has Dice 0.681, precision 0.873, and recall 0.558 (1,180 false positives versus 6,412 false negatives). PennPed00013 has both missed person area and background spill. These examples suggest the model can miss parts of people even when it finds other people in the image.
3. **Thin boundaries and limbs are less stable.** Error maps show small misses or spill around lower legs and silhouette edges, consistent with losing detail when the original images are resized to 256 × 256.

Easy images are segmented much more cleanly, so the aggregate score hides a meaningful range of per-image quality. The error patterns point toward reviewing resize/aspect-ratio handling and context in crowded scenes before adding more augmentation. The overlays do not establish whether every apparent false positive is truly an error; crowded examples should be checked against the original instance annotations.

## Files

- `validation_overlays_sorted_by_dice.png`: all 25 validation samples, ordered from lowest to highest Dice. Error legend: green = true positive, red = false positive, blue = false negative.
- `overlays/`: full-resolution four-panel overlay cards for each image.
- `per_image_metrics.csv`: per-image Dice, IoU, precision, recall, and pixel counts.
- `summary.json`: aggregate metrics and lists of the hardest cases.
- Generator: `src/person_segmentation/visualize_predictions.py`.
