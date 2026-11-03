# Penn-Fudan dataset audit

Source: [Penn-Fudan Pedestrian dataset](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)

## Integrity

- Images: 170; masks: 170; matched pairs: 170.
- Missing masks: 0; orphan masks: 0.
- Image/mask dimension mismatches: 0.
- Original masks use pixel value 0 for background and positive integer values for separate person instances. Observed values are [0, 1, 2, 3, 4, 5, 6, 7, 8]; person instance IDs are [1, 2, 3, 4, 5, 6, 7, 8].
- Merging all positive IDs and resizing with nearest-neighbor interpolation produced binary values [0, 1].

## Image sizes

There are 170 distinct original image sizes. Widths range from 253 to 1017 pixels; heights range from 311 to 581 pixels. All image/mask pairs have matching dimensions. The table shows the first ten width × height pairs; the complete per-image data is in `per_image.csv`.

| Width × height | Images |
|---|---:|
| 1017x444 | 1 |
| 253x323 | 1 |
| 266x342 | 1 |
| 292x349 | 1 |
| 294x331 | 1 |
| 302x363 | 1 |
| 309x351 | 1 |
| 314x320 | 1 |
| 317x345 | 1 |
| 324x318 | 1 |

## Dataset statistics

- RGB channel means, scaled to [0, 1]: [0.439707, 0.444488, 0.406243].
- RGB channel standard deviations, scaled to [0, 1]: [0.263534, 0.250668, 0.257679].
- Verified normalized RGB values are finite and range from 0.0000 to 1.0000.
- Mean per-image foreground fraction before resize: 0.1728 ± 0.0766.
- Images were resized to 256 × 256; RGB used bilinear interpolation and masks used nearest-neighbor interpolation.

## Person instances per image

| Person instances | Images |
|---:|---:|
| 1 | 58 |
| 2 | 40 |
| 3 | 35 |
| 4 | 17 |
| 5 | 12 |
| 6 | 5 |
| 7 | 2 |
| 8 | 1 |

## Reproducible split

Seed 42; seeded random shuffle; 70/15/15 split. Counts: {'train': 119, 'validation': 25, 'test': 26}. The filename manifest is stored in `split_seed42.json`.

See `sample_images_masks.png` for original images, color-coded instance IDs, and merged binary masks. See `instances_per_image.png` for the instance-count distribution. `per_image.csv` and `summary.json` contain the full audit results.
