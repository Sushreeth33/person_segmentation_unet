# U-Net model, training loss, and evaluation metrics

This note describes the exact compact U-Net implementation and the loss/metrics used in the Penn-Fudan binary person-segmentation project. The model predicts one foreground class (person) against background. All positive instance IDs in each Penn-Fudan mask are merged into the person class.

## Model input and output

- Input: an RGB image resized to 256 × 256, with values scaled to [0, 1]. Tensor shape: `[batch, 3, 256, 256]`.
- Output: one raw logit per pixel. Tensor shape: `[batch, 1, 256, 256]`.
- A sigmoid maps a logit `z` to a person probability `p = sigmoid(z)` in [0, 1]. Evaluation marks a pixel as person when `p >= 0.5`.

## U-Net architecture

U-Net has a contracting encoder, a bottleneck, and an expanding decoder.

1. **Double-convolution block:** two 3 × 3 convolutions, each followed by batch normalization and ReLU. Padding preserves the block's spatial dimensions.
2. **Encoder:** four 2 × 2 max-pooling steps halve the spatial dimensions. For the trained model with 16 base channels, the feature channels progress 16 → 32 → 64 → 128, followed by a 256-channel bottleneck. At 256 × 256 input resolution, the bottleneck has spatial size 16 × 16.
3. **Decoder:** transposed convolutions double the spatial dimensions. At each level, the upsampled features are concatenated with features from the corresponding encoder level. These are the **skip connections**; they pass higher-resolution location and boundary information into the decoder.
4. **Output layer:** a 1 × 1 convolution maps the final 16 feature channels to one logit channel.

The trained baseline/flip models used 16 base channels, 30 epochs, batch size 4, Adam with learning rate 0.001, and CPU. The code's default base width is 32, but these experiments explicitly set it to 16 for CPU training.

## Training loss: BCE plus soft Dice

The loss is an equal-weight mixture:

\[
L_{\mathrm{total}} = 0.5L_{\mathrm{BCE}} + 0.5L_{\mathrm{Dice}}.
\]

### Binary cross-entropy

For target pixel `y` in {0, 1}, logit `z`, and probability `p = sigmoid(z)`, binary cross-entropy is

\[
L_{\mathrm{BCE}} = -\frac{1}{N}\sum_{j=1}^{N}
\left[y_j\log(p_j)+(1-y_j)\log(1-p_j)\right],
\]

where `N` is the number of pixels across the batch. The implementation uses PyTorch `BCEWithLogitsLoss`, which evaluates the same objective directly from logits using a numerically stable form. Targets are 1 for person and 0 for background.

### Soft Dice loss

During training, Dice uses probabilities rather than thresholded predictions so its gradient can guide the model. For each image `i` in a batch:

\[
D_i = \frac{2\sum_j p_{ij}y_{ij}+1}
{\sum_j p_{ij}+\sum_j y_{ij}+1}.
\]

The training Dice loss is

\[
L_{\mathrm{Dice}} = 1 - \frac{1}{B}\sum_{i=1}^{B}D_i,
\]

where `B` is the batch size. The `+1` numerator and denominator smoothing terms are present in the code to keep the expression defined and stable, including for an empty foreground. A perfect overlap gives Dice near 1 and Dice loss near 0.

The exact implementation sets `dice_weight=0.5`, computes sigmoid probabilities, sums over each image's channel and spatial dimensions, adds the smoothing constant 1, averages soft Dice across the batch, and combines it with BCE using equal weights.

## Evaluation: hard Dice, IoU, and other scores

Evaluation first thresholds each probability at 0.5 and compares the resulting binary prediction with the binary target. It counts:

- **TP (true positives):** person pixels correctly predicted as person.
- **FP (false positives):** background pixels predicted as person.
- **FN (false negatives):** person pixels predicted as background.
- **TN (true negatives):** background pixels correctly predicted as background.

The code accumulates these counts over every pixel in the evaluated split, then calculates:

\[
\mathrm{Dice/F1} = \frac{2TP}{2TP+FP+FN}
\qquad\qquad
\mathrm{IoU/Jaccard} = \frac{TP}{TP+FP+FN}.
\]

Both metrics focus on overlap of the person masks and do not use TN. For these same pooled counts,

\[
\mathrm{Dice} = \frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}.
\]

Other reported metrics are:

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},\quad
\mathrm{Specificity}=\frac{TN}{TN+FP},\quad
\mathrm{PixelAccuracy}=\frac{TP+TN}{TP+TN+FP+FN}.
\]

Because background occupies most pixels, pixel accuracy can be high even if some person pixels are missed. Dice and IoU are therefore the main overlap measures; precision and recall show whether errors are mostly extra foreground or missed people.

### Training Dice versus reported Dice

These are related but not identical calculations:

- The **Dice loss** uses soft probabilities, is computed per image, smoothed with `+1`, and averaged within each training batch.
- Reported **validation/test Dice and IoU** use hard 0.5-thresholded masks and pooled TP/FP/FN counts across the whole split.

Thus, the Dice component in the loss is not expected to equal the Dice score reported in a results table.

For example, the seed-42 horizontal-flip model's held-out test results were Dice 0.8573 and IoU 0.7503. Those satisfy the Dice/IoU relationship above up to rounding.

## Where this is implemented

- `src/person_segmentation/model.py`: `DoubleConv`, `UNet`, and `DiceBCELoss`.
- `src/person_segmentation/train.py`: `run_epoch`, training/evaluation mode, thresholding, pooled confusion counts, and metrics.
- `src/person_segmentation/data.py`: mask binarization, image normalization, and resizing.
- `report/baseline_results.md` and `report/augmentation_comparison.md`: experiment settings and measured scores.

## Research context

The U-Net architecture is based on the encoder-decoder design with skip connections from Ronneberger, Fischer, and Brox. Long et al.'s FCN work is foundational pixel-wise segmentation research and reports mean intersection-over-union. Chen et al.'s DeepLabv3+ was included in the project's architecture literature review. Shorten and Khoshgoftaar survey image data augmentation.

Dice-based segmentation objectives are discussed in work such as V-Net and Generalised Dice. The particular equal-weight BCE plus soft-Dice loss, its `+1` smoothing, and the hyperparameters above are the implementation used in this project; they should be described as our configured choices rather than claimed as an exact formula copied from one of those papers.

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI 2015.
2. Long, J., Shelhamer, E., & Darrell, T. (2015). [Fully Convolutional Networks for Semantic Segmentation](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html). CVPR 2015.
3. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabv3+)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.php). ECCV 2018.
4. Shorten, C., & Khoshgoftaar, T. M. (2019). [A Survey on Image Data Augmentation for Deep Learning](https://link.springer.com/article/10.1186/s40537-019-0197-0). Journal of Big Data, 6, 60.
5. Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797). 3DV 2016.
6. Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J. (2017). [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237). DLMIA 2017.
