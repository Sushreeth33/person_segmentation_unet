# Archive: Preliminary Augmentation Exploration & Early CPU Experiments

> **Note:** This internal technical archive preserves historical exploratory benchmarks and preliminary experimental data for repository completeness. Formal report deliverables focus on the finalized, standardized single-step scale jitter (small zoom 1.0–1.15×), horizontal flip, and instance Copy-Paste pipeline.

---

## 1. Early 30-Epoch CPU Exploratory Benchmark

In initial development (30 epochs, constant learning rate $\text{lr} = 0.001$, Adam optimizer, CPU, batch size 4), exploratory runs tested bidirectional scaling ($0.9\times - 1.1\times$ at $p = 1.0$) alongside horizontal flip on the fixed 119/25/26 Penn-Fudan split (`split_seed42.json`):

| Setup / Mode | Seed | Epoch | Val Precision | Val Recall | Val Dice | Val IoU | Test Dice | Test IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`none`)** | 42 | 22 | 0.8086 | 0.8177 | 0.8131 | 0.6851 | 0.8471 | 0.7348 |
| **Baseline (`none`)** | 43 | 25 | 0.8134 | 0.7926 | 0.8050 | 0.6737 | 0.8352 | 0.7180 |
| **Baseline (`none`)** | 44 | 24 | 0.8210 | 0.7940 | 0.8071 | 0.6765 | 0.8380 | 0.7215 |
| *Mean (Baseline)* | — | — | 0.8143 | 0.8014 | 0.8084 | 0.6784 | 0.8401 | 0.7248 |
| **Horizontal Flip (`flip`)** | 42 | 29 | 0.8312 | 0.8245 | 0.8278 | 0.7062 | 0.8573 | 0.7503 |
| **Horizontal Flip (`flip`)** | 43 | 27 | 0.8190 | 0.8080 | 0.8135 | 0.6856 | 0.8490 | 0.7380 |
| **Horizontal Flip (`flip`)** | 44 | 28 | 0.8250 | 0.8445 | 0.8346 | 0.7162 | 0.8512 | 0.7410 |
| *Mean (Flip)* | — | — | 0.8251 | 0.8257 | 0.8253 | 0.7027 | 0.8525 | 0.7431 |
| **Bidirectional Zoom ($0.9\times - 1.1\times$, $p=1.0$)** | 42 | 29 | 0.8180 | 0.8030 | 0.8104 | 0.6812 | 0.8563 | 0.7487 |
| **Bidirectional Zoom ($0.9\times - 1.1\times$, $p=1.0$)** | 43 | 20 | 0.8095 | 0.8028 | 0.8061 | 0.6752 | 0.8296 | 0.7087 |
| **Bidirectional Zoom ($0.9\times - 1.1\times$, $p=1.0$)** | 44 | 29 | 0.8190 | 0.8058 | 0.8123 | 0.6839 | 0.8440 | 0.7300 |
| *Mean (Bidirectional)* | — | — | 0.8155 | 0.8039 | 0.8096 | 0.6801 | 0.8433 | 0.7292 |

---

## 2. Technical Findings and Evolution to Final Pipeline

1. **Scale Range Clarification:**
   Scaling factors below $1.0\times$ naturally shrink the foreground image area and require artificial zero-padding (black borders), which introduces step discontinuities at image borders. Clarifying the problem statement specification to **small zoom-in** ($1.0\times - 1.15\times$) ensures that crops are sampled cleanly from authentic image pixels.
2. **Single-Step Interpolation:**
   Performing cropping and resizing directly from raw resolution data avoids double bilinear resampling artifacts that occur when transforming already-downsampled intermediate copies.
3. **Cosine Annealing Optimization:**
   Extending training to 50 epochs with `CosineAnnealingLR` ($\eta_{\min} = 10^{-5}$) on GPU provided full gradient convergence and stabilized seed-to-seed variance, serving as the master standard for all project report findings.
