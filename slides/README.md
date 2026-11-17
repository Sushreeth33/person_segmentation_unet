# Project Presentation and Academic Poster

This directory contains the presentation and poster materials for **Binary Person Segmentation Using U-Net and Controlled Data Augmentation**.

---

## 1. Digital Presentation Links

- **Google Slides Presentation:**  
  [https://docs.google.com/presentation/d/1qUeX6-WzLKwkcdSYB7CSsgymCDjGfnK7/edit](https://docs.google.com/presentation/d/1qUeX6-WzLKwkcdSYB7CSsgymCDjGfnK7/edit?usp=sharing&ouid=111754227321804437115&rtpof=true&sd=true)  
  *Online presentation slide deck following the 15-slide guidelines structure (problem definition, literature gap, data audit, methodology, experimental results, and failure analysis).*

- **Canva Project Presentation & Poster:**  
  [https://canva.link/0vf0cpiw0xeu1yi](https://canva.link/0vf0cpiw0xeu1yi)  
  *Interactive cloud-hosted digital poster and visual presentation workspace.*

---

## 2. Academic Poster File

- **`poster.pptx` (and `../poster_final.pptx`):**  
  - **Canvas Dimensions:** 48 inches × 30 inches (standard academic conference widescreen ratio).
  - **Structure:**
    1. *Abstract & Introduction:* Dense pixel-wise vision formulation, small-data pedestrian challenge, core research question.
    2. *Problem Formulation & Objective:* RGB input $X \in \mathbb{R}^{3 \times 256 \times 256}$, ground truth binary mask $Y \in \{0, 1\}^{256 \times 256}$, joint BCE + soft Dice loss.
    3. *Dataset Integrity & Audit Hygiene Summary:* Penn-Fudan Pedestrian benchmark (170 photographic urban scenes, 345 pedestrian instances), zero-leakage 70/15/15 partitions sealed via SHA-256 manifest.
    4. *Methodology & End-to-End Architecture:* 5-Level skip-connected U-Net (7.76M parameters) with high-resolution system pipeline diagram.
    5. *Algorithm 1:* Controlled Augmentation & Segmentation Pipeline pseudo-code.
    6. *Key Contributions & Controlled Inductive Biases:* Bilateral horizontal reflection ($p=0.5$), single-step zoom-in crop ($1.0-1.15\times, p=0.5$), and training-only Copy-Paste synthesis.
    7. *Results & Multi-Seed Benchmark:* Official 3-seed validation and test performance table, ablation comparisons, and two-tailed paired t-test validation ($p = 0.00049$).
    8. *Qualitative Error Analysis & Insights:* Side-by-side silhouette comparisons (RGB, Ground Truth, Baseline, Proposed, Error overlay) demonstrating limb recovery and false-positive suppression.
    9. *Conclusion & References:* Key takeaways and literature citations.
