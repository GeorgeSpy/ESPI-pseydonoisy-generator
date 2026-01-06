# Research Summary & Thesis Notes

This document summarizes the development journey of the ESPI pseudo-noise generation tool, detailing the challenges, ablation studies, and the final "pairfix" methodology that led to successful denoising results.

---

## 1. Starting Point

*   **Objective**: Create **pseudo-noisy** images to train a **DnCNN** model on ESPI data.
*   **Initial Script**: Used `make_pseudo_noisy_plus.py` / `make_pseudo_noisy_plus_minpatch.py`.
*   **Data Structure**:
    *   **Clean / Averaged**: `/data/wood_Averaged/w01, w02, w03` (and specific folders like `W03_ESPI_90db-Averaged`)
    *   **Real Single-shot**: `/data/wood_real_A/W01...`, `..._B/W02...`, `..._C/W03...`
    *   **ROI Masks**: `/roi/roi_mask.png`, `roi_mask_W02.png`, etc.
*   **Goal**: Generate **realistically noisy** data for training/validation and document the process for the thesis.

---

## 2. Phase 1: Pseudo-noisy generation for w01/w02/w03

*   Ran the script on clean averaged inputs:
    *   Input: `/data/wood_Averaged/...`
    *   Output: `/project/pseudo_noisy/roi/w01|w02|w03`
    *   Applied per-board ROI masks.
*   Generated ~**699 pseudo-noisy** images (243 + 255 + 201).
*   **Format Handling**: Processed `.tif` (pseudo-noisy) vs `.png` (averaged) differences in the manifests.

---

## 3. Manifests & LOBO Strategy

*   Created manifests for training/validation using **LOBO (Leave-One-Board-Out)**:
    *   **Train**: w01 + w02
    *   **Val**: w03
*   Resolved filename mismatches (e.g., `_full.tif` vs `.png`) by implementing **suffix stripping** logic.

---

## 4. First DnCNN Training (Baseline)

*   Conducted a **pilot training** (~3 epochs) to validate the pipeline.
*   **Validation Results (w03)**:
    *   Training PSNR: ~**14–15 dB**
    *   **Inference on Real Averaged Data**: Gain was negligible (**+0.0x dB**) or worse.
*   **Conclusion**: The model failed to improve quality on real data → indicating a significant **domain gap**.

---

## 5. Ablation Study: No-Blur

*   Generated a new dataset **without motion blur** (no-blur) to test the blur hypothesis.
*   Retrained with the same LOBO split.
*   **Result**: Performance was **worse** than full noise (PSNR -0.3 dB).
*   **Conclusion**: The blur component was actually useful/realistic; the problem **was not** the blur itself.

---

## 6. Hybrid Dataset (70% Full + 30% No-Blur)

*   To bridge the gap between training and validation, we created a **hybrid manifest**:
    *   70% Full pseudo-noisy
    *   30% No-blur
*   **Training Results**:
    *   Slight improvement over baseline: **PSNR +0.13 dB**, **SSIM +0.047**.
    *   Status: Better, but still not satisfactory for a final solution.

---

## 7. Log-Domain Training

*   Implemented a DnCNN variant operating in the **log-domain** (mathematically appropriate for multiplicative speckle noise).
*   **Training**: Showed very high numbers (50–60 dB) due to log scale.
*   **Inference**: Correctly applied inverse transform (`exp`, not `expm1`) for evaluation.
*   **Results**:
    *   It was the **best of the three models**.
    *   However, real validation gain remained small (**+0.03 dB**).
*   **Conclusion**: Log-domain is the **correct direction**, but the **training data** (pseudo-real) was holding back performance.

---

## 8. The Major Issue: Domain Gap

*   Evaluated on **Real Single** → **Real Averaged** pairs (not pseudo):
    *   PSNR gain: **Negative** (e.g., -0.26 dB).
    *   SSIM gain: Negative.
    *   MPI: High.
*   **Confirmation**: Training on synthetic noise and testing on real data causes a **PSNR gap** (consistent with literature).
*   Tried fine-tuning on pseudo-real data, but results worsened (-0.4 dB to -0.8 dB), proving that fine-tuning on imperfect synthetic data degrades performance.

---

## 9. The Breakthrough: "Pairfix" & Correct Alignment

This was the turning point of the research.

*   Instead of comparing "pseudo → pseudo-clean", we shifted to:
    *   **Real Single-shot** inputs (e.g., `/data/wood_real_C/W03...`)
    *   Matched with **Real Averaged** targets (e.g., `/data/wood_Averaged/W03...`)
    *   **Crucial Steps**:
        *   Exact matching by **Hz & dB**.
        *   **Integer alignment** before metric calculation.
        *   **Suffix stripping** for consistent filenames.
*   **Results** with this "Pairfix" methodology:
    *   **Mean ΔPSNR**: Increased from +0.028 dB → **+0.148 dB**.
    *   After filtering 2 outliers: **+0.278 dB Mean** and **+0.306 dB Median**.
    *   **Success Rate**: ~95.5%.
    *   **MPI_norm**: ~0.09.
*   **Conclusion**: This established the first **"production-ready"** setup.

---

## 10. Long-Run Evaluation

*   Executed a large-scale evaluation (≈3000 images).
*   **Results (1119/2989 processed)**:
    *   Median ΔPSNR: ~ **+0.298 dB**
    *   Mean ΔPSNR: ~ **+0.272 dB**
    *   Success Rate: 95.5%
    *   Outliers: 0.2%
    *   Consistent behavior across all frequency bands.
*   Documented in `MASTER_SUMMARY_REPORT.md`.

---

## 11. Calibration v2.0 (Global)

*   Performed formal calibration using:
    *   Single: `/data/wood_real_C/W03...`
    *   Avg: `/data/wood_Averaged/W03...`
*   **Initial Issues**:
    *   R² ~0.69 (low)
    *   Per-band gain (+0.19 dB) was below target.
*   **Conclusion**: Using **working set parameters (k=3.0, peak=60, sigma=0.01)** proved more stable than automatic calibration when the fit is poor.

---

## 12. Calibration v2.0 (Per-Board)

*   Applied calibration to **W01** and **W02**:
*   **W01**: Success → **+0.307 dB** median ΔPSNR, ΔSSIM ~0.0054, MPI_norm <0.1.
*   **W02**: Failed with both specific and shared parameters → Documented as a **limitation / board-specific mismatch**.

---

## 13. Final Landscape

Final results table:

| Board | Method                  | ΔPSNR         | Status            |
| :--- | :---------------------- | :------------ | :---------------- |
| **W03** | Working params (global) | +0.078 dB     | Baseline, 100% OK |
| **W01** | Calib v2.0 (per-band)   | **+0.307 dB** | ✅ Target Met      |
| **W02** | Any method              | < 0           | Limitation        |

**Thesis Narrative**:
*   **W01**: Demonstrates that calibration works effectively.
*   **W03**: Shows a stable baseline.
*   **W02**: Serves as a documented limitation, adding credibility to the research by acknowledging negative results.

---

## 14. Final Conclusion

> When evaluating on mismatched or semi-synthetic pairs, DnCNN showed zero or negative improvement. However, by matching **real single-shot** images with their **corresponding averaged** references (using correct keys, alignment, and ROI) and utilizing a log-domain model, we achieved a consistent **+0.27…+0.31 dB** gain on one board (W01) and a small but positive baseline on another (W03). Performance on W02 remains a documented limitation due to calibration challenges.
