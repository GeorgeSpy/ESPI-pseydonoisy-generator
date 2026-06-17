# Reproducibility Guide -- v3.2 Execution Layer

This document maps the **v3.2 reproducibility layer** (paper contribution #4,
Sections 3.2, 4.3, 5.5) to exact commands and frozen parameters, so that the
results reported in *"Calibrated Pseudo-Noisy Supervision for ESPI Denoising
with Severity-Controlled Evaluation"* (SETN 2026) can be regenerated from this
repository.

> Scope: this repository is the **pseudo-noisy generation component** only.
> The fixed V4 DnCNN-Lite-ECA probe, its checkpoints, and all probe-side
> numbers (Tables 1, 3, 4, 5) belong to the separate denoising repository
> `ESPI-DnCNN-ECA`. Phase 5 below is included for completeness but requires
> that external checkpoint.

---

## 1. Requirements

```
pip install -r requirements.txt        # numpy, Pillow (required); scipy/opencv optional
# Phase 5 additionally requires: torch
```

## 2. Files

| File | Role |
|------|------|
| `make_pseudo_noisy_plus_v3_2.py` | Frozen v3.2 generator (manifest, per-image RNG, calibration modes) |
| `make_pseudo_noisy_plus_v3_1.py` | Frozen v3.1 baseline (kept for backward-compatibility checks) |
| `phase1_backward_compat_validation.py` | v3.2 reproduces v3.1 outputs exactly (legacy/global path) |
| `phase2_order_invariance_validation.py` | Per-image RNG -> output independent of file order |
| `phase3_replayability_validation.py` | A run is fully replayable from its provenance manifest |
| `phase4_calibration_validation.py` | global / by-regime / by-image calibration modes |
| `phase5_utility_validation.py` | Probe-side utility (needs external V4 checkpoint + torch) |

`make_pseudo_noisy_plus_v3_1.py` is a frozen compatibility reference only. It
does not define a separate published calibration block; the calibration values
below are the authoritative parameter sets for the paper-level v3.2 layer.

## 3. Path configuration (portable)

The validators no longer use hard-coded `C:\` / `D:\` paths. They resolve all
locations from environment variables, with repo-relative defaults:

| Variable | Meaning | Default |
|----------|---------|---------|
| `ESPI_REPO` | folder holding the generator scripts | the validator's own folder |
| `ESPI_DATA` | dataset root (contains `wood_Averaged/`, `wood_real_A/`) | `<repo>/data` |
| `ESPI_OUT`  | output root for validation artifacts | `<repo>/_validation_out` |
| `ESPI_V4_CKPT` | frozen V4 DnCNN-Lite-ECA checkpoint (phase 5 only) | `<repo>/checkpoints/v4_canonical_best.pth` |

Expected dataset layout under `ESPI_DATA`:

```
<ESPI_DATA>/
  wood_Averaged/W01_ESPI_90db-Averaged/   # averaged (clean) references
  wood_real_A/W01_ESPI_90db/              # real single-shot images (for calibration)
```

Example (PowerShell):

```powershell
$env:ESPI_REPO = "C:\path\to\repo"
$env:ESPI_DATA = "C:\path\to\data"
$env:ESPI_OUT  = "C:\path\to\validation_out"
```

Example (bash):

```bash
export ESPI_REPO=/path/to/repo
export ESPI_DATA=/path/to/data
export ESPI_OUT=/path/to/validation_out
```

---

## 4. Frozen calibration values

These are the parameter blocks reported in the paper. `(k, p, sigma)` =
(speckle shape, Poisson peak, Gaussian sigma).

| Setting | k | p | sigma | Paper ref |
|---------|------|--------|--------|-----------|
| **Canonical Mid (frozen baseline)** | 1.89 | 12.5 | 0.0919 | Section 3.2 |
| Calibration mode: global | 2.111 | 24.765 | 0.086 | Section 3.2 |
| Calibration mode: by-regime -- Mid | 2.015 | 15.221 | 0.084 | Section 3.2 / Section 5.5 |
| Calibration mode: by-regime -- High | 2.224 | 27.184 | 0.087 | Section 3.2 / Section 5.5 |
| Wood reference (carbon study) | 2.1112 | 24.7652 | 0.0861 | Section 5.4 |
| Carbon domain | 1.9908 | 16.0772 | 0.0720 | Section 5.4 |

The canonical Mid block is the **frozen parameterization** used throughout the
main study (post-noise matching: `--match meanstd`).

---

## 5. Generate the canonical Mid pseudo-noisy set

```bash
python make_pseudo_noisy_plus_v3_2.py \
  --input  "$ESPI_DATA/wood_Averaged/W01_ESPI_90db-Averaged" \
  --output "$ESPI_OUT/mid_canonical" \
  --seed 123 \
  --speckle-k 1.89 --poisson-peak 12.5 --gauss-sigma 0.0919 \
  --match meanstd \
  --rng-mode per_image \
  --write-manifest "$ESPI_OUT/mid_canonical/run_manifest.json" \
  --write-per-image-params "$ESPI_OUT/mid_canonical/per_image_params.csv"
```

`--rng-mode per_image` makes each image's noise depend only on its own
identity (deterministic, order-independent). Use `--rng-mode legacy` to
reproduce the exact v3.1 byte stream.

## 6. Reproduce the calibration modes (Section 3.2)

```bash
# global single block
python make_pseudo_noisy_plus_v3_2.py \
  --input  "$ESPI_DATA/wood_Averaged/W01_ESPI_90db-Averaged" \
  --output "$ESPI_OUT/calib_global" \
  --calib-single "$ESPI_DATA/wood_real_A/W01_ESPI_90db" \
  --calib-avg    "$ESPI_DATA/wood_Averaged/W01_ESPI_90db-Averaged" \
  --calibration-mode global --calib-override \
  --calibration-summary-path "$ESPI_OUT/calib_global/calibration_summary.json"

# by-regime (distinct Mid / High blocks)
python make_pseudo_noisy_plus_v3_2.py \
  --input  "$ESPI_DATA/wood_Averaged/W01_ESPI_90db-Averaged" \
  --output "$ESPI_OUT/calib_by_regime" \
  --calib-single "$ESPI_DATA/wood_real_A/W01_ESPI_90db" \
  --calib-avg    "$ESPI_DATA/wood_Averaged/W01_ESPI_90db-Averaged" \
  --calibration-mode by-regime --calib-override \
  --calibration-summary-path "$ESPI_OUT/calib_by_regime/calibration_summary.json"
```

---

## 7. Run the reproducibility validators

The validators are **chained** -- run them in order. Phase 2/3/4 read Phase 1's
output root; Phase 5 reads Phase 4's.

```bash
python phase1_backward_compat_validation.py     # v3.2 == v3.1 (legacy path)
python phase2_order_invariance_validation.py    # order invariance (per_image RNG)
python phase3_replayability_validation.py       # replay from provenance manifest
python phase4_calibration_validation.py         # global / by-regime / by-image
python phase5_utility_validation.py             # needs ESPI_V4_CKPT + torch
```

### Expected verdicts (Phases 1-3 confirmed on the W01 wood set)

| Phase | pairs | byte-identical | max deviation | Verdict |
|-------|-------|----------------|---------------|---------|
| 1 backward-compat | 30 | 30 | 0.0 | v3.2 reproduces v3.1 exactly (global mode) |
| 2 order invariance | 30 | 30 | 0.0 | v3.2 order-invariant under per-image RNG |
| 3 replayability | 30 | 30 | 0.0 | runs fully replayable from provenance |

This matches the paper's Section 5.5 statement: *"all confirmed by 30/30
byte-identical outputs with maximum deviation 0.0."*

---

## 8. Notes on the lost `D:` drive

All artifacts that previously lived under `D:\Pseudonoisy Test\...` were
**derived** outputs of the validators and are regenerated by the commands
above. The only non-regenerable item was the trained V4 probe checkpoint
(`ESPI_V4_CKPT`), which belongs to the `ESPI-DnCNN-ECA` repository, not to this
generator repo. Nothing in Sections 5-7 of this guide depends on `D:`.
