# ESPI-PseudoNoisy: Physically Calibrated Pseudo-Noisy Generation for ESPI

This repository contains a physically calibrated pseudo-noisy generator for ESPI denoising research under limited real paired supervision. Its role is to generate controlled synthetic supervision that is closer to the real single-shot acquisition regime, supporting synthetic-real gap analysis and severity-controlled denoising experiments.

The repository should be understood as a **research software component for calibrated ESPI synthetic supervision**, not as a standalone end-to-end solution for denoising or classification.

## Research positioning

Within the broader ESPI research workflow, this repository supports denoising studies by providing a calibrated synthetic-corruption model when matched real noisy/clean training pairs are limited or incomplete.

Its purpose is to:

- generate pseudo-noisy ESPI samples from cleaner reference images,
- support denoising model development under limited real supervision,
- study how synthetic supervision behaves relative to real-aligned supervision,
- document the calibration and development history behind the pseudo-noisy generation process.

It should **not** be interpreted as a general-purpose augmentation toolkit, and it should **not** be read as implying that this repository alone establishes downstream denoising or classification performance.

## Core modeling idea

The generator is built around a physically motivated cascade designed to approximate ESPI acquisition noise:

1. **Multiplicative speckle (Gamma)** as the primary coherent-noise component
2. **Poisson shot noise** to reflect photon-counting effects
3. **Gaussian floor noise** to approximate electronic noise and quantization

On top of this baseline cascade, the repository includes calibration and realism-oriented mechanisms such as:

- calibration from **real single-shot vs averaged reference pairs**,
- specimen-aware and material-aware parameter defaults,
- frequency- and amplitude-dependent motion blur,
- optional spatial variation and temporal correlation,
- matched pair generation for denoising experiments.

## Main scripts

- `make_pseudo_noisy_plus.py`
  Main calibrated generator with the full research-oriented noise cascade and calibration options.

- `make_pseudo_noisy_matched.py`
  Utility for generating matched clean/noisy pairs for denoising workflows.

- `generate_pseudo_noisy.py`
  Lightweight batch-style wrapper for pseudo-noisy generation.

- `make_pseudo_noisy_v3.py`
  Earlier generator variant retained for historical development context.

## Quick start

### Installation

```bash
pip install -r requirements.txt
```

### Example usage

```bash
python make_pseudo_noisy_plus.py \
    --input /path/to/clean/images \
    --output /path/to/output \
    --material wood \
    --freq-hz 180 \
    --amp-db 90.0 \
    --seed 42
```

## Research interpretation

The scientific interpretation is **regime-dependent**, not generator-only:

- physically calibrated pseudo-noisy supervision is useful when real denoising supervision is scarce,
- reducing the synthetic-real gap matters more than simply increasing synthetic quantity,
- downstream benefit depends on how closely the generated supervision matches the real acquisition regime,
- denoising and downstream classification conclusions must be interpreted together with the separate denoising and classification repositories.

## Repository contents

This repository currently contains the public generator scripts and research-supporting notes in the repository root:

- `README.md`
- `RESEARCH_SUMMARY.md`
- `DEVELOPMENT_LOG.md`
- `make_pseudo_noisy_plus.py`
- `make_pseudo_noisy_matched.py`
- `generate_pseudo_noisy.py`
- `make_pseudo_noisy_v3.py`
- `requirements.txt`
- `CITATION.cff`
- `examples/`

## Related repositories

The broader ESPI research codebase is split across three public code components:

- **Pseudo-noisy generation (this repository)** (`https://github.com/GeorgeSpy/ESPI-pseydonoisy-generator`)
- **DnCNN-ECA denoising** (`https://github.com/GeorgeSpy/ESPI-DnCNN-ECA`)
- **Classification and evaluation** (`https://github.com/GeorgeSpy/espi-classification-models_2`)

## Citation

If you use this repository, please cite the software metadata in `CITATION.cff`. A repository-level BibTeX example is:

```bibtex
@software{spyridakis2025espi_pseudonoisy,
  title   = {ESPI-PseudoNoisy: Physically Calibrated Pseudo-Noisy Generation for ESPI},
  author  = {Spyridakis, Georgios},
  year    = {2025},
  url     = {https://github.com/GeorgeSpy/ESPI-pseydonoisy-generator}
}
```

## License

MIT License. See `LICENSE` for details.


## Reproducibility layer (v3.2)

In addition to the baseline generator, this repository ships the **v3.2
execution layer** used for the methodology contribution of the associated research paper
(deterministic replay, order invariance, and regime-aware calibration). v3.2
preserves the frozen scientific baseline: with `--rng-mode legacy` it produces
byte-identical output to v3.1.

### Scripts

- `make_pseudo_noisy_plus_v3_2.py` -- generator with provenance manifests,
  per-image deterministic RNG, manifest-driven conditioning, and
  global / by-regime / by-image calibration modes.
- `make_pseudo_noisy_plus_v3_1.py` -- frozen v3.1 baseline (reference for the
  backward-compatibility check).
- `phase1..5_*_validation.py` -- the reproducibility validation harness.

### Key options added in v3.2

| Option | Purpose |
|--------|---------|
| `--rng-mode {legacy, per_image}` | per-image deterministic RNG (order-independent) |
| `--conditioning-mode {global, manifest}` | per-image parameters from a manifest |
| `--calibration-mode {global, by-regime, by-image}` | regime-aware calibration |
| `--manifest`, `--write-manifest` | manifest-driven build + provenance output |
| `--write-per-image-params` | per-image parameter ledger (CSV) |
| `--blur-mode {linear, sqrt, piecewise}` | frequency->blur mapping |

### Reproducibility evidence

Running the chained validators (`phase1` -> `phase5`) confirms the three core
properties reported in the paper. Phases 1-3 yield **30/30 byte-identical
outputs with maximum deviation 0.0**:

| Phase | Property | Verdict |
|-------|----------|---------|
| 1 | backward compatibility | v3.2 reproduces v3.1 exactly (global mode) |
| 2 | order invariance | identical output regardless of file order |
| 3 | replayability | runs reproduced from provenance manifests |

See [`REPRODUCE.md`](REPRODUCE.md) for exact commands, frozen calibration
values, and path configuration.

### Scope

This repository covers **pseudo-noisy generation** only. The fixed V4
DnCNN-Lite-ECA probe and all probe-side results (validation loss, PSNR/SSIM/
EdgeF1 tables) live in the companion repository
[`ESPI-DnCNN-ECA`](https://github.com/GeorgeSpy/ESPI-DnCNN-ECA).
