# Development Log

This document records the development history of the ESPI pseudo-noisy generation
pipeline. It replaces the older comprehensive archive note, which mixed useful
project history with unsupported status claims and local-machine paths.

## Scope

This repository provides research software for calibrated pseudo-noisy ESPI
generation. It supports controlled synthetic-supervision experiments,
synthetic-real gap analysis, and reproducibility validation for the generator
layer. It is not a universal ESPI noise model and does not, by itself, establish
downstream denoising or classification performance.

## Generator Model

The generator uses a physically motivated corruption cascade:

1. multiplicative Gamma speckle as a coherent-noise proxy,
2. Poisson shot noise as a photon-counting proxy,
3. additive Gaussian readout noise as an electronic/quantization proxy,
4. optional frequency-aware blur and optional ROI-aware spatial modulation.

The v3.2 execution layer adds provenance manifests, deterministic per-image RNG,
manifest-driven conditioning, and reproducible calibration modes.

## Calibration

Calibration is moment-based and uses real single-shot images paired with averaged
references. The pipeline estimates parameters from real single-shot / averaged
pairs; it does not claim a full maximum-likelihood physical sensor model.

The paper-level v3.2 calibration blocks are:

| Setting | k | poisson peak | sigma |
|---|---:|---:|---:|
| global | 2.1112 | 24.7652 | 0.08613 |
| by-regime Mid | 2.0147 | 15.2205 | 0.08422 |
| by-regime High | 2.2379 | 27.1845 | 0.08727 |

The canonical Mid baseline used in the main experiments is:

| k | poisson peak | sigma |
|---:|---:|---:|
| 1.89 | 12.5 | 0.0919 |

## Metrics

The repository reports image-similarity and structure-proxy metrics. These should
be interpreted carefully:

- PSNR and SSIM compare pseudo-noisy images against averaged references.
- EdgeF1 is an edge-overlap proxy, not a complete physical quality score.
- The historical MPI naming refers to an edge/fringe-distance proxy based on
  Sobel/Hausdorff-style contour distance. Lower distance is better; it should
  not be read as a conventional index where larger is always better.
- Historical phase-coherence wording should be read only as a gradient-smoothness
  proxy unless explicit wrapped/unwrapped phase maps are used. This repository
  does not implement a full optical phase-reconstruction pipeline.

## Validation Status

The v3.2 validators support the reproducibility claim for the generator layer:

| Phase | Result |
|---|---|
| Phase 1: backward compatibility | 30/30 byte-identical, max deviation 0.0 |
| Phase 2: order invariance | 30/30 byte-identical, max deviation 0.0 |
| Phase 3: replayability | 30/30 byte-identical, max deviation 0.0 |
| Phase 4: calibration modes | reproducible global / by-regime calibration blocks |
| Phase 5: utility validation | structurally valid; practical utility differences limited |

Phase 5 depends on the companion denoising repository and a V4 DnCNN-Lite-ECA
checkpoint. Checkpoints and probe-side outputs are not stored in this repository.

## What This Repository Does Not Claim

This repository does not claim:

- universal ESPI noise modeling,
- uniformly superior denoising performance,
- standalone downstream classification performance,
- validated optical phase recovery,
- hardware-independent generation speed benchmarks,
- complete replacement for real single-shot / averaged training pairs.

Any downstream denoising or classification result must be interpreted together
with the companion repositories and their evaluation protocols.

## Reproducibility Notes

Use `REPRODUCE.md` for exact commands and expected validation outputs. Generated
validation artifacts should be written to `_validation_out/`, which is ignored by
git. Public examples live under `examples/` and are intentionally small.
