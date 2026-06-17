# Minimal Examples

This folder contains three small real ESPI averaged-reference images and their
reproducible pseudo-noisy outputs generated with the v3.2 execution layer.

## Contents

- `clean/`: averaged ESPI reference images from the W01 wood set.
- `pseudo_noisy/`: pseudo-noisy outputs generated from the clean images.
- `metrics_example.csv`: per-image PSNR / SSIM / structure-proxy metrics.
- `summary_example.json`: aggregate metrics over the three example images.
- `per_image_params_example.csv`: per-image resolved parameters and RNG subseeds.
- `run_manifest_schema_example.json`: sanitized provenance-manifest schema without host-specific absolute paths.

## Exact command

Run from the repository root:

```bash
python make_pseudo_noisy_plus_v3_2.py \
  --input examples/clean \
  --output examples/pseudo_noisy \
  --seed 42 \
  --speckle-k 1.89 \
  --poisson-peak 12.5 \
  --gauss-sigma 0.0919 \
  --match meanstd \
  --rng-mode per_image \
  --out-bitdepth 8 \
  --out-format png \
  --export-metrics examples/metrics_example.csv \
  --export-summary examples/summary_example.json \
  --write-per-image-params examples/per_image_params_example.csv
```

## Expected summary

The committed run produced 3/3 pseudo-noisy outputs with:

- mean PSNR: 19.124005
- mean SSIM: 0.238541
- mean EdgeF1: 0.335768

These examples are for reproducibility and CLI demonstration only. They are not a
benchmark and should not be used to infer downstream denoising or classification
performance.

## Metric caveat

The historical `mpi` column is an edge/fringe-distance proxy. Lower distance is
better; it is not a conventional score where larger is necessarily better. The
historical `pcs` column is a gradient-smoothness proxy, not a full optical phase
coherence measure.
