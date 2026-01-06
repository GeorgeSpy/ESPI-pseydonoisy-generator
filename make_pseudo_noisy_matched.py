#!/usr/bin/env python3
# make_pseudo_noisy_matched.py
# Create pseudo-noisy images from clean PNGs while matching per-image mean/std
# so that PSNR stays in a reasonable range. Handles 8/16-bit grayscale PNGs.

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def imread_gray_f32(p: Path):
    im = Image.open(p).convert("I;16") if Image.open(p).mode in ("I;16","I;16B","I;16L") else Image.open(p).convert("L")
    arr = np.array(im)
    if arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
        bd = 16
    else:
        x = arr.astype(np.float32) / 255.0
        bd = 8
    return np.clip(x,0,1), bd

def imsave_gray(x: np.ndarray, p: Path, bitdepth: int):
    x = np.clip(x, 0.0, 1.0)
    if bitdepth == 16:
        arr = (x*65535.0 + 0.5).astype(np.uint16)
        Image.fromarray(arr, mode="I;16").save(p)
    else:
        arr = (x*255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(p)

def gamma_speckle(h, w, sigma):
    # Create multiplicative speckle with mean=1 and std=sigma
    # Gamma distribution with shape k=1/sigma^2, scale theta=sigma^2
    sigma = max(0.0, float(sigma))
    if sigma < 1e-6:
        return np.ones((h,w), dtype=np.float32)
    k = 1.0/(sigma**2 + 1e-12)
    theta = sigma**2
    sp = np.random.gamma(k, theta, size=(h,w)).astype(np.float32)
    # mean= k*theta = 1  (by construction)
    return sp

def add_poisson(x, scale):
    if scale <= 0: return x
    lam = np.clip(x, 0, 1) * float(scale)
    y = np.random.poisson(lam).astype(np.float32) / float(scale)
    return y

def add_gauss(x, sigma):
    if sigma <= 0: return x
    return x + np.random.normal(0.0, float(sigma), size=x.shape).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with clean PNGs")
    ap.add_argument("--output", required=True, help="Output folder for pseudo-noisy PNGs")
    # noise controls
    ap.add_argument("--speckle-sigma", type=float, default=0.05, help="Std of multiplicative speckle (mean=1)")
    ap.add_argument("--poisson-scale", type=float, default=512.0, help="Poisson shots scale (0=disable)")
    ap.add_argument("--gauss-sigma", type=float, default=0.002, help="Additive Gaussian sigma")
    # matching
    ap.add_argument("--match", choices=["none","mean","meanstd"], default="meanstd",
                    help="Match output to input statistics")
    ap.add_argument("--bitdepth", type=int, choices=[8,16], default=16, help="Save depth (8 or 16)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.input); out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    if args.limit>0: files = files[:args.limit]
    if not files:
        print("No PNGs found in input."); return

    for i, f in enumerate(files, 1):
        clean, bd_in = imread_gray_f32(f)
        h,w = clean.shape

        # Record clean stats
        mu = float(np.mean(clean)); sd = float(np.std(clean))

        # Build noise
        sp = gamma_speckle(h, w, args.speckle_sigma)
        noisy = clean * sp
        noisy = add_poisson(noisy, args.poisson_scale)
        noisy = add_gauss(noisy, args.gauss_sigma)

        # Match stats
        if args.match == "mean":
            mu_n = float(np.mean(noisy))
            if mu_n > 1e-8:
                noisy = noisy * (mu / mu_n)
        elif args.match == "meanstd":
            mu_n = float(np.mean(noisy)); sd_n = float(np.std(noisy))
            if sd_n < 1e-8:
                sd_n = 1e-8
            noisy = (noisy - mu_n) * (sd / sd_n) + mu

        noisy = np.clip(noisy, 0.0, 1.0)
        imsave_gray(noisy, out_dir/f.name, args.bitdepth)

        if i % 50 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {f.name}  mu={mu:.3f}→{float(np.mean(noisy)):.3f}  sd={sd:.3f}→{float(np.std(noisy)):.3f}")

    print(f"[DONE] Wrote {len(files)} pseudo-noisy PNGs to: {out_dir}")

if __name__ == "__main__":
    main()
