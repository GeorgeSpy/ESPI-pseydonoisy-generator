#!/usr/bin/env python3
# make_pseudo_noisy_v3.py
# Gentler pseudo-noise with tunable speckle clamp + higher k to reduce variance.
# Target baseline PSNR(noisy->clean) ~ 22–26 dB.

import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def imread_f32(p: Path):
    im = Image.open(p)
    arr = np.array(im)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def imsave_8u(path: Path, arr_f32: np.ndarray):
    arr = np.clip(arr_f32, 0.0, 1.0)
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(path)

def radial_vignette(h, w, strength=0.1):
    y, x = np.ogrid[:h, :w]
    cy, cx = (h-1)/2.0, (w-1)/2.0
    r = np.sqrt((y-cy)**2 + (x-cx)**2)
    r /= (r.max() + 1e-6)
    v = 1.0 - strength * (r**2)
    return v.astype(np.float32)

def add_noise(img, args):
    x = img.copy()
    # 1) Speckle (Gamma) -> normalize to mean≈1, reduce variance via large k, clamp tighter
    if args.speckle_k > 0 and args.speckle_theta > 0:
        g = np.random.gamma(args.speckle_k, args.speckle_theta, size=x.shape).astype(np.float32)
        g = g / float(args.speckle_k * args.speckle_theta)  # mean≈1
        g = np.clip(g, args.spk_min, args.spk_max)          # clamp e.g. [0.85, 1.15]
        x = x * g
    # 2) Poisson (shot)
    if args.poisson_gain > 0:
        sc = max(1.0, args.poisson_gain * 255.0)
        lam = np.clip(x * sc, 0, None)
        x = np.random.poisson(lam).astype(np.float32) / sc
    # 3) Gaussian (readout)
    if args.gauss_sigma > 0:
        x = x + np.random.normal(0.0, args.gauss_sigma, x.shape).astype(np.float32)
    # 4) Vignetting
    if args.vignette > 0:
        v = radial_vignette(x.shape[0], x.shape[1], strength=args.vignette)
        x = x * v
    # 5) Tiny jitter (optional)
    if args.jitter > 0:
        dy = random.randint(-args.jitter, args.jitter)
        dx = random.randint(-args.jitter, args.jitter)
        x = np.roll(x, shift=(dy, dx), axis=(0,1))
    return np.clip(x, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Averaged (clean) png folder")
    ap.add_argument("--output", required=True, help="Output pseudo-noisy folder")
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--limit", type=int, default=0)

    # Gentler defaults
    ap.add_argument("--speckle-k", type=float, default=25.0)       # larger k -> lower variance
    ap.add_argument("--speckle-theta", type=float, default=(1.0/25.0))  # mean=1
    ap.add_argument("--spk-min", type=float, default=0.85)
    ap.add_argument("--spk-max", type=float, default=1.15)
    ap.add_argument("--poisson-gain", type=float, default=0.05)
    ap.add_argument("--gauss-sigma", type=float, default=0.002)
    ap.add_argument("--vignette", type=float, default=0.02)
    ap.add_argument("--jitter", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in inp.glob(f"*{args.ext}")])
    if args.limit>0: files = files[:args.limit]
    if not files:
        raise SystemExit(f"No files found under {inp}")

    for p in tqdm(files, desc="Pseudo-noisy"):
        img = imread_f32(p)
        if img.ndim == 3:
            img = img[...,0]
        noisy = add_noise(img, args)
        imsave_8u(out / p.name, noisy)

    print(f"[DONE] Wrote {len(files)} pseudo-noisy PNGs to: {out}")
    print("Tip: run dataset_baseline_psnr.py to verify Mean PSNR ~22–26 dB.")

if __name__ == "__main__":
    main()
