#!/usr/bin/env python3
"""
make_pseudo_noisy_plus_v3.py

Improved, physics-aware ESPI synthetic noise generator.

Key improvements over the original version
- robust grayscale I/O for 8-bit and 16-bit images
- optional per-image mean / mean-std matching after noise synthesis
- corrected multiplicative speckle logic so spatial speckle does not stack on top
  of a separate global speckle pass
- configurable output format (PNG or TIFF)
- calibration that uses residual statistics more robustly
- optional metrics summary JSON and richer CSV metadata
- configurable temporal correlation for sequences
- stricter validation and safer ROI handling
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    import cv2 as _cv2  # optional
except Exception:
    _cv2 = None

try:
    from scipy.signal import convolve2d as _sp_convolve2d  # optional
except Exception:
    _sp_convolve2d = None

try:
    from PIL import Image
except Exception:
    print("Pillow is required. Please `pip install pillow`.", file=sys.stderr)
    raise

IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


# ------------------------------ filesystem / image I/O ------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def gather_files(p: Path) -> list[Path]:
    return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS)


def _to_float01(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return np.clip(x.astype(np.float32) / 255.0, 0.0, 1.0)
    if x.dtype == np.uint16:
        return np.clip(x.astype(np.float32) / 65535.0, 0.0, 1.0)
    if np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
        if x.min() >= 0.0 and x.max() <= 1.0:
            return np.clip(x, 0.0, 1.0)
        rng = float(x.max() - x.min())
        if rng <= 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - x.min()) / rng, 0.0, 1.0)
    x = x.astype(np.float32)
    x -= x.min()
    rng = float(x.max() - x.min())
    if rng <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / rng, 0.0, 1.0)


def imread_uint01(path: Path) -> np.ndarray:
    """Read image as float32 grayscale in [0, 1]."""
    with Image.open(path) as im:
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.array(im)
        else:
            # Convert RGB / palette / 8-bit grayscale through luminance.
            arr = np.array(im.convert("L"))
    return _to_float01(arr)


def imsave_uint01(arr: np.ndarray, out_path: Path, bitdepth: int = 8, out_format: str = "png") -> Path:
    arr = np.clip(arr, 0.0, 1.0)
    ensure_dir(out_path.parent)

    ext = ".png" if out_format == "png" else ".tiff"
    out_path = out_path.with_suffix(ext)

    if bitdepth == 16:
        data = (arr * 65535.0 + 0.5).astype(np.uint16)
        img = Image.fromarray(data)
    else:
        data = (arr * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(data)

    img.save(out_path)
    return out_path


# ------------------------------ convolution helpers ------------------------------

def _pad_symmetric(x: np.ndarray, ph: int, pw: int) -> np.ndarray:
    return np.pad(x, ((ph, ph), (pw, pw)), mode="symmetric")


def _conv2d_same_fallback(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    xpad = _pad_symmetric(x, ph, pw)
    y = np.zeros_like(x, dtype=np.float32)
    kk = np.flipud(np.fliplr(k)).astype(np.float32)
    for i in range(y.shape[0]):
        row = xpad[i : i + kh]
        for j in range(y.shape[1]):
            patch = row[:, j : j + kw]
            y[i, j] = float((patch * kk).sum())
    return y


def conv2d_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    if _cv2 is not None:
        try:
            return _cv2.filter2D(x, -1, k, borderType=_cv2.BORDER_REFLECT).astype(np.float32)
        except Exception:
            pass
    if _sp_convolve2d is not None:
        try:
            return _sp_convolve2d(x, k, mode="same", boundary="symm").astype(np.float32)
        except Exception:
            pass
    return _conv2d_same_fallback(x, k)


# ------------------------------ statistics helpers ------------------------------

def _uniform_filter(img: np.ndarray, win: int) -> np.ndarray:
    """Fast box filter using convolution when possible; fallback otherwise."""
    win = max(1, int(win))
    kernel = np.full((win, win), 1.0 / float(win * win), dtype=np.float32)
    return conv2d_same(img.astype(np.float32), kernel)


def _local_mean_var(x: np.ndarray, win: int = 9) -> tuple[np.ndarray, np.ndarray]:
    mean = _uniform_filter(x, win)
    mean2 = _uniform_filter(x * x, win)
    var = np.clip(mean2 - mean * mean, 0.0, None)
    return mean, var


def apply_match_stats(clean: np.ndarray, noisy: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.clip(noisy, 0.0, 1.0)

    mu = float(np.mean(clean))
    sd = float(np.std(clean))
    mu_n = float(np.mean(noisy))
    sd_n = float(np.std(noisy))

    y = noisy.copy()
    if mode == "mean":
        if abs(mu_n) > 1e-8:
            y = y * (mu / mu_n)
    elif mode == "meanstd":
        sd_n = max(sd_n, 1e-8)
        y = (y - mu_n) * (sd / sd_n) + mu

    return np.clip(y, 0.0, 1.0)


# ------------------------------ noise primitives ------------------------------

def sample_gamma_speckle_field(shape: tuple[int, int], k_field: np.ndarray | float,
                               rng: np.random.Generator) -> np.ndarray:
    if np.isscalar(k_field):
        k = float(k_field)
        if k <= 0:
            return np.ones(shape, dtype=np.float32)
        return rng.gamma(shape=k, scale=1.0 / k, size=shape).astype(np.float32)

    k_map = np.asarray(k_field, dtype=np.float32)
    out = np.ones(shape, dtype=np.float32)
    # Quantize for efficiency.
    q = np.clip(np.round(k_map * 2.0) / 2.0, 0.5, 12.0)
    for kv in np.unique(q):
        mask = q == kv
        if not np.any(mask):
            continue
        sp = rng.gamma(shape=float(kv), scale=1.0 / float(kv), size=int(mask.sum())).astype(np.float32)
        out[mask] = sp
    return out


def add_speckle_multiplicative(img01: np.ndarray, k_field: np.ndarray | float,
                               rng: np.random.Generator) -> np.ndarray:
    speckle = sample_gamma_speckle_field(img01.shape, k_field, rng)
    return np.clip(img01 * speckle, 0.0, 1.0)


def add_poisson_shot(img01: np.ndarray, peak: float, rng: np.random.Generator) -> np.ndarray:
    if peak <= 0:
        return img01
    counts = np.clip(img01 * peak, 0.0, None)
    noisy = rng.poisson(counts).astype(np.float32) / float(peak)
    return np.clip(noisy, 0.0, 1.0)


def add_gaussian(img01: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img01
    return np.clip(img01 + rng.normal(0.0, sigma, size=img01.shape).astype(np.float32), 0.0, 1.0)


# ------------------------------ blur ------------------------------

def _motion_blur_kernel(length_px: int, angle_deg: float) -> np.ndarray:
    L = max(1, int(length_px))
    if L % 2 == 0:
        L += 1
    k = np.zeros((L, L), np.float32)
    rr = np.arange(L) - (L - 1) / 2.0
    xx, yy = np.meshgrid(rr, rr, indexing="xy")
    th = math.radians(angle_deg)
    line = np.abs(xx * math.cos(th) + yy * math.sin(th)) < 0.5
    k[line] = 1.0
    s = float(k.sum())
    return k / s if s > 0 else np.eye(1, dtype=np.float32)


def add_frequency_aware_blur(img01: np.ndarray, freq_hz: float, amp_db: float,
                             blur_scale: float, rng: np.random.Generator) -> np.ndarray:
    length_px = int(1 + blur_scale * (0.02 * float(freq_hz) + 0.05 * max(float(amp_db), 0.0)))
    length_px = max(1, min(length_px, 101))
    angle = float(rng.uniform(0.0, 180.0))
    k = _motion_blur_kernel(length_px, angle)
    return np.clip(conv2d_same(img01, k), 0.0, 1.0)


# ------------------------------ ROI / spatial speckle ------------------------------

def _resize_mask_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    img = img.resize((w, h), resample=Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def _distance_map(roi_mask: np.ndarray) -> np.ndarray:
    """Normalized distance-to-ROI map in [0, 1]."""
    mask = (roi_mask > 0).astype(np.uint8)
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
        dist = distance_transform_edt(1 - mask).astype(np.float32)
    except Exception:
        h, w = mask.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        dist = np.hypot(yy - cy, xx - cx).astype(np.float32)
    mx = float(dist.max())
    if mx <= 1e-8:
        return np.zeros_like(dist, dtype=np.float32)
    return dist / mx


def build_spatial_k_map(base_k: float, roi_mask: np.ndarray | None, strength: float,
                        img_shape: tuple[int, int]) -> np.ndarray | float:
    if roi_mask is None or base_k <= 0 or strength <= 0:
        return float(base_k)
    if roi_mask.shape != img_shape:
        roi_mask = _resize_mask_nearest(roi_mask, img_shape)
    dist = _distance_map(roi_mask)
    k_map = base_k * (1.0 + float(strength) * dist)
    return np.clip(k_map, 0.5, 12.0).astype(np.float32)


# ------------------------------ calibration ------------------------------

def _gamma_shape_from_contrast(C: float) -> float:
    C = max(float(C), 1e-6)
    return 1.0 / (C * C)


def _clip_with_warning(value: float, lo: float, hi: float, name: str) -> float:
    clipped = float(np.clip(value, lo, hi))
    if not math.isclose(clipped, float(value), rel_tol=0.0, abs_tol=1e-12):
        warnings.warn(
            f"[CALIB] {name} estimate {value:.6g} hit clip bound and was clamped to {clipped:.6g} "
            f"(allowed range [{lo}, {hi}]). Check calibration data quality.",
            RuntimeWarning,
            stacklevel=2,
        )
    return clipped


def estimate_noise_params_from_pair(single01: np.ndarray, avg01: np.ndarray, win: int = 9) -> tuple[float, float, float]:
    eps = 1e-6

    # Speckle estimate from multiplicative residual.
    ratio = np.clip(single01 / (avg01 + eps), 0.0, 4.0).astype(np.float32)
    m, v = _local_mean_var(ratio, win=win)
    valid = np.isfinite(m) & np.isfinite(v) & (avg01 > 0.03)
    if np.any(valid):
        local_cv = np.sqrt(v[valid]) / (m[valid] + eps)
        C = float(np.median(local_cv))
    else:
        C = float(np.sqrt(np.mean(v)) / (np.mean(m) + eps))
    k_hat = _clip_with_warning(_gamma_shape_from_contrast(C), 0.5, 12.0, "k")

    # Residual variance model: diff^2 ~= a * avg + b.
    diff = (single01 - avg01).astype(np.float32)
    x = avg01.ravel()
    y = (diff * diff).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.02) & (x < 0.98)
    x = x[mask]
    y = y[mask]
    if x.size >= 256:
        bins = np.linspace(0.02, 0.98, 17)
        xc, yc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            bm = (x >= lo) & (x < hi)
            if np.count_nonzero(bm) < 32:
                continue
            xc.append(float(np.median(x[bm])))
            yc.append(float(np.median(y[bm])))
        if len(xc) >= 4:
            a, b = np.polyfit(np.asarray(xc, dtype=np.float32), np.asarray(yc, dtype=np.float32), 1)
        else:
            a, b = np.polyfit(x[: min(5000, x.size)], y[: min(5000, y.size)], 1)
    else:
        a, b = 1.0, float(np.var(diff))

    a = float(max(a, 1e-6))
    b = float(max(b, 0.0))
    peak_hat = _clip_with_warning(1.0 / a, 1.0, 4096.0, "poisson_peak")
    sigma_hat = _clip_with_warning(math.sqrt(b), 0.0, 0.25, "gauss_sigma")
    return k_hat, peak_hat, sigma_hat


def analyze_real_noise(single_dir: Path, averaged_dir: Path, max_pairs: int = 64) -> tuple[float, float, float] | None:
    import re
    singles = gather_files(single_dir)
    avgs = gather_files(averaged_dir)
    avg_map = {f.name: f for f in avgs}
    for f in avgs:
        avg_map[f.stem] = f

    ks, peaks, sigs = [], [], []
    cnt = 0
    for f in singles:
        af = avg_map.get(f.name)
        if af is None:
            m = re.match(r"(.*?)_\d+$", f.stem)
            if m:
                af = avg_map.get(m.group(1))
        if af is None:
            continue
        s = imread_uint01(f)
        a = imread_uint01(af)
        if s.shape != a.shape:
            continue
        k, p, sg = estimate_noise_params_from_pair(s, a)
        ks.append(k)
        peaks.append(p)
        sigs.append(sg)
        cnt += 1
        if cnt >= max_pairs:
            break

    if not ks:
        print("[CALIB] No matched pairs.")
        return None

    k_mean = float(np.median(ks))
    p_mean = float(np.median(peaks))
    s_mean = float(np.median(sigs))
    print(f"[CALIB] k={k_mean:.2f}  peak={p_mean:.1f}  sigma={s_mean:.4f}  from {cnt} pairs")
    return k_mean, p_mean, s_mean


# ------------------------------ metrics ------------------------------

def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(x: np.ndarray, y: np.ndarray, win: int = 11, K1: float = 0.01, K2: float = 0.03) -> float:
    C1 = K1 ** 2
    C2 = K2 ** 2
    ux = _uniform_filter(x, win)
    uy = _uniform_filter(y, win)
    uxx = _uniform_filter(x * x, win)
    uyy = _uniform_filter(y * y, win)
    uxy = _uniform_filter(x * y, win)
    vx = np.clip(uxx - ux * ux, 0.0, None)
    vy = np.clip(uyy - uy * uy, 0.0, None)
    vxy = uxy - ux * uy
    num = (2 * ux * uy + C1) * (2 * vxy + C2)
    den = (ux * ux + uy * uy + C1) * (vx + vy + C2)
    return float(np.mean(num / (den + 1e-12)))


def _edge_map_sobel(x: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 8.0
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 8.0
    gx = conv2d_same(x, kx)
    gy = conv2d_same(x, ky)
    mag = np.hypot(gx, gy)
    th = float(np.mean(mag) + 0.5 * np.std(mag))
    return (mag >= th).astype(np.uint8)


def _sample_points(mask: np.ndarray, max_pts: int = 1500) -> np.ndarray:
    pts = np.column_stack(np.nonzero(mask))
    if len(pts) == 0:
        return pts.astype(np.float32)
    if len(pts) <= max_pts:
        return pts.astype(np.float32)
    idx = np.random.default_rng(123).choice(len(pts), size=max_pts, replace=False)
    return pts[idx].astype(np.float32)


def _directed_hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    if len(A) == 0 or len(B) == 0:
        return 1e3
    diff = A[:, None, :] - B[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    mins = np.min(d2, axis=1)
    return float(np.sqrt(np.max(mins)))


def modal_preservation_index(clean: np.ndarray, test: np.ndarray) -> float:
    e1 = _edge_map_sobel(clean)
    e2 = _edge_map_sobel(test)
    p1 = _sample_points(e1)
    p2 = _sample_points(e2)
    return 0.5 * (_directed_hausdorff(p1, p2) + _directed_hausdorff(p2, p1))


def phase_coherence_score(x: np.ndarray) -> float:
    gy, gx = np.gradient(x)
    gmag = np.hypot(gy, gx)
    return float(-np.std(gmag))


def summarize_metrics(rows: list[dict]) -> dict:
    out: dict[str, dict[str, float | int]] = {}
    modes = sorted(set(r["mode"] for r in rows))
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "count": len(sub),
            "psnr_mean": float(np.mean([float(r["psnr"]) for r in sub])),
            "psnr_std": float(np.std([float(r["psnr"]) for r in sub])),
            "ssim_mean": float(np.mean([float(r["ssim"]) for r in sub])),
            "ssim_std": float(np.std([float(r["ssim"]) for r in sub])),
            "mpi_mean": float(np.mean([float(r["mpi"]) for r in sub])),
            "mpi_std": float(np.std([float(r["mpi"]) for r in sub])),
            "pcs_mean": float(np.mean([float(r["pcs"]) for r in sub])),
            "pcs_std": float(np.std([float(r["pcs"]) for r in sub])),
        }
    return out


# ------------------------------ config ------------------------------

PROFILE_PRESETS = {
    "lite": {"k": 2.4, "peak": 50.0, "sigma": 0.012},
    "mid": {"k": 2.9, "peak": 58.0, "sigma": 0.016},
    "heavy": {"k": 3.4, "peak": 66.0, "sigma": 0.022},
}

MATERIAL_PROFILES = {
    "wood": {"k": 2.8, "peak": 55.0, "sigma": 0.015},
    "carbon_fiber": {"k": 3.5, "peak": 70.0, "sigma": 0.008},
}


@dataclass
class Args:
    input: Path
    output: Path
    seed: int
    speckle_k: float
    poisson_peak: float
    gauss_sigma: float
    freq_hz: float | None
    amp_db: float | None
    blur_scale: float
    material: str | None
    profile: str | None
    roi_mask: Path | None
    spatial_speckle: bool
    spatial_strength: float
    calib_single: Path | None
    calib_avg: Path | None
    calib_override: bool
    calib_max_pairs: int
    out_bitdepth: int
    out_format: str
    n_frames: int
    seq_alpha: float
    match: str
    ablate: str | None
    export_metrics: Path | None
    export_summary: Path | None


# ------------------------------ pipeline ------------------------------

def validate_args(args: Args) -> None:
    if args.speckle_k < 0:
        raise ValueError("--speckle-k must be >= 0")
    if args.poisson_peak < 0:
        raise ValueError("--poisson-peak must be >= 0")
    if args.gauss_sigma < 0:
        raise ValueError("--gauss-sigma must be >= 0")
    if args.n_frames < 0:
        raise ValueError("--n-frames must be >= 0 (use 0 or 1 for single-image mode)")
    if args.calib_max_pairs < 1:
        raise ValueError("--calib-max-pairs must be >= 1")
    if not (0.0 <= args.spatial_strength <= 2.0):
        raise ValueError("--spatial-strength must be in [0, 2]")
    if not (0.0 <= args.seq_alpha <= 1.0):
        raise ValueError("--seq-alpha must be in [0, 1]")
    if args.roi_mask is not None and not args.roi_mask.exists():
        raise FileNotFoundError(f"ROI mask not found: {args.roi_mask}")
    if args.spatial_speckle and args.roi_mask is None:
        raise ValueError("--spatial-speckle requires --roi-mask")


def resolve_noise_defaults(args: Args, ap: argparse.ArgumentParser, args_ns: argparse.Namespace) -> Args:
    out = args
    if args.material:
        prof = MATERIAL_PROFILES[args.material]
        print(f"[MATERIAL] defaults: {prof}")
        if args_ns.speckle_k == ap.get_default("speckle_k"):
            out = replace(out, speckle_k=prof["k"])
        if args_ns.poisson_peak == ap.get_default("poisson_peak"):
            out = replace(out, poisson_peak=prof["peak"])
        if args_ns.gauss_sigma == ap.get_default("gauss_sigma"):
            out = replace(out, gauss_sigma=prof["sigma"])
    if args.profile:
        preset = PROFILE_PRESETS[args.profile]
        print(f"[PROFILE] defaults: {preset}")
        if args_ns.speckle_k == ap.get_default("speckle_k"):
            out = replace(out, speckle_k=preset["k"])
        if args_ns.poisson_peak == ap.get_default("poisson_peak"):
            out = replace(out, poisson_peak=preset["peak"])
        if args_ns.gauss_sigma == ap.get_default("gauss_sigma"):
            out = replace(out, gauss_sigma=preset["sigma"])
    return out


def add_noise_chain(clean01: np.ndarray, rng: np.random.Generator, args: Args,
                    roi_mask_arr: np.ndarray | None = None) -> np.ndarray:
    y = clean01.copy()

    if args.freq_hz is not None and args.amp_db is not None:
        y = add_frequency_aware_blur(y, args.freq_hz, args.amp_db, args.blur_scale, rng)

    if args.ablate != "no-speckle":
        k_field: np.ndarray | float = args.speckle_k
        if args.spatial_speckle and roi_mask_arr is not None:
            k_field = build_spatial_k_map(args.speckle_k, roi_mask_arr, args.spatial_strength, y.shape)
        y = add_speckle_multiplicative(y, k_field, rng)

    if args.ablate != "no-poisson":
        y = add_poisson_shot(y, args.poisson_peak, rng)

    if args.ablate != "no-gaussian":
        y = add_gaussian(y, args.gauss_sigma, rng)

    y = apply_match_stats(clean01, y, args.match)
    return np.clip(y, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Improved calibrated ESPI synthetic noise generator")
    ap.add_argument("--input", type=Path, required=True, help="Folder with clean images")
    ap.add_argument("--output", type=Path, required=True, help="Output folder for pseudo-noisy images")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--speckle-k", type=float, default=3.0)
    ap.add_argument("--poisson-peak", type=float, default=60.0)
    ap.add_argument("--gauss-sigma", type=float, default=0.01)
    ap.add_argument("--freq-hz", type=float, default=None, help="Excitation frequency (Hz)")
    ap.add_argument("--amp-db", type=float, default=None, help="Excitation amplitude (dB)")
    ap.add_argument("--blur-scale", type=float, default=1.0, help="Global multiplier for blur length")
    ap.add_argument("--material", type=str, default=None, choices=list(MATERIAL_PROFILES.keys()))
    ap.add_argument("--profile", type=str, default=None, choices=list(PROFILE_PRESETS.keys()))
    ap.add_argument("--roi-mask", type=Path, default=None, help="ROI mask image (white=ROI)")
    ap.add_argument("--spatial-speckle", action="store_true", help="Enable spatially varying speckle")
    ap.add_argument("--spatial-strength", type=float, default=0.3)
    ap.add_argument("--calib-single", type=Path, default=None, help="Folder with real single-shot images")
    ap.add_argument("--calib-avg", type=Path, default=None, help="Folder with averaged reference images")
    ap.add_argument("--calib-override", action="store_true")
    ap.add_argument("--calib-max-pairs", type=int, default=64)
    ap.add_argument("--out-bitdepth", "--bitdepth", dest="out_bitdepth", type=int, default=8, choices=[8, 16])
    ap.add_argument("--out-format", type=str, default="png", choices=["png", "tiff"])
    ap.add_argument("--n-frames", type=int, default=0, help="If >1, generate a temporal sequence per image")
    ap.add_argument("--seq-alpha", type=float, default=0.15,
                    help="Temporal update factor: new = (1-alpha)*prev + alpha*current")
    ap.add_argument("--match", type=str, default="none", choices=["none", "mean", "meanstd"],
                    help="Optional post-noise per-image statistics matching")
    ap.add_argument("--ablate", type=str, default=None,
                    choices=["no-speckle", "no-poisson", "no-gaussian", "full", "all"])
    ap.add_argument("--export-metrics", type=Path, default=None, help="CSV path for per-image metrics")
    ap.add_argument("--export-summary", type=Path, default=None, help="JSON path for aggregated metrics summary")

    args_ns = ap.parse_args()
    args = Args(
        input=args_ns.input,
        output=args_ns.output,
        seed=args_ns.seed,
        speckle_k=args_ns.speckle_k,
        poisson_peak=args_ns.poisson_peak,
        gauss_sigma=args_ns.gauss_sigma,
        freq_hz=args_ns.freq_hz,
        amp_db=args_ns.amp_db,
        blur_scale=args_ns.blur_scale,
        material=args_ns.material,
        profile=args_ns.profile,
        roi_mask=args_ns.roi_mask,
        spatial_speckle=args_ns.spatial_speckle,
        spatial_strength=args_ns.spatial_strength,
        calib_single=args_ns.calib_single,
        calib_avg=args_ns.calib_avg,
        calib_override=args_ns.calib_override,
        calib_max_pairs=args_ns.calib_max_pairs,
        out_bitdepth=args_ns.out_bitdepth,
        out_format=args_ns.out_format,
        n_frames=args_ns.n_frames,
        seq_alpha=args_ns.seq_alpha,
        match=args_ns.match,
        ablate=args_ns.ablate,
        export_metrics=args_ns.export_metrics,
        export_summary=args_ns.export_summary,
    )

    validate_args(args)
    args = resolve_noise_defaults(args, ap, args_ns)
    rng = np.random.default_rng(args.seed)

    if args.calib_single and args.calib_avg:
        res = analyze_real_noise(args.calib_single, args.calib_avg, max_pairs=args.calib_max_pairs)
        if res and args.calib_override:
            ck, cp, cs = res
            args = replace(args, speckle_k=ck, poisson_peak=cp, gauss_sigma=cs)
            print(f"[CALIB->ARGS] Using k={ck:.2f}, peak={cp:.1f}, sigma={cs:.4f}")
        elif res:
            ck, cp, cs = res
            print(f"[CALIB] Estimated (k, peak, sigma)=({ck:.2f}, {cp:.1f}, {cs:.4f}) (not overriding)")

    roi_mask_arr = None
    if args.roi_mask and args.roi_mask.exists():
        roi_mask_arr = (imread_uint01(args.roi_mask) > 0.5).astype(np.uint8)

    files = gather_files(args.input)
    if not files:
        print(f"No images found under {args.input}", file=sys.stderr)
        sys.exit(1)

    ablate_modes: list[str | None]
    if args.ablate == "all":
        ablate_modes = [None, "no-speckle", "no-poisson", "no-gaussian"]
    elif args.ablate in {"no-speckle", "no-poisson", "no-gaussian"}:
        ablate_modes = [args.ablate]
    else:
        ablate_modes = [None]

    metrics_rows: list[dict[str, str]] = []
    total = len(files)
    t0 = time.time()

    for idx, f in enumerate(files, 1):
        clean = imread_uint01(f)
        base = f.stem

        for abl in ablate_modes:
            mode_tag = "full" if abl is None else abl
            out_dir = args.output if args.ablate != "all" else (args.output / mode_tag)
            ensure_dir(out_dir)
            run_args = replace(args, ablate=abl)

            if args.n_frames > 1:
                prev = None
                seq_dir = out_dir / f"{base}_seq"
                ensure_dir(seq_dir)
                for t in range(args.n_frames):
                    y = add_noise_chain(clean, rng, run_args, roi_mask_arr=roi_mask_arr)
                    if prev is not None:
                        alpha = args.seq_alpha
                        y = np.clip((1.0 - alpha) * prev + alpha * y, 0.0, 1.0)
                    prev = y
                    out_path = imsave_uint01(y, seq_dir / f"{base}_t{t:03d}", bitdepth=args.out_bitdepth, out_format=args.out_format)
                    metrics_rows.append({
                        "source": str(f),
                        "file": str(out_path),
                        "mode": mode_tag,
                        "match": args.match,
                        "psnr": f"{psnr(clean, y):.3f}",
                        "ssim": f"{ssim(clean, y):.4f}",
                        "mpi": f"{modal_preservation_index(clean, y):.3f}",
                        "pcs": f"{phase_coherence_score(y):.5f}",
                        "speckle_k": f"{run_args.speckle_k:.4f}",
                        "poisson_peak": f"{run_args.poisson_peak:.4f}",
                        "gauss_sigma": f"{run_args.gauss_sigma:.6f}",
                    })
            else:
                noisy = add_noise_chain(clean, rng, run_args, roi_mask_arr=roi_mask_arr)
                out_path = imsave_uint01(noisy, out_dir / f"{base}_{mode_tag}", bitdepth=args.out_bitdepth, out_format=args.out_format)
                metrics_rows.append({
                    "source": str(f),
                    "file": str(out_path),
                    "mode": mode_tag,
                    "match": args.match,
                    "psnr": f"{psnr(clean, noisy):.3f}",
                    "ssim": f"{ssim(clean, noisy):.4f}",
                    "mpi": f"{modal_preservation_index(clean, noisy):.3f}",
                    "pcs": f"{phase_coherence_score(noisy):.5f}",
                    "speckle_k": f"{run_args.speckle_k:.4f}",
                    "poisson_peak": f"{run_args.poisson_peak:.4f}",
                    "gauss_sigma": f"{run_args.gauss_sigma:.6f}",
                })

        if idx % 5 == 0 or idx == total:
            dt = time.time() - t0
            print(f"[{idx}/{total}] processed in {dt:.1f}s")

    if args.export_metrics and metrics_rows:
        ensure_dir(args.export_metrics.parent)
        with open(args.export_metrics, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"[METRICS] Wrote {args.export_metrics}")

    if args.export_summary and metrics_rows:
        summary = summarize_metrics(metrics_rows)
        ensure_dir(args.export_summary.parent)
        with open(args.export_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[SUMMARY] Wrote {args.export_summary}")
        print(json.dumps(summary, indent=2))

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
