#!/usr/bin/env python3
"""
make_pseudo_noisy_plus_v3_2.py

Methodology-first upgrade of the v3.1 ESPI pseudo-noisy generator.

Design principles
- keep v3.1 frozen; this is a new branch/version
- preserve the baseline global CLI path when no manifest is used
- add manifest-driven per-image conditioning and full provenance
- make generation deterministic per image, independent of file ordering
- support regime-aware calibration summaries and optional per-regime application
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

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


def normalize_relpath(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except Exception:
        return path.name


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
    with Image.open(path) as im:
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.array(im)
        else:
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


def resolve_blur_length_px(freq_hz: float, amp_db: float, blur_scale: float,
                           blur_mode: str, piecewise_break_hz: float = 680.0) -> int:
    freq_hz = float(freq_hz)
    amp_db = float(amp_db)
    if blur_mode == "sqrt":
        blur_driver = 0.65 * math.sqrt(max(freq_hz, 0.0)) + 0.05 * max(amp_db, 0.0)
    elif blur_mode == "piecewise":
        if freq_hz <= piecewise_break_hz:
            blur_driver = 0.02 * max(freq_hz, 0.0) + 0.05 * max(amp_db, 0.0)
        else:
            low = 0.02 * piecewise_break_hz
            high = 0.012 * (freq_hz - piecewise_break_hz)
            blur_driver = low + high + 0.05 * max(amp_db, 0.0)
    else:  # linear / backward-compatible default
        blur_driver = 0.02 * max(freq_hz, 0.0) + 0.05 * max(amp_db, 0.0)

    length_px = int(1 + blur_scale * blur_driver)
    return max(1, min(length_px, 101))


def add_frequency_aware_blur(img01: np.ndarray, freq_hz: float, amp_db: float,
                             blur_scale: float, blur_mode: str,
                             rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Apply frequency-aware motion blur.

    v3.2 signature change vs v3.1: returns (blurred_image, blur_length_px) tuple
    instead of just blurred_image, so that the resolved blur length can be
    recorded in the per-image provenance ledger. The blur math itself is
    identical to v3.1 when blur_mode='linear'.
    """
    length_px = resolve_blur_length_px(freq_hz, amp_db, blur_scale, blur_mode)
    angle = float(rng.uniform(0.0, 180.0))
    k = _motion_blur_kernel(length_px, angle)
    return np.clip(conv2d_same(img01, k), 0.0, 1.0), length_px


# ------------------------------ ROI / spatial speckle ------------------------------

def _resize_mask_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    img = img.resize((w, h), resample=Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def _distance_map(roi_mask: np.ndarray) -> np.ndarray:
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

@dataclass
class ClipTracker:
    total: int = 0
    k_hits: int = 0
    peak_hits: int = 0
    sigma_hits: int = 0

    def bump(self, name: str) -> None:
        self.total += 1
        if name == "k":
            self.k_hits += 1
        elif name == "poisson_peak":
            self.peak_hits += 1
        elif name == "gauss_sigma":
            self.sigma_hits += 1


@dataclass
class CalibrationPairRecord:
    key: str
    filename: str
    regime: str
    k: float
    peak: float
    sigma: float
    matched: bool


@dataclass
class CalibrationBundle:
    mode: str
    applied: bool
    global_params: dict[str, float] | None
    regime_params: dict[str, dict[str, float]]
    image_params: dict[str, dict[str, float]]
    pair_records: list[CalibrationPairRecord]
    clip_tracker: ClipTracker


def _gamma_shape_from_contrast(C: float) -> float:
    C = max(float(C), 1e-6)
    return 1.0 / (C * C)


def _clip_with_warning(value: float, lo: float, hi: float, name: str, tracker: ClipTracker | None = None) -> float:
    clipped = float(np.clip(value, lo, hi))
    if not math.isclose(clipped, float(value), rel_tol=0.0, abs_tol=1e-12):
        if tracker is not None:
            tracker.bump(name)
        warnings.warn(
            f"[CALIB] {name} estimate {value:.6g} hit clip bound and was clamped to {clipped:.6g} "
            f"(allowed range [{lo}, {hi}]). Check calibration data quality.",
            RuntimeWarning,
            stacklevel=2,
        )
    return clipped


def estimate_noise_params_from_pair(single01: np.ndarray, avg01: np.ndarray, win: int = 9,
                                    tracker: ClipTracker | None = None) -> tuple[float, float, float]:
    eps = 1e-6
    ratio = np.clip(single01 / (avg01 + eps), 0.0, 4.0).astype(np.float32)
    m, v = _local_mean_var(ratio, win=win)
    valid = np.isfinite(m) & np.isfinite(v) & (avg01 > 0.03)
    if np.any(valid):
        local_cv = np.sqrt(v[valid]) / (m[valid] + eps)
        C = float(np.median(local_cv))
    else:
        C = float(np.sqrt(np.mean(v)) / (np.mean(m) + eps))
    k_hat = _clip_with_warning(_gamma_shape_from_contrast(C), 0.5, 12.0, "k", tracker)

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
    peak_hat = _clip_with_warning(1.0 / a, 1.0, 4096.0, "poisson_peak", tracker)
    sigma_hat = _clip_with_warning(math.sqrt(b), 0.0, 0.25, "gauss_sigma", tracker)
    return k_hat, peak_hat, sigma_hat


def _index_manifest_rows_by_filename(resolved_rows: list["ResolvedImageContext"]) -> dict[str, "ResolvedImageContext"]:
    out: dict[str, ResolvedImageContext] = {}
    for row in resolved_rows:
        fn = Path(row.image_rel_path).name
        if fn in out:
            raise ValueError(
                f"Calibration by-regime/by-image requires unique filenames in manifest; duplicate filename found: {fn}"
            )
        out[fn] = row
    return out


def analyze_real_noise(single_dir: Path, averaged_dir: Path, max_pairs: int = 64,
                       calibration_mode: str = "global",
                       resolved_rows: list["ResolvedImageContext"] | None = None) -> CalibrationBundle:
    import re
    tracker = ClipTracker()
    singles = gather_files(single_dir)
    avgs = gather_files(averaged_dir)
    avg_map = {f.name: f for f in avgs}
    # Also index by stem so that "0040Hz_91.0db.png" matches "0040Hz_91.0db_00.png"
    for f in avgs:
        avg_map[f.stem] = f

    row_by_name = _index_manifest_rows_by_filename(resolved_rows or []) if resolved_rows else {}
    records: list[CalibrationPairRecord] = []
    cnt = 0
    for f in singles:
        af = avg_map.get(f.name)
        if af is None:
            # v3.1 regex fallback: strip trailing _00, _01, etc. from stem
            m = re.match(r"(.*?)_\d+$", f.stem)
            if m:
                af = avg_map.get(m.group(1))
        if af is None:
            continue
        regime = "global"
        key = f.name
        matched = False
        if row_by_name:
            row = row_by_name.get(f.name)
            if row is None:
                continue
            regime = row.regime
            key = row.image_key
            matched = True
        s = imread_uint01(f)
        a = imread_uint01(af)
        if s.shape != a.shape:
            continue
        k, p, sg = estimate_noise_params_from_pair(s, a, tracker=tracker)
        records.append(CalibrationPairRecord(key=key, filename=f.name, regime=regime, k=k, peak=p, sigma=sg, matched=matched))
        cnt += 1
        if cnt >= max_pairs:
            break

    if not records:
        return CalibrationBundle(
            mode=calibration_mode,
            applied=False,
            global_params=None,
            regime_params={},
            image_params={},
            pair_records=[],
            clip_tracker=tracker,
        )

    def _median_params(sub: list[CalibrationPairRecord]) -> dict[str, float]:
        return {
            "k": float(np.median([r.k for r in sub])),
            "peak": float(np.median([r.peak for r in sub])),
            "sigma": float(np.median([r.sigma for r in sub])),
        }

    global_params = _median_params(records)
    regime_params: dict[str, dict[str, float]] = {}
    image_params: dict[str, dict[str, float]] = {}

    if calibration_mode == "by-regime":
        by_regime: dict[str, list[CalibrationPairRecord]] = {}
        for r in records:
            by_regime.setdefault(r.regime or "global", []).append(r)
        regime_params = {regime: _median_params(sub) for regime, sub in by_regime.items()}
    elif calibration_mode == "by-image":
        image_params = {r.key: {"k": r.k, "peak": r.peak, "sigma": r.sigma} for r in records}

    if tracker.total > 0:
        hit_ratio = (tracker.k_hits + tracker.peak_hits + tracker.sigma_hits) / max(1.0, 3.0 * len(records))
        if hit_ratio > 0.2:
            warnings.warn(
                f"[CALIB] Frequent clipping detected in calibration estimates ({hit_ratio:.1%} of parameter estimates clipped).",
                RuntimeWarning,
                stacklevel=2,
            )

    return CalibrationBundle(
        mode=calibration_mode,
        applied=False,
        global_params=global_params,
        regime_params=regime_params,
        image_params=image_params,
        pair_records=records,
        clip_tracker=tracker,
    )


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


def summarize_metrics(rows: list[dict[str, Any]]) -> dict:
    out: dict[str, dict[str, float | int]] = {}
    modes = sorted(set(str(r["mode"]) for r in rows))
    for mode in modes:
        sub = [r for r in rows if str(r["mode"]) == mode]
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
            "edgef1_mean": float(np.mean([float(r["edgef1"]) for r in sub])),
            "edgef1_std": float(np.std([float(r["edgef1"]) for r in sub])),
        }
    return out


def fringe_edge_f1(x: np.ndarray, y: np.ndarray) -> float:
    """Pixel-level binary edge F1 score between clean and noisy/denoised images.

    New in v3.2. Uses the same Sobel edge detector as modal_preservation_index
    but computes precision/recall/F1 on the binary edge maps directly, rather
    than the Hausdorff distance approach used by MPI.
    """
    ex = _edge_map_sobel(x)
    ey = _edge_map_sobel(y)
    tp = float(np.count_nonzero((ex == 1) & (ey == 1)))
    fp = float(np.count_nonzero((ex == 0) & (ey == 1)))
    fn = float(np.count_nonzero((ex == 1) & (ey == 0)))
    denom = 2.0 * tp + fp + fn
    return 1.0 if denom <= 0 else float((2.0 * tp) / denom)


# ------------------------------ config / manifest ------------------------------

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
    blur_mode: str
    blur_piecewise_break_hz: float
    material: str | None
    profile: str | None
    roi_mask: Path | None
    spatial_speckle: bool
    spatial_strength: float
    calib_single: Path | None
    calib_avg: Path | None
    calib_override: bool
    calib_max_pairs: int
    calibration_mode: str
    out_bitdepth: int
    out_format: str
    n_frames: int
    seq_alpha: float
    match: str
    ablate: str | None
    export_metrics: Path | None
    export_summary: Path | None
    manifest: Path | None
    conditioning_mode: str
    rng_mode: str
    regime_column: str
    image_id_column: str
    freq_column: str
    amp_column: str
    roi_mask_column: str
    write_manifest: Path | None
    write_per_image_params: Path | None
    calibration_summary_path: Path | None
    calibration_pairs_path: Path | None
    generator_version: str


@dataclass
class ManifestRow:
    image_key: str
    source_image_path: str
    freq_hz: float | None
    amp_db: float | None
    roi_mask: str | None
    regime: str
    raw: dict[str, Any]


@dataclass
class ResolvedImageContext:
    image_key: str
    image_path: Path
    image_rel_path: str
    freq_hz: float | None
    amp_db: float | None
    roi_mask_path: Path | None
    regime: str
    source_manifest_row: dict[str, Any] | None


@dataclass
class ResolvedNoiseParams:
    speckle_k: float
    poisson_peak: float
    gauss_sigma: float
    blur_length_px: int | None
    calibration_applied: bool
    calibration_source: str


# ------------------------------ helpers ------------------------------

def _script_sha256() -> str:
    """SHA256 of this script file (informational, not enforcement)."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return "unavailable"


def _dependency_versions() -> dict[str, str | None]:
    """Collect installed versions of key dependencies."""
    versions: dict[str, str | None] = {}
    versions["numpy"] = np.__version__
    try:
        from PIL import __version__ as pil_ver
        versions["pillow"] = pil_ver
    except Exception:
        versions["pillow"] = None
    try:
        versions["opencv"] = _cv2.__version__ if _cv2 is not None else None
    except Exception:
        versions["opencv"] = None
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = None
    return versions


def _stable_seed(base_seed: int, *parts: str) -> int:
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for part in parts:
        h.update(b"||")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little", signed=False) % (2**63 - 1)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _parse_optional_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    return float(x)


def _load_json_or_csv_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "rows" in payload:
            payload = payload["rows"]
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list of objects or contain a top-level 'rows' list")
        rows = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("Each JSON manifest entry must be an object")
            rows.append(item)
        return rows

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def _canonical_manifest_key(raw_path: str, input_root: Path) -> str:
    p = Path(str(raw_path))
    if p.is_absolute():
        try:
            return p.resolve().relative_to(input_root.resolve()).as_posix()
        except Exception as exc:
            raise ValueError(f"Manifest image_path points outside input root: {raw_path}") from exc
    return Path(raw_path).as_posix().lstrip("./")


def load_manifest(path: Path, input_root: Path, args: Args) -> dict[str, ManifestRow]:
    rows = _load_json_or_csv_rows(path)
    out: dict[str, ManifestRow] = {}
    required = {args.image_id_column}
    if args.conditioning_mode == "manifest":
        required.update({args.freq_column, args.amp_column})
    seen_keys: set[str] = set()
    missing_required_rows = 0
    for idx, row in enumerate(rows, start=1):
        missing = [col for col in required if col not in row]
        if missing:
            missing_required_rows += 1
            raise ValueError(f"Manifest row {idx} missing required columns: {missing}")
        key = _canonical_manifest_key(str(row[args.image_id_column]), input_root)
        if key in seen_keys:
            raise ValueError(f"Duplicate manifest image key detected: {key}")
        seen_keys.add(key)
        regime = str(row.get(args.regime_column, "global") or "global")
        mr = ManifestRow(
            image_key=key,
            source_image_path=key,
            freq_hz=_parse_optional_float(row.get(args.freq_column)),
            amp_db=_parse_optional_float(row.get(args.amp_column)),
            roi_mask=str(row.get(args.roi_mask_column)).strip() if row.get(args.roi_mask_column) not in (None, "") else None,
            regime=regime,
            raw=dict(row),
        )
        out[key] = mr

    # strict compatibility checks against actual source files
    files = gather_files(input_root)
    file_keys = {normalize_relpath(f, input_root): f for f in files}
    missing_rows = [k for k in file_keys if k not in out]
    extra_rows = [k for k in out if k not in file_keys]
    if missing_rows:
        preview = ", ".join(missing_rows[:5])
        raise ValueError(f"Manifest is incomplete: {len(missing_rows)} input files have no manifest row. Examples: {preview}")
    if extra_rows:
        preview = ", ".join(extra_rows[:5])
        raise ValueError(f"Manifest contains {len(extra_rows)} rows not present under input root. Examples: {preview}")
    return out


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
    if args.manifest is not None and not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.spatial_speckle and args.roi_mask is None and args.conditioning_mode == "global":
        raise ValueError("--spatial-speckle requires --roi-mask in global conditioning mode")
    if args.conditioning_mode == "manifest" and args.manifest is None:
        raise ValueError("--conditioning-mode manifest requires --manifest")
    if args.calibration_mode in {"by-regime", "by-image"} and args.manifest is None:
        raise ValueError(f"--calibration-mode {args.calibration_mode} requires --manifest")
    if args.calibration_mode in {"by-regime", "by-image"} and (args.calib_single is None or args.calib_avg is None):
        raise ValueError(f"--calibration-mode {args.calibration_mode} requires both --calib-single and --calib-avg")


def warn_if_out_of_range(freq_hz: float | None, amp_db: float | None, key: str) -> None:
    if freq_hz is not None and not (0.0 <= freq_hz <= 5000.0):
        warnings.warn(f"[{key}] freq_hz={freq_hz} looks out of expected range [0, 5000].", RuntimeWarning, stacklevel=2)
    if amp_db is not None and not (-40.0 <= amp_db <= 200.0):
        warnings.warn(f"[{key}] amp_db={amp_db} looks out of expected range [-40, 200].", RuntimeWarning, stacklevel=2)


def resolve_contexts(args: Args, files: list[Path], manifest_rows: dict[str, ManifestRow] | None) -> list[ResolvedImageContext]:
    out: list[ResolvedImageContext] = []
    for f in files:
        rel = normalize_relpath(f, args.input)
        row = None if manifest_rows is None else manifest_rows.get(rel)
        if row is None:
            out.append(
                ResolvedImageContext(
                    image_key=rel,
                    image_path=f,
                    image_rel_path=rel,
                    freq_hz=args.freq_hz,
                    amp_db=args.amp_db,
                    roi_mask_path=args.roi_mask,
                    regime="global",
                    source_manifest_row=None,
                )
            )
            warn_if_out_of_range(args.freq_hz, args.amp_db, rel)
            continue

        roi_mask_path = args.roi_mask
        if args.conditioning_mode == "manifest":
            roi_mask_path = Path(row.roi_mask) if row.roi_mask else args.roi_mask
            if roi_mask_path is not None and not roi_mask_path.is_absolute():
                roi_mask_path = (
                    (args.input / roi_mask_path).resolve()
                    if (args.input / roi_mask_path).exists()
                    else (args.manifest.parent / roi_mask_path).resolve()
                )
            if roi_mask_path is not None and not roi_mask_path.exists():
                raise FileNotFoundError(f"Manifest-specified roi_mask not found for {rel}: {roi_mask_path}")
            freq_hz = row.freq_hz
            amp_db = row.amp_db
        else:
            freq_hz = args.freq_hz
            amp_db = args.amp_db

        out.append(
            ResolvedImageContext(
                image_key=rel,
                image_path=f,
                image_rel_path=rel,
                freq_hz=freq_hz,
                amp_db=amp_db,
                roi_mask_path=roi_mask_path,
                regime=row.regime or "global",
                source_manifest_row=row.raw,
            )
        )
        warn_if_out_of_range(freq_hz, amp_db, rel)
    return out


def resolve_calibration_for_image(ctx: ResolvedImageContext, args: Args,
                                  cal_bundle: CalibrationBundle | None) -> tuple[float, float, float, bool, str]:
    if cal_bundle is None or cal_bundle.global_params is None or not args.calib_override:
        return args.speckle_k, args.poisson_peak, args.gauss_sigma, False, "cli_or_profile_defaults"

    if args.calibration_mode == "by-regime":
        params = cal_bundle.regime_params.get(ctx.regime)
        if params is None:
            raise ValueError(f"No regime calibration available for image {ctx.image_rel_path} with regime '{ctx.regime}'")
        return float(params["k"]), float(params["peak"]), float(params["sigma"]), True, f"calibration:regime:{ctx.regime}"

    if args.calibration_mode == "by-image":
        params = cal_bundle.image_params.get(ctx.image_key)
        if params is None:
            # fallback by filename if exact key unavailable and unique mapping exists in pair records
            by_filename = {r.filename: r for r in cal_bundle.pair_records}
            rec = by_filename.get(Path(ctx.image_rel_path).name)
            if rec is None:
                raise ValueError(f"No by-image calibration available for image {ctx.image_rel_path}")
            params = {"k": rec.k, "peak": rec.peak, "sigma": rec.sigma}
        return float(params["k"]), float(params["peak"]), float(params["sigma"]), True, f"calibration:image:{ctx.image_key}"

    params = cal_bundle.global_params
    return float(params["k"]), float(params["peak"]), float(params["sigma"]), True, "calibration:global"


def make_roi_mask_cache() -> dict[str, np.ndarray]:
    return {}


def load_roi_mask_cached(cache: dict[str, np.ndarray], path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    key = str(path.resolve())
    if key not in cache:
        cache[key] = (imread_uint01(path) > 0.5).astype(np.uint8)
    return cache[key]


def add_noise_chain(clean01: np.ndarray, rng: np.random.Generator, args: Args,
                    freq_hz: float | None, amp_db: float | None,
                    roi_mask_arr: np.ndarray | None,
                    resolved_noise: ResolvedNoiseParams) -> np.ndarray:
    y = clean01.copy()
    if freq_hz is not None and amp_db is not None:
        y, _ = add_frequency_aware_blur(y, freq_hz, amp_db, args.blur_scale, args.blur_mode, rng)

    if args.ablate != "no-speckle":
        k_field: np.ndarray | float = resolved_noise.speckle_k
        if args.spatial_speckle and roi_mask_arr is not None:
            k_field = build_spatial_k_map(resolved_noise.speckle_k, roi_mask_arr, args.spatial_strength, y.shape)
        y = add_speckle_multiplicative(y, k_field, rng)

    if args.ablate != "no-poisson":
        y = add_poisson_shot(y, resolved_noise.poisson_peak, rng)

    if args.ablate != "no-gaussian":
        y = add_gaussian(y, resolved_noise.gauss_sigma, rng)

    y = apply_match_stats(clean01, y, args.match)
    return np.clip(y, 0.0, 1.0)


# ------------------------------ main ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="v3.2 methodology upgrade of the calibrated ESPI synthetic noise generator")
    ap.add_argument("--input", type=Path, required=True, help="Folder with clean images")
    ap.add_argument("--output", type=Path, required=True, help="Output folder for pseudo-noisy images")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--speckle-k", type=float, default=3.0)
    ap.add_argument("--poisson-peak", type=float, default=60.0)
    ap.add_argument("--gauss-sigma", type=float, default=0.01)
    ap.add_argument("--freq-hz", type=float, default=None, help="Global excitation frequency (Hz)")
    ap.add_argument("--amp-db", type=float, default=None, help="Global excitation amplitude (dB)")
    ap.add_argument("--blur-scale", type=float, default=1.0, help="Global multiplier for blur length")
    ap.add_argument("--blur-mode", type=str, default="linear", choices=["linear", "sqrt", "piecewise"],
                    help="Blur scaling model. 'linear' is v3.1-compatible default. 'sqrt' and 'piecewise' are forward-compatible additions (planned for v3.3 evaluation).")
    ap.add_argument("--blur-piecewise-break-hz", type=float, default=680.0)
    ap.add_argument("--material", type=str, default=None, choices=list(MATERIAL_PROFILES.keys()))
    ap.add_argument("--profile", type=str, default=None, choices=list(PROFILE_PRESETS.keys()))
    ap.add_argument("--roi-mask", type=Path, default=None, help="Global ROI mask image (white=ROI)")
    ap.add_argument("--spatial-speckle", action="store_true", help="Enable spatially varying speckle")
    ap.add_argument("--spatial-strength", type=float, default=0.3)
    ap.add_argument("--calib-single", type=Path, default=None, help="Folder with real single-shot images")
    ap.add_argument("--calib-avg", type=Path, default=None, help="Folder with averaged reference images")
    ap.add_argument("--calib-override", action="store_true", help="Apply estimated calibration params to generation")
    ap.add_argument("--calib-max-pairs", type=int, default=64)
    ap.add_argument("--calibration-mode", type=str, default="global", choices=["global", "by-regime", "by-image"])
    ap.add_argument("--out-bitdepth", "--bitdepth", dest="out_bitdepth", type=int, default=8, choices=[8, 16])
    ap.add_argument("--out-format", type=str, default="png", choices=["png", "tiff"])
    ap.add_argument("--n-frames", type=int, default=0, help="If >1, generate a temporal sequence per image")
    ap.add_argument("--seq-alpha", type=float, default=0.15,
                    help="Temporal update factor: new = (1-alpha)*prev + alpha*current")
    ap.add_argument("--match", type=str, default="none", choices=["none", "mean", "meanstd"],
                    help="Optional post-noise per-image statistics matching")
    ap.add_argument("--ablate", type=str, default=None,
                    choices=["no-speckle", "no-poisson", "no-gaussian", "full", "all"])
    ap.add_argument("--export-metrics", type=Path, default=None,
                    help="(backward-compat) CSV path for per-image metrics. Superseded by per_image_params.csv but kept for v3.1 CLI compatibility.")
    ap.add_argument("--export-summary", type=Path, default=None,
                    help="(backward-compat) JSON path for aggregated metrics summary. Superseded by run_manifest_final.json but kept for v3.1 CLI compatibility.")

    ap.add_argument("--manifest", type=Path, default=None, help="CSV/JSON manifest with per-image conditioning metadata")
    ap.add_argument("--conditioning-mode", type=str, default="global", choices=["global", "manifest"])
    ap.add_argument("--rng-mode", type=str, default="legacy", choices=["legacy", "per_image"])
    ap.add_argument("--regime-column", type=str, default="regime")
    ap.add_argument("--image-id-column", type=str, default="image_path")
    ap.add_argument("--freq-column", type=str, default="freq_hz")
    ap.add_argument("--amp-column", type=str, default="amp_db")
    ap.add_argument("--roi-mask-column", type=str, default="roi_mask")
    ap.add_argument("--write-manifest", type=Path, default=None)
    ap.add_argument("--write-per-image-params", type=Path, default=None)
    ap.add_argument("--calibration-summary-path", type=Path, default=None)
    ap.add_argument("--calibration-pairs-path", type=Path, default=None)
    ap.add_argument("--generator-version", type=str, default="v3.2")

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
        blur_mode=args_ns.blur_mode,
        blur_piecewise_break_hz=args_ns.blur_piecewise_break_hz,
        material=args_ns.material,
        profile=args_ns.profile,
        roi_mask=args_ns.roi_mask,
        spatial_speckle=args_ns.spatial_speckle,
        spatial_strength=args_ns.spatial_strength,
        calib_single=args_ns.calib_single,
        calib_avg=args_ns.calib_avg,
        calib_override=args_ns.calib_override,
        calib_max_pairs=args_ns.calib_max_pairs,
        calibration_mode=args_ns.calibration_mode,
        out_bitdepth=args_ns.out_bitdepth,
        out_format=args_ns.out_format,
        n_frames=args_ns.n_frames,
        seq_alpha=args_ns.seq_alpha,
        match=args_ns.match,
        ablate=args_ns.ablate,
        export_metrics=args_ns.export_metrics,
        export_summary=args_ns.export_summary,
        manifest=args_ns.manifest,
        conditioning_mode=args_ns.conditioning_mode,
        rng_mode=args_ns.rng_mode,
        regime_column=args_ns.regime_column,
        image_id_column=args_ns.image_id_column,
        freq_column=args_ns.freq_column,
        amp_column=args_ns.amp_column,
        roi_mask_column=args_ns.roi_mask_column,
        write_manifest=args_ns.write_manifest,
        write_per_image_params=args_ns.write_per_image_params,
        calibration_summary_path=args_ns.calibration_summary_path,
        calibration_pairs_path=args_ns.calibration_pairs_path,
        generator_version=args_ns.generator_version,
    )

    validate_args(args)
    args = resolve_noise_defaults(args, ap, args_ns)

    ensure_dir(args.output)
    run_manifest_start_path = args.output / "run_manifest_start.json"
    run_manifest_final_path = args.output / "run_manifest_final.json"
    write_manifest_path = args.write_manifest
    write_per_image_params = args.write_per_image_params or (args.output / "per_image_params.csv")
    calibration_summary_path = args.calibration_summary_path or (args.output / "calibration_summary.json")
    calibration_pairs_path = args.calibration_pairs_path or (args.output / "calibration_pairs.csv")

    files = gather_files(args.input)
    if not files:
        print(f"No images found under {args.input}", file=sys.stderr)
        sys.exit(1)

    manifest_rows = None
    if args.manifest is not None:
        manifest_rows = load_manifest(args.manifest, args.input, args)

    resolved_contexts = resolve_contexts(args, files, manifest_rows)

    cal_bundle: CalibrationBundle | None = None
    if args.calib_single and args.calib_avg:
        cal_bundle = analyze_real_noise(
            args.calib_single,
            args.calib_avg,
            max_pairs=args.calib_max_pairs,
            calibration_mode=args.calibration_mode,
            resolved_rows=resolved_contexts if args.conditioning_mode == "manifest" else None,
        )
        cal_bundle.applied = bool(args.calib_override and cal_bundle.global_params is not None)
        if cal_bundle.global_params is not None:
            print(
                f"[CALIB] global estimate k={cal_bundle.global_params['k']:.2f} "
                f"peak={cal_bundle.global_params['peak']:.1f} sigma={cal_bundle.global_params['sigma']:.4f}"
            )

    # write calibration artifacts early
    calibration_summary = {
        "version": args.generator_version,
        "calibration_mode": args.calibration_mode,
        "calibration_applied": bool(cal_bundle.applied) if cal_bundle else False,
        "calibration_sources": {
            "single_dir": str(args.calib_single) if args.calib_single else None,
            "avg_dir": str(args.calib_avg) if args.calib_avg else None,
            "max_pairs": args.calib_max_pairs,
        },
        "global_params": cal_bundle.global_params if cal_bundle else None,
        "regime_params": cal_bundle.regime_params if cal_bundle else {},
        "image_params_count": len(cal_bundle.image_params) if cal_bundle else 0,
        "pair_count": len(cal_bundle.pair_records) if cal_bundle else 0,
        "clip_tracker": asdict(cal_bundle.clip_tracker) if cal_bundle else asdict(ClipTracker()),
    }
    ensure_dir(calibration_summary_path.parent)
    calibration_summary_path.write_text(json.dumps(_jsonify(calibration_summary), indent=2), encoding="utf-8")
    with calibration_pairs_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["key", "filename", "regime", "k", "peak", "sigma", "matched"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in (cal_bundle.pair_records if cal_bundle else []):
            writer.writerow(asdict(rec))

    roi_cache = make_roi_mask_cache()
    metrics_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    total = len(resolved_contexts)
    t0 = time.time()
    timestamp_start = time.strftime("%Y-%m-%dT%H:%M:%S")
    shared_rng = np.random.default_rng(args.seed) if args.rng_mode == "legacy" else None

    ablate_modes: list[str | None]
    if args.ablate == "all":
        ablate_modes = [None, "no-speckle", "no-poisson", "no-gaussian"]
    elif args.ablate in {"no-speckle", "no-poisson", "no-gaussian"}:
        ablate_modes = [args.ablate]
    else:
        ablate_modes = [None]

    source_file_list = [ctx.image_rel_path for ctx in resolved_contexts]
    run_manifest_start = {
        "generator_version": args.generator_version,
        "timestamp_start": timestamp_start,
        "completed": False,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "exact_cli": sys.argv,
        "args": _jsonify(vars(args_ns)),
        "resolved_global_args": _jsonify(asdict(args)),
        "seed": args.seed,
        "conditioning_mode": args.conditioning_mode,
        "rng_mode": args.rng_mode,
        "calibration_mode": args.calibration_mode,
        "manifest_path": str(args.manifest) if args.manifest else None,
        "source_root": str(args.input),
        "output_root": str(args.output),
        "source_file_count": len(source_file_list),
        "source_files": source_file_list,
        "calibration_sources": {
            "single_dir": str(args.calib_single) if args.calib_single else None,
            "avg_dir": str(args.calib_avg) if args.calib_avg else None,
            "summary_json": str(calibration_summary_path),
            "pairs_csv": str(calibration_pairs_path),
        },
        "backward_compatibility_mode": args.conditioning_mode == "global" and args.rng_mode == "legacy",
        "run_status": "running",
    }
    run_manifest_start_path.write_text(json.dumps(_jsonify(run_manifest_start), indent=2), encoding="utf-8")

    for idx, ctx in enumerate(resolved_contexts, start=1):
        clean = imread_uint01(ctx.image_path)
        base = ctx.image_path.stem
        roi_mask_arr = load_roi_mask_cached(roi_cache, ctx.roi_mask_path)

        sk, pp, gs, cal_applied, cal_source = resolve_calibration_for_image(ctx, args, cal_bundle)
        blur_length_px = None
        if ctx.freq_hz is not None and ctx.amp_db is not None:
            blur_length_px = resolve_blur_length_px(
                ctx.freq_hz, ctx.amp_db, args.blur_scale, args.blur_mode, args.blur_piecewise_break_hz
            )
        resolved_noise = ResolvedNoiseParams(
            speckle_k=sk,
            poisson_peak=pp,
            gauss_sigma=gs,
            blur_length_px=blur_length_px,
            calibration_applied=cal_applied,
            calibration_source=cal_source,
        )

        for abl in ablate_modes:
            mode_tag = "full" if abl is None else abl
            out_dir = args.output if args.ablate != "all" else (args.output / mode_tag)
            ensure_dir(out_dir)
            run_args = replace(args, ablate=abl)
            rng_subseed = args.seed if args.rng_mode == "legacy" else _stable_seed(args.seed, ctx.image_key)
            rng = shared_rng if args.rng_mode == "legacy" else np.random.default_rng(rng_subseed)

            if args.n_frames > 1:
                prev = None
                seq_dir = out_dir / f"{base}_seq"
                ensure_dir(seq_dir)
                for t in range(args.n_frames):
                    y = add_noise_chain(clean, rng, run_args, ctx.freq_hz, ctx.amp_db, roi_mask_arr, resolved_noise)
                    if prev is not None:
                        alpha = args.seq_alpha
                        y = np.clip((1.0 - alpha) * prev + alpha * y, 0.0, 1.0)
                    prev = y
                    out_path = imsave_uint01(y, seq_dir / f"{base}_t{t:03d}", bitdepth=args.out_bitdepth, out_format=args.out_format)
                    metrics_rows.append({
                        "source": str(ctx.image_path),
                        "file": str(out_path),
                        "mode": mode_tag,
                        "match": args.match,
                        "psnr": f"{psnr(clean, y):.6f}",
                        "ssim": f"{ssim(clean, y):.6f}",
                        "mpi": f"{modal_preservation_index(clean, y):.6f}",
                        "pcs": f"{phase_coherence_score(y):.6f}",
                        "edgef1": f"{fringe_edge_f1(clean, y):.6f}",
                        "speckle_k": f"{resolved_noise.speckle_k:.6f}",
                        "poisson_peak": f"{resolved_noise.poisson_peak:.6f}",
                        "gauss_sigma": f"{resolved_noise.gauss_sigma:.6f}",
                    })
                    param_rows.append({
                        "image_id": ctx.image_key,
                        "source_path": ctx.image_rel_path,
                        "output_path": normalize_relpath(out_path, args.output),
                        "mode": mode_tag,
                        "frame_idx": t,
                        "conditioning_mode": args.conditioning_mode,
                        "rng_mode": args.rng_mode,
                        "rng_subseed": rng_subseed,
                        "freq_hz": ctx.freq_hz,
                        "amp_db": ctx.amp_db,
                        "blur_length_px": resolved_noise.blur_length_px,
                        "speckle_k": resolved_noise.speckle_k,
                        "poisson_peak": resolved_noise.poisson_peak,
                        "gauss_sigma": resolved_noise.gauss_sigma,
                        "match": args.match,
                        "roi_mask_path": str(ctx.roi_mask_path) if ctx.roi_mask_path else "",
                        "regime": ctx.regime,
                        "group": str((ctx.source_manifest_row or {}).get("group", "")),
                        "psnr": f"{psnr(clean, y):.6f}",
                        "ssim": f"{ssim(clean, y):.6f}",
                        "mpi": f"{modal_preservation_index(clean, y):.6f}",
                        "pcs": f"{phase_coherence_score(y):.6f}",
                        "edgef1": f"{fringe_edge_f1(clean, y):.6f}",
                    })
            else:
                noisy = add_noise_chain(clean, rng, run_args, ctx.freq_hz, ctx.amp_db, roi_mask_arr, resolved_noise)
                out_path = imsave_uint01(noisy, out_dir / f"{base}_{mode_tag}", bitdepth=args.out_bitdepth, out_format=args.out_format)
                metrics_rows.append({
                    "source": str(ctx.image_path),
                    "file": str(out_path),
                    "mode": mode_tag,
                    "match": args.match,
                    "psnr": f"{psnr(clean, noisy):.6f}",
                    "ssim": f"{ssim(clean, noisy):.6f}",
                    "mpi": f"{modal_preservation_index(clean, noisy):.6f}",
                    "pcs": f"{phase_coherence_score(noisy):.6f}",
                    "edgef1": f"{fringe_edge_f1(clean, noisy):.6f}",
                    "speckle_k": f"{resolved_noise.speckle_k:.6f}",
                    "poisson_peak": f"{resolved_noise.poisson_peak:.6f}",
                    "gauss_sigma": f"{resolved_noise.gauss_sigma:.6f}",
                })
                param_rows.append({
                    "image_id": ctx.image_key,
                    "source_path": ctx.image_rel_path,
                    "output_path": normalize_relpath(out_path, args.output),
                    "mode": mode_tag,
                    "frame_idx": 0,
                    "conditioning_mode": args.conditioning_mode,
                    "rng_mode": args.rng_mode,
                    "rng_subseed": rng_subseed,
                    "freq_hz": ctx.freq_hz,
                    "amp_db": ctx.amp_db,
                    "blur_length_px": resolved_noise.blur_length_px,
                    "speckle_k": resolved_noise.speckle_k,
                    "poisson_peak": resolved_noise.poisson_peak,
                    "gauss_sigma": resolved_noise.gauss_sigma,
                    "match": args.match,
                    "roi_mask_path": str(ctx.roi_mask_path) if ctx.roi_mask_path else "",
                    "regime": ctx.regime,
                    "group": str((ctx.source_manifest_row or {}).get("group", "")),
                    "psnr": f"{psnr(clean, noisy):.6f}",
                    "ssim": f"{ssim(clean, noisy):.6f}",
                    "mpi": f"{modal_preservation_index(clean, noisy):.6f}",
                    "pcs": f"{phase_coherence_score(noisy):.6f}",
                    "edgef1": f"{fringe_edge_f1(clean, noisy):.6f}",
                })

        if idx % 5 == 0 or idx == total:
            dt = time.time() - t0
            print(f"[{idx}/{total}] processed in {dt:.1f}s")

    if args.export_metrics is None:
        export_metrics = args.output / "metrics.csv"
    else:
        export_metrics = args.export_metrics
    if args.export_summary is None:
        export_summary = args.output / "summary.json"
    else:
        export_summary = args.export_summary

    if metrics_rows:
        ensure_dir(export_metrics.parent)
        with export_metrics.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        ensure_dir(export_summary.parent)
        summary = summarize_metrics(metrics_rows)
        export_summary.write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")

    if param_rows:
        ensure_dir(write_per_image_params.parent)
        with write_per_image_params.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(param_rows[0].keys()))
            writer.writeheader()
            writer.writerows(param_rows)

    elapsed_seconds = time.time() - t0
    run_manifest_final = {
        "generator_version": args.generator_version,
        "timestamp_start": timestamp_start,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": elapsed_seconds,
        "completed": True,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _script_sha256(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "exact_cli": sys.argv,
        "args": _jsonify(vars(args_ns)),
        "resolved_global_args": _jsonify(asdict(args)),
        "seed": args.seed,
        "conditioning_mode": args.conditioning_mode,
        "rng_mode": args.rng_mode,
        "calibration_mode": args.calibration_mode,
        "calibration_applied": bool(cal_bundle.applied) if cal_bundle else False,
        "manifest_path": str(args.manifest) if args.manifest else None,
        "source_root": str(args.input),
        "output_root": str(args.output),
        "source_file_count": len(source_file_list),
        "source_files": source_file_list,
        "output_file_count": len(param_rows),
        "calibration_sources": {
            "single_dir": str(args.calib_single) if args.calib_single else None,
            "avg_dir": str(args.calib_avg) if args.calib_avg else None,
            "summary_json": str(calibration_summary_path),
            "pairs_csv": str(calibration_pairs_path),
        },
        "exports": {
            "run_manifest_start": str(run_manifest_start_path),
            "run_manifest_final": str(run_manifest_final_path),
            "per_image_params": str(write_per_image_params),
            "metrics_csv": str(export_metrics),
            "summary_json": str(export_summary),
        },
        "backward_compatibility_mode": args.conditioning_mode == "global" and args.rng_mode == "legacy",
        "run_status": "completed",
    }
    run_manifest_final_path.write_text(json.dumps(_jsonify(run_manifest_final), indent=2), encoding="utf-8")
    if write_manifest_path is not None:
        ensure_dir(write_manifest_path.parent)
        write_manifest_path.write_text(json.dumps(_jsonify(run_manifest_final), indent=2), encoding="utf-8")

    print(f"[PROVENANCE] Wrote {run_manifest_start_path}")
    print(f"[PROVENANCE] Wrote {run_manifest_final_path}")
    if write_manifest_path is not None:
        print(f"[PROVENANCE] Wrote {write_manifest_path}")
    print(f"[PROVENANCE] Wrote {write_per_image_params}")
    print(f"[CALIB] Wrote {calibration_summary_path}")
    print(f"[CALIB] Wrote {calibration_pairs_path}")
    print(f"[METRICS] Wrote {export_metrics}")
    print(f"[SUMMARY] Wrote {export_summary}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
