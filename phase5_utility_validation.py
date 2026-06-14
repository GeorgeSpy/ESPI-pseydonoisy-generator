from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# --- portable path configuration (added for public repo) ---
# Override any of these with environment variables; defaults are repo-relative.
#   ESPI_REPO  : folder holding the generator scripts (default: this file's folder)
#   ESPI_DATA  : dataset root with wood_Averaged/ and wood_real_A/ (default: <repo>/data)
#   ESPI_OUT   : output root for validation artifacts (default: <repo>/_validation_out)
#   ESPI_V4_CKPT : path to the frozen V4 DnCNN-Lite-ECA checkpoint (phase5 only)
import os as _os
_THIS_FILE = Path(__file__).resolve()
_REPO = Path(_os.environ.get("ESPI_REPO", _THIS_FILE.parent))
_DATA = Path(_os.environ.get("ESPI_DATA", _REPO / "data"))
_OUT  = Path(_os.environ.get("ESPI_OUT", _REPO / "_validation_out"))
# --- end path configuration ---
from typing import Any

import numpy as np
import torch


PHASE4_ROOT = _OUT / "v3_2_phase4_calibration_modes"
PHASE5_ROOT = _OUT / "v3_2_phase5_utility_validation"
V32_PATH = _REPO / "make_pseudo_noisy_plus_v3_2.py"
V4_CKPT = Path(_os.environ.get("ESPI_V4_CKPT", str(_REPO / "checkpoints" / "v4_canonical_best.pth")))


class ValidationStop(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeEvalRow:
    image_key: str
    regime: str
    noisy_path: Path
    clean_path: Path
    psnr: float
    ssim: float
    edgef1: float


def run_command(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValidationStop(
            "Run failed:\n"
            f"command: {subprocess.list2cmdline(cmd)}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def prepare_root() -> None:
    if PHASE5_ROOT.exists():
        raise ValidationStop(f"Phase 5 output root already exists, refusing to mix artifacts: {PHASE5_ROOT}")
    for subdir in [
        PHASE5_ROOT / "inputs",
        PHASE5_ROOT / "run_global",
        PHASE5_ROOT / "run_by_regime",
        PHASE5_ROOT / "probe_eval",
        PHASE5_ROOT / "comparison",
        PHASE5_ROOT / "reports",
    ]:
        subdir.mkdir(parents=True, exist_ok=False)


def copy_phase4_inputs() -> tuple[Path, Path]:
    phase4_inputs = PHASE4_ROOT / "inputs"
    phase5_inputs = PHASE5_ROOT / "inputs"
    if not phase4_inputs.exists():
        raise ValidationStop(f"Missing Phase 4 inputs directory: {phase4_inputs}")

    for child in phase4_inputs.iterdir():
        target = phase5_inputs / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)

    subset_dir = phase5_inputs / "subset_clean_aligned"
    manifest_path = phase5_inputs / "phase4_input_manifest.csv"
    subset_csv = phase5_inputs / "phase4_subset_sources.csv"
    if not subset_dir.exists() or not manifest_path.exists() or not subset_csv.exists():
        raise ValidationStop("Failed to copy frozen Phase 4 subset/manifest into Phase 5 inputs")
    return subset_dir, manifest_path


def build_generator_command(phase4_run_manifest: Path, input_root: Path, manifest_path: Path, output_root: Path, calibration_mode: str) -> list[str]:
    payload = load_json(phase4_run_manifest)
    args = payload.get("resolved_global_args")
    if not isinstance(args, dict):
        raise ValidationStop(f"Missing resolved_global_args in {phase4_run_manifest}")

    cmd = [
        sys.executable,
        str(V32_PATH),
        "--input",
        str(input_root),
        "--output",
        str(output_root),
        "--manifest",
        str(manifest_path),
        "--conditioning-mode",
        "manifest",
        "--rng-mode",
        "per_image",
        "--calibration-mode",
        calibration_mode,
        "--seed",
        str(args["seed"]),
        "--speckle-k",
        str(args["speckle_k"]),
        "--poisson-peak",
        str(args["poisson_peak"]),
        "--gauss-sigma",
        str(args["gauss_sigma"]),
        "--match",
        str(args["match"]),
        "--out-bitdepth",
        str(args["out_bitdepth"]),
        "--out-format",
        str(args["out_format"]),
        "--blur-mode",
        str(args["blur_mode"]),
        "--blur-scale",
        str(args["blur_scale"]),
        "--n-frames",
        str(args["n_frames"]),
        "--seq-alpha",
        str(args["seq_alpha"]),
        "--calib-single",
        str(args["calib_single"]),
        "--calib-avg",
        str(args["calib_avg"]),
        "--calib-override",
        "--calib-max-pairs",
        str(args["calib_max_pairs"]),
        "--export-metrics",
        str(output_root / "metrics.csv"),
        "--export-summary",
        str(output_root / "summary.json"),
    ]
    return cmd


def verify_run_outputs(run_root: Path, label: str) -> tuple[Path, Path, Path]:
    run_manifest = run_root / "run_manifest_final.json"
    params_csv = run_root / "per_image_params.csv"
    summary_json = run_root / "calibration_summary.json"
    missing = [p for p in [run_manifest, params_csv, summary_json] if not p.exists()]
    if missing:
        raise ValidationStop(f"{label} run missing required outputs: {missing}")
    return run_manifest, params_csv, summary_json


def map_by_key(rows: list[dict[str, str]], key_field: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[key_field]
        if key in out:
            raise ValidationStop(f"Duplicate key {key} in CSV ledger")
        out[key] = row
    return out


def write_generator_side_comparison(global_rows: list[dict[str, str]], by_rows: list[dict[str, str]]) -> tuple[Path, bool]:
    out_path = PHASE5_ROOT / "comparison" / "generator_side_comparison.csv"
    g_map = map_by_key(global_rows, "image_id")
    b_map = map_by_key(by_rows, "image_id")
    if set(g_map) != set(b_map):
        raise ValidationStop("Generator-side comparison failed: image_id sets differ between runs")

    measurable = False
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_key",
                "regime",
                "global_k",
                "byregime_k",
                "global_peak",
                "byregime_peak",
                "global_sigma",
                "byregime_sigma",
                "global_psnr",
                "byregime_psnr",
                "global_ssim",
                "byregime_ssim",
                "global_edgef1",
                "byregime_edgef1",
            ],
        )
        writer.writeheader()
        for key in sorted(g_map):
            g = g_map[key]
            b = b_map[key]
            g_psnr = float(g["psnr"])
            b_psnr = float(b["psnr"])
            g_ssim = float(g["ssim"])
            b_ssim = float(b["ssim"])
            g_edge = float(g["edgef1"])
            b_edge = float(b["edgef1"])
            if any(
                [
                    float(g["speckle_k"]) != float(b["speckle_k"]),
                    float(g["poisson_peak"]) != float(b["poisson_peak"]),
                    float(g["gauss_sigma"]) != float(b["gauss_sigma"]),
                    abs(g_psnr - b_psnr) > 1e-9,
                    abs(g_ssim - b_ssim) > 1e-9,
                    abs(g_edge - b_edge) > 1e-9,
                ]
            ):
                measurable = True
            writer.writerow(
                {
                    "image_key": key,
                    "regime": g["regime"],
                    "global_k": g["speckle_k"],
                    "byregime_k": b["speckle_k"],
                    "global_peak": g["poisson_peak"],
                    "byregime_peak": b["poisson_peak"],
                    "global_sigma": g["gauss_sigma"],
                    "byregime_sigma": b["gauss_sigma"],
                    "global_psnr": g["psnr"],
                    "byregime_psnr": b["psnr"],
                    "global_ssim": g["ssim"],
                    "byregime_ssim": b["ssim"],
                    "global_edgef1": g["edgef1"],
                    "byregime_edgef1": b["edgef1"],
                }
            )
    return out_path, measurable


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(vals: list[float]) -> float:
    return float(np.std(vals)) if vals else float("nan")


def load_v4_probe():
    from espi_dncnn_lite_v5 import DnCNNLiteECA, DnCNNLiteECAConfig  # type: ignore

    ckpt = torch.load(V4_CKPT, map_location="cpu")
    cfg = ckpt["config"]
    if not isinstance(cfg, DnCNNLiteECAConfig):
        raise ValidationStop(f"Unexpected config type in V4 checkpoint: {type(cfg)}")
    model = DnCNNLiteECA(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt.get("args", {}), cfg


def evaluate_probe_branch(run_root: Path, params_rows: list[dict[str, str]], clean_root: Path, label: str) -> tuple[list[ProbeEvalRow], Path]:
    from espi_dncnn_lite_v5 import denoise_tiled, fringe_edge_f1, imread_uint, psnr, set_seed, ssim_metric  # type: ignore

    probe_dir = PHASE5_ROOT / "probe_eval"
    probe_dir.mkdir(parents=True, exist_ok=True)
    out_csv = probe_dir / f"{label}_probe_metrics.csv"

    model, args, _cfg = load_v4_probe()
    device = torch.device("cuda" if (str(args.get("device", "cuda")).lower() == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    tile = int(args.get("tile_size", 256))
    overlap = int(args.get("overlap", 32))
    set_seed(int(args.get("seed", 42)), deterministic=True)

    rows_out: list[ProbeEvalRow] = []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_key", "regime", "noisy_path", "clean_path", "probe_psnr", "probe_ssim", "probe_edgef1"],
        )
        writer.writeheader()

        for row in params_rows:
            image_key = row["image_id"]
            regime = row["regime"]
            noisy_path = run_root / row["output_path"]
            clean_path = clean_root / row["source_path"]
            if not noisy_path.exists() or not clean_path.exists():
                raise ValidationStop(f"Missing probe evaluation input for {image_key}: noisy={noisy_path.exists()} clean={clean_path.exists()}")

            noisy_np = imread_uint(noisy_path)
            clean_np = imread_uint(clean_path)
            noisy = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).float().to(device)
            clean = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                den = denoise_tiled(model, noisy, tile=tile, overlap=overlap)
                den = torch.clamp(den, 0.0, 1.0)

            probe_psnr = float(psnr(den, clean))
            probe_ssim = float(ssim_metric(den, clean))
            probe_edge = float(fringe_edge_f1(den, clean))
            rec = ProbeEvalRow(
                image_key=image_key,
                regime=regime,
                noisy_path=noisy_path,
                clean_path=clean_path,
                psnr=probe_psnr,
                ssim=probe_ssim,
                edgef1=probe_edge,
            )
            rows_out.append(rec)
            writer.writerow(
                {
                    "image_key": image_key,
                    "regime": regime,
                    "noisy_path": str(noisy_path),
                    "clean_path": str(clean_path),
                    "probe_psnr": f"{probe_psnr:.6f}",
                    "probe_ssim": f"{probe_ssim:.6f}",
                    "probe_edgef1": f"{probe_edge:.6f}",
                }
            )

    return rows_out, out_csv


def write_probe_side_comparison(global_probe: list[ProbeEvalRow], by_probe: list[ProbeEvalRow]) -> tuple[Path, bool]:
    out_path = PHASE5_ROOT / "comparison" / "probe_side_comparison.csv"
    g_map = {row.image_key: row for row in global_probe}
    b_map = {row.image_key: row for row in by_probe}
    if set(g_map) != set(b_map):
        raise ValidationStop("Probe-side comparison failed: image sets differ between branches")

    measurable = False
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_key",
                "regime",
                "probe_psnr_global",
                "probe_psnr_byregime",
                "probe_ssim_global",
                "probe_ssim_byregime",
                "probe_edgef1_global",
                "probe_edgef1_byregime",
            ],
        )
        writer.writeheader()
        for key in sorted(g_map):
            g = g_map[key]
            b = b_map[key]
            if any(
                [
                    abs(g.psnr - b.psnr) > 1e-9,
                    abs(g.ssim - b.ssim) > 1e-9,
                    abs(g.edgef1 - b.edgef1) > 1e-9,
                ]
            ):
                measurable = True
            writer.writerow(
                {
                    "image_key": key,
                    "regime": g.regime,
                    "probe_psnr_global": f"{g.psnr:.6f}",
                    "probe_psnr_byregime": f"{b.psnr:.6f}",
                    "probe_ssim_global": f"{g.ssim:.6f}",
                    "probe_ssim_byregime": f"{b.ssim:.6f}",
                    "probe_edgef1_global": f"{g.edgef1:.6f}",
                    "probe_edgef1_byregime": f"{b.edgef1:.6f}",
                }
            )
    return out_path, measurable


def write_aggregate_summary(
    global_rows: list[dict[str, str]],
    by_rows: list[dict[str, str]],
    global_probe: list[ProbeEvalRow],
    by_probe: list[ProbeEvalRow],
) -> Path:
    out_path = PHASE5_ROOT / "comparison" / "aggregate_comparison_summary.csv"
    probe_maps = {
        "global": {row.image_key: row for row in global_probe},
        "by_regime": {row.image_key: row for row in by_probe},
    }
    rows_by_branch = {"global": global_rows, "by_regime": by_rows}
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run",
                "regime_scope",
                "count",
                "mean_generator_psnr",
                "std_generator_psnr",
                "mean_generator_ssim",
                "std_generator_ssim",
                "mean_generator_edgef1",
                "std_generator_edgef1",
                "mean_probe_psnr",
                "std_probe_psnr",
                "mean_probe_ssim",
                "std_probe_ssim",
                "mean_probe_edgef1",
                "std_probe_edgef1",
            ],
        )
        writer.writeheader()

        for run_label, rows in rows_by_branch.items():
            for regime_scope in ["all", "mid", "high"]:
                subset = rows if regime_scope == "all" else [r for r in rows if r["regime"] == regime_scope]
                if not subset:
                    continue
                probe_subset = [probe_maps[run_label][r["image_id"]] for r in subset]
                gen_psnr = [float(r["psnr"]) for r in subset]
                gen_ssim = [float(r["ssim"]) for r in subset]
                gen_edge = [float(r["edgef1"]) for r in subset]
                probe_psnr = [r.psnr for r in probe_subset]
                probe_ssim = [r.ssim for r in probe_subset]
                probe_edge = [r.edgef1 for r in probe_subset]
                writer.writerow(
                    {
                        "run": run_label,
                        "regime_scope": regime_scope,
                        "count": len(subset),
                        "mean_generator_psnr": f"{_mean(gen_psnr):.6f}",
                        "std_generator_psnr": f"{_std(gen_psnr):.6f}",
                        "mean_generator_ssim": f"{_mean(gen_ssim):.6f}",
                        "std_generator_ssim": f"{_std(gen_ssim):.6f}",
                        "mean_generator_edgef1": f"{_mean(gen_edge):.6f}",
                        "std_generator_edgef1": f"{_std(gen_edge):.6f}",
                        "mean_probe_psnr": f"{_mean(probe_psnr):.6f}",
                        "std_probe_psnr": f"{_std(probe_psnr):.6f}",
                        "mean_probe_ssim": f"{_mean(probe_ssim):.6f}",
                        "std_probe_ssim": f"{_std(probe_ssim):.6f}",
                        "mean_probe_edgef1": f"{_mean(probe_edge):.6f}",
                        "std_probe_edgef1": f"{_std(probe_edge):.6f}",
                    }
                )
    return out_path


def write_inputs_used(subset_dir: Path, manifest_path: Path, global_cmd: list[str], by_cmd: list[str]) -> Path:
    out_path = PHASE5_ROOT / "reports" / "phase5_inputs_used.txt"
    lines = [
        f"subset_dir={subset_dir}",
        f"manifest_path={manifest_path}",
        f"generator_script={V32_PATH}",
        f"frozen_v4_checkpoint={V4_CKPT}",
        f"phase4_source_root={PHASE4_ROOT}",
        f"global_command={subprocess.list2cmdline(global_cmd)}",
        f"by_regime_command={subprocess.list2cmdline(by_cmd)}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_report(
    manifest_path: Path,
    subset_dir: Path,
    global_run: Path,
    by_run: Path,
    inputs_used: Path,
    generator_csv: Path,
    probe_csv: Path,
    aggregate_csv: Path,
    global_rows: list[dict[str, str]],
    by_rows: list[dict[str, str]],
    global_probe: list[ProbeEvalRow],
    by_probe: list[ProbeEvalRow],
    generator_measurable: bool,
    probe_measurable: bool,
) -> tuple[Path, str]:
    report_path = PHASE5_ROOT / "reports" / "phase5_utility_validation_report.md"

    def mean_branch(rows: list[dict[str, str]], field: str) -> float:
        return _mean([float(r[field]) for r in rows])

    def mean_probe(rows: list[ProbeEvalRow], attr: str) -> float:
        return _mean([float(getattr(r, attr)) for r in rows])

    generator_delta_psnr = mean_branch(by_rows, "psnr") - mean_branch(global_rows, "psnr")
    generator_delta_ssim = mean_branch(by_rows, "ssim") - mean_branch(global_rows, "ssim")
    generator_delta_edge = mean_branch(by_rows, "edgef1") - mean_branch(global_rows, "edgef1")
    probe_delta_psnr = mean_probe(by_probe, "psnr") - mean_probe(global_probe, "psnr")
    probe_delta_ssim = mean_probe(by_probe, "ssim") - mean_probe(global_probe, "ssim")
    probe_delta_edge = mean_probe(by_probe, "edgef1") - mean_probe(global_probe, "edgef1")

    if probe_measurable and (abs(probe_delta_psnr) >= 0.10 or abs(probe_delta_ssim) >= 0.005 or abs(probe_delta_edge) >= 0.010):
        verdict = "v3.2 regime-aware calibration shows practically meaningful utility under a fixed probe"
    elif generator_measurable or probe_measurable:
        verdict = "v3.2 regime-aware calibration is structurally valid, but utility differences are limited"
    else:
        verdict = "v3.2 regime-aware calibration does not yet show clear practical utility beyond traceability"

    text = f"""# Phase 5 Utility Validation Report

## 1. Objective
The objective was to test whether `by-regime calibration` has practical consequence beyond traceability, using the same frozen manifest-driven subset and a fixed frozen V4 probe.

## 2. Frozen Inputs
- manifest: `{manifest_path}`
- subset: `{subset_dir}`
- generator script: `{V32_PATH}`
- seed: `123`
- calibration sources: inherited from the Phase 4 frozen run manifests
- frozen V4 checkpoint: `{V4_CKPT}`
- run_global path: `{global_run}`
- run_by_regime path: `{by_run}`

## 3. Controlled Experimental Setup
- The same script, manifest, subset, seed, RNG mode, source set, blur mode, conditioning mode, calibration sources, and output policy were used in both branches.
- The only variable changed was `--calibration-mode` (`global` vs `by-regime`).
- Full command/input provenance is recorded in `{inputs_used}`.

## 4. Generator-Side Comparison
- comparison CSV: `{generator_csv}`
- aggregate summary CSV: `{aggregate_csv}`
- mean generator deltas (`by-regime - global`):
  - PSNR: `{generator_delta_psnr:+.6f}`
  - SSIM: `{generator_delta_ssim:+.6f}`
  - EdgeF1: `{generator_delta_edge:+.6f}`
- Generator-side difference is `{"measurable" if generator_measurable else "not measurable"}` under the fixed Phase 5 setup.

## 5. Probe-Side Comparison
- probe comparison CSV: `{probe_csv}`
- aggregate summary CSV: `{aggregate_csv}`
- frozen probe path: `V4 canonical checkpoint + espi_dncnn_lite_v5 denoise_tiled metrics family`
- mean probe deltas (`by-regime - global`):
  - PSNR: `{probe_delta_psnr:+.6f}`
  - SSIM: `{probe_delta_ssim:+.6f}`
  - EdgeF1: `{probe_delta_edge:+.6f}`
- Probe-side difference is `{"measurable" if probe_measurable else "not measurable"}` under the fixed probe evaluation path.

## 6. Interpretation
This is a strengthening experiment, not a new benchmark branch. The result is interpreted only in terms of whether regime-aware calibration leaves a practical footprint under a fixed probe and fixed evaluation logic.

## 7. Limitations
- Sample size is limited to the frozen Phase 4 subset.
- The subset inherits the Phase 4 aligned-key design used to match calibration singles.
- This is not a classifier-level validation.
- No retraining was performed for either the generator or the probe.

## 8. Final Verdict
`{verdict}`
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path, verdict


def main() -> None:
    if not V32_PATH.exists():
        raise ValidationStop(f"Missing v3.2 script: {V32_PATH}")
    if not V4_CKPT.exists():
        raise ValidationStop(f"Missing frozen V4 checkpoint: {V4_CKPT}")

    prepare_root()
    subset_dir, manifest_path = copy_phase4_inputs()

    phase4_global_manifest = PHASE4_ROOT / "run_global" / "run_manifest_final.json"
    phase4_by_manifest = PHASE4_ROOT / "run_by_regime" / "run_manifest_final.json"
    if not phase4_global_manifest.exists() or not phase4_by_manifest.exists():
        raise ValidationStop("Missing Phase 4 run manifests needed to freeze the Phase 5 generator setup")

    global_run = PHASE5_ROOT / "run_global"
    by_run = PHASE5_ROOT / "run_by_regime"
    global_cmd = build_generator_command(phase4_global_manifest, subset_dir, manifest_path, global_run, "global")
    by_cmd = build_generator_command(phase4_by_manifest, subset_dir, manifest_path, by_run, "by-regime")
    inputs_used = write_inputs_used(subset_dir, manifest_path, global_cmd, by_cmd)

    run_command(global_cmd, Path.cwd())
    run_command(by_cmd, Path.cwd())

    _g_manifest, g_params_csv, _g_summary = verify_run_outputs(global_run, "Global")
    _b_manifest, b_params_csv, _b_summary = verify_run_outputs(by_run, "By-regime")
    global_rows = load_csv(g_params_csv)
    by_rows = load_csv(b_params_csv)

    generator_csv, generator_measurable = write_generator_side_comparison(global_rows, by_rows)

    global_probe, _g_probe_csv = evaluate_probe_branch(global_run, global_rows, subset_dir, "global")
    by_probe, _b_probe_csv = evaluate_probe_branch(by_run, by_rows, subset_dir, "by_regime")
    probe_csv, probe_measurable = write_probe_side_comparison(global_probe, by_probe)

    aggregate_csv = write_aggregate_summary(global_rows, by_rows, global_probe, by_probe)
    report_path, verdict = write_report(
        manifest_path,
        subset_dir,
        global_run,
        by_run,
        inputs_used,
        generator_csv,
        probe_csv,
        aggregate_csv,
        global_rows,
        by_rows,
        global_probe,
        by_probe,
        generator_measurable,
        probe_measurable,
    )

    summary = {
        "output_root": str(PHASE5_ROOT),
        "manifest_used": str(manifest_path),
        "frozen_v4_checkpoint": str(V4_CKPT),
        "generator_side_comparison_csv": str(generator_csv),
        "probe_side_comparison_csv": str(probe_csv),
        "aggregate_comparison_summary_csv": str(aggregate_csv),
        "phase5_utility_validation_report_md": str(report_path),
        "images_compared": len(global_rows),
        "measurable_generator_side_difference": generator_measurable,
        "measurable_probe_side_difference": probe_measurable,
        "final_verdict": verdict,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValidationStop as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
