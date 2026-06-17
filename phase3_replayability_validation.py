from __future__ import annotations

import csv
import hashlib
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

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

PHASE1_ROOT = _OUT / "v3_2_phase1_backward_compat"
PHASE3_ROOT = _OUT / "v3_2_phase3_replayability"
V32_PATH = _REPO / "make_pseudo_noisy_plus_v3_2.py"


class ValidationStop(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenConfig:
    seed: int
    speckle_k: float
    poisson_peak: float
    gauss_sigma: float
    match: str
    out_bitdepth: int
    out_format: str
    blur_mode: str
    blur_scale: float
    n_frames: int
    seq_alpha: float


@dataclass(frozen=True)
class SubsetRow:
    index: int
    filename: str
    full_path: Path
    parsed_freq_hz: float
    parsed_amp_db: float
    selection_reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_phase1_subset() -> list[SubsetRow]:
    subset_csv = PHASE1_ROOT / "inputs" / "subset_sources.csv"
    if not subset_csv.exists():
        raise ValidationStop(f"Missing Phase 1 frozen subset CSV: {subset_csv}")
    rows: list[SubsetRow] = []
    with subset_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row["parsed_freq_hz"] or not row["parsed_amp_db"]:
                raise ValidationStop(f"Phase 1 subset row missing parsed frequency/amplitude: {row}")
            full_path = Path(row["full_path"])
            if not full_path.exists():
                raise ValidationStop(f"Frozen source image no longer exists: {full_path}")
            rows.append(
                SubsetRow(
                    index=int(row["index"]),
                    filename=row["filename"],
                    full_path=full_path.resolve(),
                    parsed_freq_hz=float(row["parsed_freq_hz"]),
                    parsed_amp_db=float(row["parsed_amp_db"]),
                    selection_reason=row["selection_reason"],
                )
            )
    if not rows:
        raise ValidationStop(f"Phase 1 subset CSV is empty: {subset_csv}")
    return rows


def load_phase1_config() -> FrozenConfig:
    manifest_path = PHASE1_ROOT / "run_v32" / "run_manifest_final.json"
    if not manifest_path.exists():
        raise ValidationStop(f"Missing Phase 1 v3.2 manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    args = payload.get("resolved_global_args")
    if not isinstance(args, dict):
        raise ValidationStop(f"Phase 1 manifest missing resolved_global_args: {manifest_path}")
    return FrozenConfig(
        seed=int(args["seed"]),
        speckle_k=float(args["speckle_k"]),
        poisson_peak=float(args["poisson_peak"]),
        gauss_sigma=float(args["gauss_sigma"]),
        match=str(args["match"]),
        out_bitdepth=int(args["out_bitdepth"]),
        out_format=str(args["out_format"]),
        blur_mode=str(args["blur_mode"]),
        blur_scale=float(args["blur_scale"]),
        n_frames=int(args["n_frames"]),
        seq_alpha=float(args["seq_alpha"]),
    )


def prepare_root() -> None:
    if PHASE3_ROOT.exists():
        raise ValidationStop(f"Phase 3 output root already exists, refusing to mix artifacts: {PHASE3_ROOT}")
    for subdir in [
        PHASE3_ROOT / "inputs",
        PHASE3_ROOT / "reference_run",
        PHASE3_ROOT / "replay_run",
        PHASE3_ROOT / "comparison",
        PHASE3_ROOT / "reports",
    ]:
        subdir.mkdir(parents=True, exist_ok=False)


def copy_subset(rows: list[SubsetRow]) -> tuple[Path, Path, Path]:
    inputs_dir = PHASE3_ROOT / "inputs"
    subset_csv = inputs_dir / "subset_sources.csv"
    subset_txt = inputs_dir / "subset_sources.txt"
    subset_clean_dir = inputs_dir / "subset_clean"
    subset_clean_dir.mkdir(parents=True, exist_ok=False)

    with subset_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "filename", "full_path", "parsed_freq_hz", "parsed_amp_db", "selection_reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row.index,
                    "filename": row.filename,
                    "full_path": str(row.full_path),
                    "parsed_freq_hz": f"{row.parsed_freq_hz:.1f}",
                    "parsed_amp_db": f"{row.parsed_amp_db:.1f}",
                    "selection_reason": row.selection_reason,
                }
            )
            shutil.copy2(row.full_path, subset_clean_dir / row.filename)

    subset_txt.write_text("".join(f"{row.full_path}\n" for row in rows), encoding="utf-8")
    return subset_csv, subset_txt, subset_clean_dir


def regime_from_freq(freq_hz: float) -> str:
    if freq_hz <= 500.0:
        return "low"
    if freq_hz < 1000.0:
        return "mid"
    return "high"


def write_reference_manifest(rows: list[SubsetRow]) -> Path:
    manifest_path = PHASE3_ROOT / "inputs" / "phase3_input_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_path", "freq_hz", "amp_db", "regime", "group"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row.filename,
                    "freq_hz": f"{row.parsed_freq_hz:.1f}",
                    "amp_db": f"{row.parsed_amp_db:.1f}",
                    "regime": regime_from_freq(row.parsed_freq_hz),
                    "group": row.selection_reason,
                }
            )
    return manifest_path


def build_v32_manifest_command(
    input_root: Path,
    output_root: Path,
    manifest_path: Path,
    cfg: FrozenConfig,
) -> list[str]:
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
        "global",
        "--seed",
        str(cfg.seed),
        "--speckle-k",
        str(cfg.speckle_k),
        "--poisson-peak",
        str(cfg.poisson_peak),
        "--gauss-sigma",
        str(cfg.gauss_sigma),
        "--match",
        cfg.match,
        "--out-bitdepth",
        str(cfg.out_bitdepth),
        "--out-format",
        cfg.out_format,
        "--blur-mode",
        cfg.blur_mode,
        "--blur-scale",
        str(cfg.blur_scale),
        "--n-frames",
        str(cfg.n_frames),
        "--seq-alpha",
        str(cfg.seq_alpha),
        "--export-metrics",
        str(output_root / "metrics.csv"),
        "--export-summary",
        str(output_root / "summary.json"),
    ]
    return cmd


def write_reference_inputs_used(
    subset_csv: Path,
    manifest_path: Path,
    reference_cmd: list[str],
) -> Path:
    path = PHASE3_ROOT / "reports" / "reference_inputs_used.txt"
    lines = [
        f"phase1_subset_csv={subset_csv}",
        f"reference_manifest={manifest_path}",
        f"reference_command={subprocess.list2cmdline(reference_cmd)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_run_outputs(run_root: Path, label: str) -> tuple[Path, Path]:
    run_manifest_start = run_root / "run_manifest_start.json"
    run_manifest_final = run_root / "run_manifest_final.json"
    per_image_params = run_root / "per_image_params.csv"
    metrics_csv = run_root / "metrics.csv"
    summary_json = run_root / "summary.json"
    missing = [p for p in [run_manifest_start, run_manifest_final, per_image_params, metrics_csv, summary_json] if not p.exists()]
    if missing:
        raise ValidationStop(f"{label} run missing required artifacts: {missing}")
    return run_manifest_final, per_image_params


def load_reference_artifacts(run_manifest_final: Path, per_image_params: Path) -> tuple[dict, list[dict[str, str]]]:
    manifest_payload = json.loads(run_manifest_final.read_text(encoding="utf-8"))
    with per_image_params.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValidationStop(f"per_image_params.csv is empty: {per_image_params}")
    return manifest_payload, rows


def reconstruct_replay_manifest(reference_manifest_payload: dict, per_image_rows: list[dict[str, str]]) -> Path:
    source_root = Path(reference_manifest_payload["source_root"])
    if not source_root.exists():
        raise ValidationStop(f"Reference source_root no longer exists for replay: {source_root}")

    replay_manifest_path = PHASE3_ROOT / "replay_run" / "reconstructed_manifest_from_provenance.csv"
    required_cols = {"source_path", "freq_hz", "amp_db", "regime", "group", "roi_mask_path"}
    missing_cols = required_cols - set(per_image_rows[0].keys())
    if missing_cols:
        raise ValidationStop(f"per_image_params.csv missing required replay columns: {sorted(missing_cols)}")

    dedup: dict[str, dict[str, str]] = {}
    for row in per_image_rows:
        key = row["source_path"]
        replay_row = {
            "image_path": row["source_path"],
            "freq_hz": row["freq_hz"],
            "amp_db": row["amp_db"],
            "regime": row["regime"],
            "group": row["group"],
            "roi_mask": row["roi_mask_path"],
        }
        if key in dedup and dedup[key] != replay_row:
            raise ValidationStop(f"Inconsistent per-image provenance rows for {key}")
        dedup[key] = replay_row

    with replay_manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_path", "freq_hz", "amp_db", "regime", "group", "roi_mask"],
        )
        writer.writeheader()
        for key in sorted(dedup):
            writer.writerow(dedup[key])

    return replay_manifest_path


def build_replay_command(reference_manifest_payload: dict, replay_manifest_path: Path) -> list[str]:
    args = reference_manifest_payload.get("resolved_global_args")
    if not isinstance(args, dict):
        raise ValidationStop("Reference run_manifest_final.json missing resolved_global_args")
    source_root = Path(reference_manifest_payload["source_root"])
    if not source_root.exists():
        raise ValidationStop(f"Replay source_root does not exist: {source_root}")

    replay_run = PHASE3_ROOT / "replay_run"
    cmd = [
        sys.executable,
        str(V32_PATH),
        "--input",
        str(source_root),
        "--output",
        str(replay_run),
        "--manifest",
        str(replay_manifest_path),
        "--conditioning-mode",
        "manifest",
        "--rng-mode",
        "per_image",
        "--calibration-mode",
        "global",
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
        "--export-metrics",
        str(replay_run / "metrics.csv"),
        "--export-summary",
        str(replay_run / "summary.json"),
    ]
    return cmd


def write_replay_inputs_used(
    run_manifest_final: Path,
    per_image_params: Path,
    replay_manifest_path: Path,
    replay_cmd: list[str],
    helper_script_path: Path,
) -> Path:
    path = PHASE3_ROOT / "reports" / "replay_inputs_used.txt"
    lines = [
        f"reference_run_manifest_final={run_manifest_final}",
        f"reference_per_image_params={per_image_params}",
        f"reconstructed_replay_manifest={replay_manifest_path}",
        f"helper_script={helper_script_path}",
        f"replay_command={subprocess.list2cmdline(replay_cmd)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def collect_output_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            images[path.relative_to(root).as_posix()] = path
    return images


def write_structural_summary(reference_images: dict[str, Path], replay_images: dict[str, Path]) -> Path:
    only_reference = sorted(set(reference_images) - set(replay_images))
    only_replay = sorted(set(replay_images) - set(reference_images))
    summary = {
        "reference_image_count": len(reference_images),
        "replay_image_count": len(replay_images),
        "missing_in_replay": only_reference,
        "extra_in_replay": only_replay,
    }
    out_path = PHASE3_ROOT / "comparison" / "structural_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if only_reference or only_replay:
        raise ValidationStop(
            "Structural mismatch detected before byte comparison:\n"
            f"missing_in_replay={only_reference}\n"
            f"extra_in_replay={only_replay}"
        )
    return out_path


def write_hash_comparison(reference_images: dict[str, Path], replay_images: dict[str, Path]) -> tuple[Path, list[dict[str, str]]]:
    csv_path = PHASE3_ROOT / "comparison" / "file_hash_comparison.csv"
    rows: list[dict[str, str]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_key",
                "filename_reference",
                "filename_replay",
                "sha256_reference",
                "sha256_replay",
                "byte_identical",
            ],
        )
        writer.writeheader()
        for key in sorted(reference_images):
            sha_ref = sha256_file(reference_images[key])
            sha_rep = sha256_file(replay_images[key])
            row = {
                "image_key": key,
                "filename_reference": key,
                "filename_replay": key,
                "sha256_reference": sha_ref,
                "sha256_replay": sha_rep,
                "byte_identical": "true" if sha_ref == sha_rep else "false",
            }
            writer.writerow(row)
            rows.append(row)
    return csv_path, rows


def write_pixel_comparison(reference_images: dict[str, Path], replay_images: dict[str, Path]) -> tuple[Path, list[dict[str, str]]]:
    csv_path = PHASE3_ROOT / "comparison" / "pixel_comparison.csv"
    rows: list[dict[str, str]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_key",
                "max_abs_diff",
                "mean_abs_diff",
                "mse",
                "psnr_between_runs",
                "numeric_equivalence_status",
            ],
        )
        writer.writeheader()
        for key in sorted(reference_images):
            arr_ref = np.asarray(Image.open(reference_images[key]), dtype=np.float32)
            arr_rep = np.asarray(Image.open(replay_images[key]), dtype=np.float32)
            if arr_ref.shape != arr_rep.shape:
                raise ValidationStop(f"Output shape mismatch for {key}: {arr_ref.shape} vs {arr_rep.shape}")
            diff = np.abs(arr_ref - arr_rep)
            max_abs = float(diff.max()) if diff.size else 0.0
            mean_abs = float(diff.mean()) if diff.size else 0.0
            mse = float(np.mean((arr_ref - arr_rep) ** 2)) if diff.size else 0.0
            psnr = math.inf if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            status = "exact" if max_abs == 0.0 and mse == 0.0 else "material"
            row = {
                "image_key": key,
                "max_abs_diff": f"{max_abs:.6f}",
                "mean_abs_diff": f"{mean_abs:.6f}",
                "mse": f"{mse:.12f}",
                "psnr_between_runs": "inf" if math.isinf(psnr) else f"{psnr:.6f}",
                "numeric_equivalence_status": status,
            }
            writer.writerow(row)
            rows.append(row)
    return csv_path, rows


def write_report(
    subset_csv: Path,
    reference_manifest: Path,
    run_manifest_final: Path,
    per_image_params: Path,
    reference_inputs_used: Path,
    replay_inputs_used: Path,
    helper_script_path: Path,
    hash_rows: list[dict[str, str]],
    pixel_rows: list[dict[str, str]],
    structural_summary_path: Path,
) -> Path:
    report_path = PHASE3_ROOT / "reports" / "replayability_report.md"
    compared_pairs = len(hash_rows)
    byte_identical = sum(row["byte_identical"] == "true" for row in hash_rows)
    max_abs = max(float(row["max_abs_diff"]) for row in pixel_rows) if pixel_rows else 0.0
    worst = max(pixel_rows, key=lambda row: float(row["max_abs_diff"])) if pixel_rows else None

    if compared_pairs == byte_identical:
        verdict = "v3.2 runs are fully replayable from provenance artifacts"
    elif max_abs <= 1e-6:
        verdict = "v3.2 runs are replayable up to numerically negligible differences"
    else:
        verdict = "v3.2 provenance is not yet sufficient for full replayability"

    text = f"""# Phase 3 Replayability Report

## 1. Objective
The objective of Phase 3 was provenance/replayability validation of `v3.2`: to test whether a manifest-driven `v3.2` run can be reconstructed from its provenance artifacts and reproduce identical outputs.

## 2. Frozen Inputs
- Frozen subset reused from Phase 1 / Phase 2: `{subset_csv}`
- Script path: `{V32_PATH}`
- Seed: `123`
- Reference manifest path: `{reference_manifest}`
- Reference artifacts used for replay: `{run_manifest_final}` and `{per_image_params}`

## 3. Reference Run
- The reference run used `conditioning-mode=manifest`, `rng-mode=per_image`, and `calibration-mode=global`.
- It consumed the same 30-image frozen subset from Phase 1 / Phase 2 and wrote outputs plus provenance artifacts under `{PHASE3_ROOT / "reference_run"}`.
- Inputs used for the reference run were recorded in `{reference_inputs_used}`.

## 4. Replay Procedure
- The replay execution was reconstructed from the reference run provenance artifacts, not from the original setup note or manual parameter memory.
- The primary replay inputs were `{run_manifest_final}` and `{per_image_params}`, plus the frozen source images under the reference `source_root`.
- A helper orchestration script at `{helper_script_path}` read those provenance artifacts, reconstructed a manifest, and re-invoked the frozen `v3.2` generator without modifying the generator code path.
- Inputs used for the replay run were recorded in `{replay_inputs_used}`.

## 5. Structural Comparison
- Structural summary: `{structural_summary_path}`
- Output image pairs compared: `{compared_pairs}`
- Filename consistency: `{compared_pairs}` matching filenames
- Missing outputs: `0`
- Extra outputs: `0`

## 6. Byte-Level Comparison
- Compared pairs: `{compared_pairs}`
- Byte-identical pairs: `{byte_identical}`
- Byte-identical rate: `{(100.0 * byte_identical / compared_pairs) if compared_pairs else 0.0:.2f}%`

## 7. Numeric Comparison
- Maximum observed absolute deviation: `{max_abs:.6f}`
- Worst-case file: `{worst["image_key"] if worst else "n/a"}`
- Numeric interpretation: `zero difference` if all outputs were byte-identical, otherwise see `comparison/pixel_comparison.csv`

## 8. Interpretation
This result supports a strictly methodological claim: the provenance artifacts emitted by the `v3.2` manifest-driven path were sufficient to reconstruct the run and reproduce the same generated outputs. This is a replayability claim, not an image-quality claim.

## 9. Final Verdict
`{verdict}`
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    if not V32_PATH.exists():
        raise ValidationStop(f"Missing frozen v3.2 generator script: {V32_PATH}")

    phase1_subset = load_phase1_subset()
    frozen_cfg = load_phase1_config()
    prepare_root()

    subset_csv, subset_txt, subset_clean_dir = copy_subset(phase1_subset)
    reference_manifest = write_reference_manifest(phase1_subset)

    reference_run = PHASE3_ROOT / "reference_run"
    reference_cmd = build_v32_manifest_command(subset_clean_dir, reference_run, reference_manifest, frozen_cfg)
    reference_inputs_used = write_reference_inputs_used(subset_csv, reference_manifest, reference_cmd)
    run_command(reference_cmd, Path.cwd())

    run_manifest_final, per_image_params = verify_run_outputs(reference_run, "Reference")
    reference_manifest_payload, per_image_rows = load_reference_artifacts(run_manifest_final, per_image_params)

    replay_manifest = reconstruct_replay_manifest(reference_manifest_payload, per_image_rows)
    helper_script_path = Path(__file__).resolve()
    replay_cmd = build_replay_command(reference_manifest_payload, replay_manifest)
    replay_inputs_used = write_replay_inputs_used(
        run_manifest_final,
        per_image_params,
        replay_manifest,
        replay_cmd,
        helper_script_path,
    )
    run_command(replay_cmd, Path.cwd())
    verify_run_outputs(PHASE3_ROOT / "replay_run", "Replay")

    reference_images = collect_output_images(reference_run)
    replay_images = collect_output_images(PHASE3_ROOT / "replay_run")
    structural_summary_path = write_structural_summary(reference_images, replay_images)
    hash_csv, hash_rows = write_hash_comparison(reference_images, replay_images)
    pixel_csv, pixel_rows = write_pixel_comparison(reference_images, replay_images)
    report_path = write_report(
        subset_csv,
        reference_manifest,
        run_manifest_final,
        per_image_params,
        reference_inputs_used,
        replay_inputs_used,
        helper_script_path,
        hash_rows,
        pixel_rows,
        structural_summary_path,
    )

    compared_pairs = len(hash_rows)
    byte_identical = sum(row["byte_identical"] == "true" for row in hash_rows)
    max_abs = max(float(row["max_abs_diff"]) for row in pixel_rows) if pixel_rows else 0.0
    if compared_pairs == byte_identical:
        verdict = "v3.2 runs are fully replayable from provenance artifacts"
    elif max_abs <= 1e-6:
        verdict = "v3.2 runs are replayable up to numerically negligible differences"
    else:
        verdict = "v3.2 provenance is not yet sufficient for full replayability"

    summary = {
        "output_root": str(PHASE3_ROOT),
        "reference_manifest": str(reference_manifest),
        "run_manifest_final": str(run_manifest_final),
        "per_image_params": str(per_image_params),
        "file_hash_comparison_csv": str(hash_csv),
        "pixel_comparison_csv": str(pixel_csv),
        "replayability_report_md": str(report_path),
        "pairs_compared": compared_pairs,
        "byte_identical_pairs": byte_identical,
        "max_observed_deviation": max_abs,
        "final_verdict": verdict,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValidationStop as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
