from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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
PHASE2_ROOT = _OUT / "v3_2_phase2_order_invariance"
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
    conditioning_mode: str
    calibration_mode: str
    blur_mode: str


@dataclass(frozen=True)
class SubsetRow:
    index: int
    filename: str
    full_path: Path
    parsed_freq_hz: float | None
    parsed_amp_db: float | None
    selection_reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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
        conditioning_mode=str(args["conditioning_mode"]),
        calibration_mode=str(args["calibration_mode"]),
        blur_mode=str(args["blur_mode"]),
    )


def load_phase1_subset() -> list[SubsetRow]:
    csv_path = PHASE1_ROOT / "inputs" / "subset_sources.csv"
    if not csv_path.exists():
        raise ValidationStop(f"Missing Phase 1 subset CSV: {csv_path}")
    rows: list[SubsetRow] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_path = Path(row["full_path"])
            if not full_path.exists():
                raise ValidationStop(f"Subset source file no longer exists: {full_path}")
            rows.append(
                SubsetRow(
                    index=int(row["index"]),
                    filename=row["filename"],
                    full_path=full_path.resolve(),
                    parsed_freq_hz=float(row["parsed_freq_hz"]) if row["parsed_freq_hz"] else None,
                    parsed_amp_db=float(row["parsed_amp_db"]) if row["parsed_amp_db"] else None,
                    selection_reason=row["selection_reason"],
                )
            )
    if not rows:
        raise ValidationStop(f"Phase 1 subset CSV is empty: {csv_path}")
    return rows


def prepare_root() -> None:
    if PHASE2_ROOT.exists():
        raise ValidationStop(f"Phase 2 output root already exists, refusing to mix artifacts: {PHASE2_ROOT}")
    for subdir in [
        PHASE2_ROOT / "inputs",
        PHASE2_ROOT / "run_A",
        PHASE2_ROOT / "run_B",
        PHASE2_ROOT / "comparison",
        PHASE2_ROOT / "reports",
    ]:
        subdir.mkdir(parents=True, exist_ok=False)


def copy_subset(rows: list[SubsetRow]) -> tuple[Path, Path, Path]:
    inputs_dir = PHASE2_ROOT / "inputs"
    subset_csv = inputs_dir / "subset_sources.csv"
    subset_txt = inputs_dir / "subset_sources.txt"
    subset_clean_dir = inputs_dir / "subset_clean"
    subset_clean_dir.mkdir(parents=True, exist_ok=False)

    with open(subset_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "filename", "full_path", "parsed_freq_hz", "parsed_amp_db", "selection_reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row.index,
                    "filename": row.filename,
                    "full_path": str(row.full_path),
                    "parsed_freq_hz": "" if row.parsed_freq_hz is None else f"{row.parsed_freq_hz:.1f}",
                    "parsed_amp_db": "" if row.parsed_amp_db is None else f"{row.parsed_amp_db:.1f}",
                    "selection_reason": row.selection_reason,
                }
            )
            shutil.copy2(row.full_path, subset_clean_dir / row.filename)

    with open(subset_txt, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row.full_path}\n")

    return subset_csv, subset_txt, subset_clean_dir


def build_base_v32_command(input_root: Path, output_root: Path, cfg: FrozenConfig) -> list[str]:
    return [
        sys.executable,
        str(V32_PATH),
        "--input",
        str(input_root),
        "--output",
        str(output_root),
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
        "--conditioning-mode",
        "global",
        "--rng-mode",
        "per_image",
        "--calibration-mode",
        "global",
        "--blur-mode",
        cfg.blur_mode,
        "--export-metrics",
        str(output_root / "metrics.csv"),
        "--export-summary",
        str(output_root / "summary.json"),
    ]


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


def write_input_orders(canonical_order: list[str], reverse_order: list[str]) -> tuple[Path, Path]:
    run_a_path = PHASE2_ROOT / "reports" / "input_order_runA.txt"
    run_b_path = PHASE2_ROOT / "reports" / "input_order_runB.txt"
    run_a_path.write_text("\n".join(canonical_order) + "\n", encoding="utf-8")
    run_b_path.write_text("\n".join(reverse_order) + "\n", encoding="utf-8")
    return run_a_path, run_b_path


def collect_output_images(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            out[path.relative_to(root).as_posix()] = path
    return out


def extract_single_generated_image(temp_output_root: Path) -> Path:
    images = [p for p in sorted(temp_output_root.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if len(images) != 1:
        raise ValidationStop(f"Expected exactly one generated image under {temp_output_root}, found {len(images)}")
    return images[0]


def compare_structure(run_a_images: dict[str, Path], run_b_images: dict[str, Path]) -> tuple[list[str], list[str]]:
    only_a = sorted(set(run_a_images) - set(run_b_images))
    only_b = sorted(set(run_b_images) - set(run_a_images))
    if only_a or only_b:
        raise ValidationStop(
            "Structural mismatch detected before byte comparison:\n"
            f"missing_in_runB={only_a}\n"
            f"extra_in_runB={only_b}"
        )
    return only_a, only_b


def write_hash_comparison(run_a_images: dict[str, Path], run_b_images: dict[str, Path]) -> tuple[Path, list[dict[str, str]]]:
    csv_path = PHASE2_ROOT / "comparison" / "file_hash_comparison.csv"
    rows: list[dict[str, str]] = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_key", "filename_runA", "filename_runB", "sha256_runA", "sha256_runB", "byte_identical"],
        )
        writer.writeheader()
        for key in sorted(run_a_images):
            sha_a = sha256_file(run_a_images[key])
            sha_b = sha256_file(run_b_images[key])
            row = {
                "image_key": key,
                "filename_runA": key,
                "filename_runB": key,
                "sha256_runA": sha_a,
                "sha256_runB": sha_b,
                "byte_identical": "true" if sha_a == sha_b else "false",
            }
            writer.writerow(row)
            rows.append(row)
    return csv_path, rows


def write_pixel_comparison(run_a_images: dict[str, Path], run_b_images: dict[str, Path]) -> tuple[Path, list[dict[str, str]]]:
    csv_path = PHASE2_ROOT / "comparison" / "pixel_comparison.csv"
    rows: list[dict[str, str]] = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
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
        for key in sorted(run_a_images):
            arr_a = np.asarray(Image.open(run_a_images[key]), dtype=np.float32)
            arr_b = np.asarray(Image.open(run_b_images[key]), dtype=np.float32)
            if arr_a.shape != arr_b.shape:
                raise ValidationStop(f"Output shape mismatch for {key}: {arr_a.shape} vs {arr_b.shape}")
            diff = np.abs(arr_a - arr_b)
            max_abs = float(diff.max()) if diff.size else 0.0
            mean_abs = float(diff.mean()) if diff.size else 0.0
            mse = float(np.mean((arr_a - arr_b) ** 2)) if diff.size else 0.0
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
    run_a_order_path: Path,
    run_b_order_path: Path,
    cfg: FrozenConfig,
    hash_rows: list[dict[str, str]],
    pixel_rows: list[dict[str, str]],
) -> Path:
    report_path = PHASE2_ROOT / "reports" / "order_invariance_report.md"
    compared_pairs = len(hash_rows)
    byte_identical = sum(row["byte_identical"] == "true" for row in hash_rows)
    max_abs = max(float(row["max_abs_diff"]) for row in pixel_rows) if pixel_rows else 0.0
    worst = max(pixel_rows, key=lambda row: float(row["max_abs_diff"])) if pixel_rows else None
    verdict = (
        "v3.2 is order-invariant under deterministic per-image RNG mode"
        if byte_identical == compared_pairs and all(row["numeric_equivalence_status"] == "exact" for row in pixel_rows)
        else "v3.2 is not yet order-invariant under deterministic per-image RNG mode"
    )

    lines = [
        "# order_invariance_report",
        "",
        "## 1. Objective",
        "The objective was to validate the v3.2 deterministic per-image RNG claim by testing whether output images remain identical when the same frozen subset is processed under different execution schedules.",
        "",
        "## 2. Frozen Inputs",
        f"- v3.2 path: `{V32_PATH}`",
        f"- Source subset reused from Phase 1: `{subset_csv}`",
        f"- Seed: `{cfg.seed}`",
        f"- Frozen global parameters: `speckle_k={cfg.speckle_k}`, `poisson_peak={cfg.poisson_peak}`, `gauss_sigma={cfg.gauss_sigma}`, `match={cfg.match}`, `bitdepth={cfg.out_bitdepth}`, `format={cfg.out_format}`",
        f"- Run A order file: `{run_a_order_path}`",
        f"- Run B order file: `{run_b_order_path}`",
        "",
        "## 3. Execution Design",
        "- Run A used the frozen subset in canonical filename order under `--rng-mode per_image`.",
        "- Run B used the same frozen subset in reverse order under `--rng-mode per_image`.",
        "- Because the v3.2 CLI performs internally sorted file discovery and exposes no direct order flag, the reverse-order perturbation was implemented at the orchestration layer via one-image-at-a-time reverse scheduling while preserving the same image keys and the same global parameters.",
        "",
        "## 4. Structural Output Comparison",
        f"- Compared output pairs: `{compared_pairs}`",
        "- Structural consistency: `pass`",
        "",
        "## 5. Byte-Level Comparison",
        f"- Byte-identical pairs: `{byte_identical}`",
        f"- Byte-identical percentage: `{(100.0 * byte_identical / compared_pairs) if compared_pairs else 0.0:.2f}%`",
        "",
        "## 6. Numeric Comparison",
        f"- Max observed deviation: `{max_abs:.6f}`",
        f"- Worst-case file: `{worst['image_key'] if worst else 'N/A'}`",
        f"- Numeric interpretation: `{'exact across all pairs' if max_abs == 0.0 else 'non-zero differences detected'}`",
        "",
        "## 7. Interpretation",
        "This phase did not compare against v3.1. It tested whether v3.2 with deterministic per-image RNG removes order dependence from generation under the same frozen global configuration.",
        "",
        "## 8. Final Verdict",
        verdict,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    if not V32_PATH.exists():
        raise ValidationStop(f"Missing v3.2 script: {V32_PATH}")

    cfg = load_phase1_config()
    subset_rows = load_phase1_subset()
    prepare_root()
    subset_csv, subset_txt, subset_clean_dir = copy_subset(subset_rows)
    del subset_txt

    canonical_order = sorted(row.filename for row in subset_rows)
    reverse_order = list(reversed(canonical_order))
    order_a_path, order_b_path = write_input_orders(canonical_order, reverse_order)

    run_a_output = PHASE2_ROOT / "run_A" / "batch"
    run_a_output.mkdir(parents=True, exist_ok=False)
    run_a_cmd = build_base_v32_command(subset_clean_dir, run_a_output, cfg)
    run_command(run_a_cmd, PHASE2_ROOT)

    run_b_root = PHASE2_ROOT / "run_B"
    run_b_images_root = run_b_root / "images"
    run_b_images_root.mkdir(parents=True, exist_ok=False)
    temp_inputs_root = run_b_root / "temp_inputs"
    temp_outputs_root = run_b_root / "temp_outputs"
    temp_inputs_root.mkdir(parents=True, exist_ok=False)
    temp_outputs_root.mkdir(parents=True, exist_ok=False)

    source_by_name = {row.filename: row.full_path for row in subset_rows}
    for idx, filename in enumerate(reverse_order, start=1):
        input_dir = temp_inputs_root / f"{idx:02d}_{Path(filename).stem}"
        output_dir = temp_outputs_root / f"{idx:02d}_{Path(filename).stem}"
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_by_name[filename], input_dir / filename)
        cmd = build_base_v32_command(input_dir, output_dir, cfg)
        run_command(cmd, PHASE2_ROOT)
        generated = extract_single_generated_image(output_dir)
        shutil.copy2(generated, run_b_images_root / generated.name)

    run_a_images = collect_output_images(run_a_output)
    run_b_images = collect_output_images(run_b_images_root)
    compare_structure(run_a_images, run_b_images)
    hash_csv, hash_rows = write_hash_comparison(run_a_images, run_b_images)
    pixel_csv, pixel_rows = write_pixel_comparison(run_a_images, run_b_images)
    report_path = write_report(subset_csv, order_a_path, order_b_path, cfg, hash_rows, pixel_rows)

    byte_identical = sum(row["byte_identical"] == "true" for row in hash_rows)
    max_abs = max(float(row["max_abs_diff"]) for row in pixel_rows) if pixel_rows else 0.0
    verdict = (
        "v3.2 is order-invariant under deterministic per-image RNG mode"
        if byte_identical == len(hash_rows) and max_abs == 0.0
        else "v3.2 is not yet order-invariant under deterministic per-image RNG mode"
    )

    result = {
        "output_root": str(PHASE2_ROOT),
        "input_order_runA": str(order_a_path),
        "input_order_runB": str(order_b_path),
        "file_hash_comparison_csv": str(hash_csv),
        "pixel_comparison_csv": str(pixel_csv),
        "order_invariance_report_md": str(report_path),
        "pairs_compared": len(hash_rows),
        "byte_identical_pairs": byte_identical,
        "max_observed_deviation": max_abs,
        "final_verdict": verdict,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValidationStop as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
