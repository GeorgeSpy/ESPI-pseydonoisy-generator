from __future__ import annotations

import csv
import hashlib
import json
import math
import re
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
FILENAME_RE = re.compile(r"(?P<freq>\d+(?:\.\d+)?)Hz_(?P<amp>\d+(?:\.\d+)?)db", re.IGNORECASE)

V31_PATH = _REPO / "make_pseudo_noisy_plus_v3_1.py"
V32_PATH = _REPO / "make_pseudo_noisy_plus_v3_2.py"
V32_DOWNLOADS_PATH = Path(_os.environ.get("ESPI_V32_ALT", str(V32_PATH)))
SOURCE_ROOT = _DATA / "wood_Averaged" / "W01_ESPI_90db-Averaged"
BASE_OUTPUT_ROOT = _OUT / "v3_2_phase1_backward_compat"

SEED = 123
SPECKLE_K = 1.89
POISSON_PEAK = 12.5
GAUSS_SIGMA = 0.0919
MATCH_MODE = "meanstd"
OUT_BITDEPTH = 8
OUT_FORMAT = "png"
SUBSET_SIZE_PER_BAND = 10


class ValidationStop(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRow:
    filename: str
    full_path: Path
    parsed_freq_hz: float | None
    parsed_amp_db: float | None


@dataclass(frozen=True)
class SelectedRow:
    index: int
    filename: str
    full_path: Path
    parsed_freq_hz: float | None
    parsed_amp_db: float | None
    selection_reason: str


def parse_source_row(path: Path) -> SourceRow:
    match = FILENAME_RE.search(path.name)
    freq = float(match.group("freq")) if match else None
    amp = float(match.group("amp")) if match else None
    return SourceRow(
        filename=path.name,
        full_path=path.resolve(),
        parsed_freq_hz=freq,
        parsed_amp_db=amp,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_identical_scripts() -> dict[str, str]:
    if not V31_PATH.exists():
        raise ValidationStop(f"Missing frozen v3.1 script: {V31_PATH}")
    if not V32_PATH.exists():
        raise ValidationStop(f"Missing canonical v3.2 script: {V32_PATH}")
    if not V32_DOWNLOADS_PATH.exists():
        raise ValidationStop(f"Missing secondary v3.2 script copy: {V32_DOWNLOADS_PATH}")

    v32_hash = sha256_file(V32_PATH)
    v32_downloads_hash = sha256_file(V32_DOWNLOADS_PATH)
    if v32_hash != v32_downloads_hash:
        raise ValidationStop(
            "Ambiguity in frozen v3.2 compare target: canonical and Downloads copies differ "
            f"({V32_PATH} vs {V32_DOWNLOADS_PATH})"
        )
    return {
        "v31_sha256": sha256_file(V31_PATH),
        "v32_sha256": v32_hash,
    }


def scan_source_universe() -> list[SourceRow]:
    if not SOURCE_ROOT.exists():
        raise ValidationStop(f"Source root not found: {SOURCE_ROOT}")
    rows = [parse_source_row(p) for p in sorted(SOURCE_ROOT.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not rows:
        raise ValidationStop(f"No candidate clean files found under {SOURCE_ROOT}")
    return rows


def select_evenly(block: list[SourceRow], label: str) -> list[SelectedRow]:
    if len(block) < SUBSET_SIZE_PER_BAND:
        raise ValidationStop(f"Not enough files in {label} band for deterministic selection: {len(block)}")
    indices = [round(i * (len(block) - 1) / (SUBSET_SIZE_PER_BAND - 1)) for i in range(SUBSET_SIZE_PER_BAND)]
    selected = []
    for idx in indices:
        row = block[idx]
        selected.append(
            SelectedRow(
                index=-1,
                filename=row.filename,
                full_path=row.full_path,
                parsed_freq_hz=row.parsed_freq_hz,
                parsed_amp_db=row.parsed_amp_db,
                selection_reason=f"{label}_coverage_even_spacing",
            )
        )
    return selected


def replace_by_filename(selected: list[SelectedRow], target_name: str, replacement: SourceRow, reason: str) -> None:
    for i, row in enumerate(selected):
        if row.filename == target_name:
            selected[i] = SelectedRow(
                index=-1,
                filename=replacement.filename,
                full_path=replacement.full_path,
                parsed_freq_hz=replacement.parsed_freq_hz,
                parsed_amp_db=replacement.parsed_amp_db,
                selection_reason=reason,
            )
            return
    raise ValidationStop(f"Selection replacement target not found: {target_name}")


def build_subset(rows: list[SourceRow]) -> list[SelectedRow]:
    low = [r for r in rows if r.parsed_freq_hz is not None and r.parsed_freq_hz <= 500.0]
    mid = [r for r in rows if r.parsed_freq_hz is not None and 500.0 < r.parsed_freq_hz < 1000.0]
    high = [r for r in rows if r.parsed_freq_hz is not None and r.parsed_freq_hz >= 1000.0]

    selected = []
    selected.extend(select_evenly(low, "low"))
    selected.extend(select_evenly(mid, "mid"))
    selected.extend(select_evenly(high, "high"))

    by_name = {r.filename: r for r in rows}

    replace_by_filename(
        selected,
        "0345Hz_90.0db.png",
        by_name["0360Hz_89.0db.png"],
        "low_amp_outlier_89_substitution_near_even_spacing_anchor",
    )
    replace_by_filename(
        selected,
        "0786Hz_90.0db.png",
        by_name["0735Hz_91.0db.png"],
        "mid_amp_outlier_91_substitution_near_even_spacing_anchor",
    )
    replace_by_filename(
        selected,
        "0891Hz_90.0db.png",
        by_name["0880Hz_89.0db.png"],
        "mid_amp_outlier_89_substitution_near_even_spacing_anchor",
    )
    replace_by_filename(
        selected,
        "1130Hz_90.0db.png",
        by_name["1120Hz_91.0db.png"],
        "high_amp_outlier_91_substitution_near_even_spacing_anchor",
    )
    replace_by_filename(
        selected,
        "1340Hz_90.0db_1335.png",
        by_name["1330Hz_89.0db.png"],
        "high_amp_outlier_89_substitution_near_even_spacing_anchor",
    )
    replace_by_filename(
        selected,
        "1460Hz_90.0db.png",
        by_name["1450Hz_89.0db.png"],
        "high_amp_outlier_89_substitution_near_even_spacing_anchor",
    )

    deduped: list[SelectedRow] = []
    seen = set()
    for idx, row in enumerate(selected, start=1):
        if row.filename in seen:
            raise ValidationStop(f"Deterministic subset produced duplicate selection: {row.filename}")
        seen.add(row.filename)
        deduped.append(
            SelectedRow(
                index=idx,
                filename=row.filename,
                full_path=row.full_path,
                parsed_freq_hz=row.parsed_freq_hz,
                parsed_amp_db=row.parsed_amp_db,
                selection_reason=row.selection_reason,
            )
        )
    return deduped


def choose_output_root() -> Path:
    if not BASE_OUTPUT_ROOT.exists():
        return BASE_OUTPUT_ROOT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BASE_OUTPUT_ROOT.parent / f"{BASE_OUTPUT_ROOT.name}_{stamp}"


def write_subset_files(selected: list[SelectedRow], output_root: Path) -> tuple[Path, Path, Path]:
    inputs_dir = output_root / "inputs"
    subset_dir = inputs_dir / "subset_clean"
    subset_dir.mkdir(parents=True, exist_ok=False)

    txt_path = inputs_dir / "subset_sources.txt"
    csv_path = inputs_dir / "subset_sources.csv"

    with open(txt_path, "w", encoding="utf-8") as txt:
        for row in selected:
            txt.write(f"{row.full_path}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "filename", "full_path", "parsed_freq_hz", "parsed_amp_db", "selection_reason"],
        )
        writer.writeheader()
        for row in selected:
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
            shutil.copy2(row.full_path, subset_dir / row.filename)
    return txt_path, csv_path, subset_dir


def build_commands(output_root: Path, subset_dir: Path) -> tuple[list[str], list[str]]:
    run_v31 = output_root / "run_v31"
    run_v32 = output_root / "run_v32"
    python_exe = sys.executable

    v31_cmd = [
        python_exe,
        str(V31_PATH),
        "--input",
        str(subset_dir),
        "--output",
        str(run_v31),
        "--seed",
        str(SEED),
        "--speckle-k",
        str(SPECKLE_K),
        "--poisson-peak",
        str(POISSON_PEAK),
        "--gauss-sigma",
        str(GAUSS_SIGMA),
        "--match",
        MATCH_MODE,
        "--out-bitdepth",
        str(OUT_BITDEPTH),
        "--out-format",
        OUT_FORMAT,
        "--export-metrics",
        str(run_v31 / "metrics.csv"),
        "--export-summary",
        str(run_v31 / "summary.json"),
    ]

    v32_cmd = [
        python_exe,
        str(V32_PATH),
        "--input",
        str(subset_dir),
        "--output",
        str(run_v32),
        "--seed",
        str(SEED),
        "--speckle-k",
        str(SPECKLE_K),
        "--poisson-peak",
        str(POISSON_PEAK),
        "--gauss-sigma",
        str(GAUSS_SIGMA),
        "--match",
        MATCH_MODE,
        "--out-bitdepth",
        str(OUT_BITDEPTH),
        "--out-format",
        OUT_FORMAT,
        "--conditioning-mode",
        "global",
        "--rng-mode",
        "legacy",
        "--calibration-mode",
        "global",
        "--blur-mode",
        "linear",
        "--export-metrics",
        str(run_v32 / "metrics.csv"),
        "--export-summary",
        str(run_v32 / "summary.json"),
    ]
    return v31_cmd, v32_cmd


def write_commands_file(output_root: Path, v31_cmd: list[str], v32_cmd: list[str]) -> Path:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    commands_path = reports_dir / "commands.txt"
    with open(commands_path, "w", encoding="utf-8") as f:
        f.write("v3.1\n")
        f.write(subprocess.list2cmdline(v31_cmd))
        f.write("\n\nv3.2\n")
        f.write(subprocess.list2cmdline(v32_cmd))
        f.write("\n")
    return commands_path


def write_setup_note(output_root: Path, total_candidates: int, subset_size: int, commands_path: Path) -> Path:
    note_path = output_root / "reports" / "phase1_setup_note.md"
    content = "\n".join(
        [
            "# phase1_setup_note",
            "",
            f"- Source root: `{SOURCE_ROOT}`",
            f"- Candidate clean files scanned: `{total_candidates}`",
            f"- Frozen subset size: `{subset_size}`",
            f"- Seed: `{SEED}`",
            f"- Compared scripts: `{V31_PATH}` and `{V32_PATH}`",
            f"- Frozen global parameters: `speckle_k={SPECKLE_K}`, `poisson_peak={POISSON_PEAK}`, `gauss_sigma={GAUSS_SIGMA}`, `match={MATCH_MODE}`, `bitdepth={OUT_BITDEPTH}`, `format={OUT_FORMAT}`",
            "- Backward-compatible v3.2 mode: `--conditioning-mode global --rng-mode legacy --calibration-mode global --blur-mode linear`, with no input manifest",
            f"- Exact commands file: `{commands_path}`",
        ]
    )
    note_path.write_text(content + "\n", encoding="utf-8")
    return note_path


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


def collect_output_images(root: Path) -> dict[str, Path]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            rel = path.relative_to(root).as_posix()
            out[rel] = path
    return out


def write_structural_summary(output_root: Path, v31_images: dict[str, Path], v32_images: dict[str, Path]) -> tuple[int, int, list[str], list[str]]:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    only_v31 = sorted(set(v31_images) - set(v32_images))
    only_v32 = sorted(set(v32_images) - set(v31_images))
    summary_path = comparison_dir / "structural_comparison.json"
    payload = {
        "v31_output_count": len(v31_images),
        "v32_output_count": len(v32_images),
        "filename_consistent": not only_v31 and not only_v32,
        "missing_in_v32": only_v31,
        "extra_in_v32": only_v32,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(v31_images), len(v32_images), only_v31, only_v32


def compare_hashes(output_root: Path, v31_images: dict[str, Path], v32_images: dict[str, Path]) -> tuple[Path, list[dict[str, str]]]:
    csv_path = output_root / "comparison" / "file_hash_comparison.csv"
    rows: list[dict[str, str]] = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_key", "filename_v31", "filename_v32", "sha256_v31", "sha256_v32", "byte_identical"],
        )
        writer.writeheader()
        for key in sorted(v31_images):
            sha31 = sha256_file(v31_images[key])
            sha32 = sha256_file(v32_images[key])
            row = {
                "image_key": key,
                "filename_v31": key,
                "filename_v32": key,
                "sha256_v31": sha31,
                "sha256_v32": sha32,
                "byte_identical": "true" if sha31 == sha32 else "false",
            }
            writer.writerow(row)
            rows.append(row)
    return csv_path, rows


def compare_pixels(output_root: Path, v31_images: dict[str, Path], v32_images: dict[str, Path]) -> tuple[Path, list[dict[str, str]]]:
    csv_path = output_root / "comparison" / "pixel_comparison.csv"
    rows: list[dict[str, str]] = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_key",
                "max_abs_diff",
                "mean_abs_diff",
                "mse",
                "psnr_between_versions",
                "numeric_equivalence_status",
            ],
        )
        writer.writeheader()
        for key in sorted(v31_images):
            arr31 = np.asarray(Image.open(v31_images[key]), dtype=np.float32)
            arr32 = np.asarray(Image.open(v32_images[key]), dtype=np.float32)
            if arr31.shape != arr32.shape:
                raise ValidationStop(f"Output shape mismatch for {key}: {arr31.shape} vs {arr32.shape}")
            diff = np.abs(arr31 - arr32)
            max_abs = float(diff.max()) if diff.size else 0.0
            mean_abs = float(diff.mean()) if diff.size else 0.0
            mse = float(np.mean((arr31 - arr32) ** 2)) if diff.size else 0.0
            psnr = math.inf if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            if max_abs == 0.0 and mse == 0.0:
                status = "exact"
            elif max_abs <= 1.0 and mean_abs <= 0.01 and (math.isinf(psnr) or psnr >= 60.0):
                status = "negligible"
            else:
                status = "material"
            row = {
                "image_key": key,
                "max_abs_diff": f"{max_abs:.6f}",
                "mean_abs_diff": f"{mean_abs:.6f}",
                "mse": f"{mse:.12f}",
                "psnr_between_versions": "inf" if math.isinf(psnr) else f"{psnr:.6f}",
                "numeric_equivalence_status": status,
            }
            writer.writerow(row)
            rows.append(row)
    return csv_path, rows


def write_report(
    output_root: Path,
    script_hashes: dict[str, str],
    total_candidates: int,
    selected: list[SelectedRow],
    commands_path: Path,
    v31_cmd: list[str],
    v32_cmd: list[str],
    v31_count: int,
    v32_count: int,
    only_v31: list[str],
    only_v32: list[str],
    hash_rows: list[dict[str, str]],
    pixel_rows: list[dict[str, str]],
) -> Path:
    report_path = output_root / "reports" / "backward_compatibility_report.md"
    byte_identical_count = sum(row["byte_identical"] == "true" for row in hash_rows)
    total_pairs = len(hash_rows)
    max_abs = max(float(row["max_abs_diff"]) for row in pixel_rows) if pixel_rows else 0.0
    mean_abs = sum(float(row["mean_abs_diff"]) for row in pixel_rows) / len(pixel_rows) if pixel_rows else 0.0
    worst = max(pixel_rows, key=lambda row: float(row["max_abs_diff"])) if pixel_rows else None

    if only_v31 or only_v32:
        verdict = "v3.2 does not currently preserve the v3.1 baseline path in backward-compatible global mode."
    elif byte_identical_count == total_pairs:
        verdict = "v3.2 reproduces the v3.1 baseline outputs exactly in backward-compatible global mode."
    elif all(row["numeric_equivalence_status"] in {"exact", "negligible"} for row in pixel_rows):
        verdict = "v3.2 preserves the v3.1 baseline noise chain up to numerically negligible differences in backward-compatible global mode."
    else:
        verdict = "v3.2 does not currently preserve the v3.1 baseline path in backward-compatible global mode."

    lines = [
        "# backward_compatibility_report",
        "",
        "## 1. Objective",
        "The objective was backward-compatibility validation of the frozen `make_pseudo_noisy_plus_v3_2.py` against the frozen `make_pseudo_noisy_plus_v3_1.py`, restricted to the legacy/global path.",
        "",
        "## 2. Frozen Inputs",
        f"- v3.1 path: `{V31_PATH}`",
        f"- v3.1 sha256: `{script_hashes['v31_sha256']}`",
        f"- v3.2 path: `{V32_PATH}`",
        f"- v3.2 sha256: `{script_hashes['v32_sha256']}`",
        f"- Source root: `{SOURCE_ROOT}`",
        f"- Subset size: `{len(selected)}`",
        f"- Seed: `{SEED}`",
        f"- Exact commands file: `{commands_path}`",
        f"- Exact v3.1 command: `{subprocess.list2cmdline(v31_cmd)}`",
        f"- Exact v3.2 command: `{subprocess.list2cmdline(v32_cmd)}`",
        "",
        "## 3. Subset Selection",
        f"- Candidate files scanned: `{total_candidates}`",
        f"- Selected files: `{len(selected)}`",
        "- Coverage rationale: deterministic three-band frequency coverage (`low <= 500 Hz`, `mid 500-1000 Hz`, `high >= 1000 Hz`) with even-spacing anchors plus explicit inclusion of rare SPL outliers (`89 dB`, `91 dB`, `95 dB`) when available.",
        "",
        "## 4. Structural Output Comparison",
        f"- Total outputs v3.1: `{v31_count}`",
        f"- Total outputs v3.2: `{v32_count}`",
        f"- Filename consistency: `{'yes' if not only_v31 and not only_v32 else 'no'}`",
        f"- Missing in v3.2: `{len(only_v31)}`",
        f"- Extra in v3.2: `{len(only_v32)}`",
        "",
        "## 5. Byte-Level Comparison",
        f"- Compared output pairs: `{total_pairs}`",
        f"- Byte-identical pairs: `{byte_identical_count}`",
        f"- Byte-identical percentage: `{(100.0 * byte_identical_count / total_pairs) if total_pairs else 0.0:.2f}%`",
        "",
        "## 6. Numeric Comparison",
        f"- Max observed deviation: `{max_abs:.6f}`",
        f"- Mean deviation across files: `{mean_abs:.6f}`",
        f"- Worst-case file: `{worst['image_key'] if worst else 'N/A'}`",
        f"- Numeric interpretation: `{'all differences exact/negligible' if pixel_rows and all(row['numeric_equivalence_status'] in {'exact', 'negligible'} for row in pixel_rows) else 'material differences present'}`",
        "",
        "## 7. Interpretation",
        "This phase did not evaluate denoising quality. It tested whether the v3.2 legacy/global path preserves the v3.1 baseline generator behavior on the same frozen subset and the same frozen global parameters.",
        "",
        "## 8. Final Verdict",
        verdict,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    script_hashes = ensure_identical_scripts()
    rows = scan_source_universe()
    selected = build_subset(rows)
    output_root = choose_output_root()
    inputs_txt, inputs_csv, subset_dir = write_subset_files(selected, output_root)
    v31_cmd, v32_cmd = build_commands(output_root, subset_dir)
    commands_path = write_commands_file(output_root, v31_cmd, v32_cmd)
    write_setup_note(output_root, len(rows), len(selected), commands_path)

    run_command(v31_cmd, output_root)
    run_command(v32_cmd, output_root)

    v31_images = collect_output_images(output_root / "run_v31")
    v32_images = collect_output_images(output_root / "run_v32")
    v31_count, v32_count, only_v31, only_v32 = write_structural_summary(output_root, v31_images, v32_images)
    if only_v31 or only_v32:
        raise ValidationStop(
            "Structural mismatch detected before byte/pixel comparison:\n"
            f"missing_in_v32={only_v31}\nextra_in_v32={only_v32}"
        )

    hash_csv, hash_rows = compare_hashes(output_root, v31_images, v32_images)
    pixel_csv, pixel_rows = compare_pixels(output_root, v31_images, v32_images)
    report_path = write_report(
        output_root,
        script_hashes,
        len(rows),
        selected,
        commands_path,
        v31_cmd,
        v32_cmd,
        v31_count,
        v32_count,
        only_v31,
        only_v32,
        hash_rows,
        pixel_rows,
    )

    byte_identical_count = sum(row["byte_identical"] == "true" for row in hash_rows)
    max_abs = max(float(row["max_abs_diff"]) for row in pixel_rows) if pixel_rows else 0.0
    if byte_identical_count == len(hash_rows):
        verdict = "v3.2 reproduces the v3.1 baseline outputs exactly in backward-compatible global mode."
    elif all(row["numeric_equivalence_status"] in {"exact", "negligible"} for row in pixel_rows):
        verdict = "v3.2 preserves the v3.1 baseline noise chain up to numerically negligible differences in backward-compatible global mode."
    else:
        verdict = "v3.2 does not currently preserve the v3.1 baseline path in backward-compatible global mode."

    result = {
        "output_root": str(output_root),
        "subset_sources_csv": str(inputs_csv),
        "file_hash_comparison_csv": str(hash_csv),
        "pixel_comparison_csv": str(pixel_csv),
        "backward_compatibility_report_md": str(report_path),
        "pairs_compared": len(hash_rows),
        "byte_identical_pairs": byte_identical_count,
        "max_observed_deviation": max_abs,
        "final_verdict": verdict,
        "subset_sources_txt": str(inputs_txt),
        "commands_txt": str(commands_path),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValidationStop as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
