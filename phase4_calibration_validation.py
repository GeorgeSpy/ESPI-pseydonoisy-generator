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


PHASE1_ROOT = _OUT / "v3_2_phase1_backward_compat"
PHASE4_ROOT = _OUT / "v3_2_phase4_calibration_modes"
V32_PATH = _REPO / "make_pseudo_noisy_plus_v3_2.py"
CALIB_SINGLE = _DATA / "wood_real_A" / "W01_ESPI_90db"
CALIB_AVG = _DATA / "wood_Averaged" / "W01_ESPI_90db-Averaged"


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
    calib_max_pairs: int


@dataclass(frozen=True)
class Phase1SubsetRow:
    index: int
    filename: str
    full_path: Path
    parsed_freq_hz: float
    parsed_amp_db: float
    selection_reason: str


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


def load_phase1_subset() -> list[Phase1SubsetRow]:
    subset_csv = PHASE1_ROOT / "inputs" / "subset_sources.csv"
    if not subset_csv.exists():
        raise ValidationStop(f"Missing Phase 1 subset CSV: {subset_csv}")
    rows: list[Phase1SubsetRow] = []
    with subset_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row["parsed_freq_hz"] or not row["parsed_amp_db"]:
                raise ValidationStop(f"Phase 1 subset row missing parsed metadata: {row}")
            full_path = Path(row["full_path"])
            if not full_path.exists():
                raise ValidationStop(f"Phase 1 source image no longer exists: {full_path}")
            rows.append(
                Phase1SubsetRow(
                    index=int(row["index"]),
                    filename=row["filename"],
                    full_path=full_path.resolve(),
                    parsed_freq_hz=float(row["parsed_freq_hz"]),
                    parsed_amp_db=float(row["parsed_amp_db"]),
                    selection_reason=row["selection_reason"],
                )
            )
    if not rows:
        raise ValidationStop("Phase 1 subset CSV is empty")
    return rows


def load_phase1_config() -> FrozenConfig:
    manifest_path = PHASE1_ROOT / "run_v32" / "run_manifest_final.json"
    if not manifest_path.exists():
        raise ValidationStop(f"Missing Phase 1 v3.2 manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    args = payload.get("resolved_global_args")
    if not isinstance(args, dict):
        raise ValidationStop("Phase 1 v3.2 manifest missing resolved_global_args")
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
        calib_max_pairs=int(args.get("calib_max_pairs", 64)),
    )


def regime_from_freq(freq_hz: float) -> str:
    return "mid" if freq_hz <= 680.0 else "high"


def prepare_root() -> None:
    if PHASE4_ROOT.exists():
        raise ValidationStop(f"Phase 4 output root already exists, refusing to mix artifacts: {PHASE4_ROOT}")
    for subdir in [
        PHASE4_ROOT / "inputs",
        PHASE4_ROOT / "run_global",
        PHASE4_ROOT / "run_by_regime",
        PHASE4_ROOT / "comparison",
        PHASE4_ROOT / "reports",
    ]:
        subdir.mkdir(parents=True, exist_ok=False)


def build_aligned_name(original_filename: str) -> str:
    return f"{Path(original_filename).stem}_00.png"


def verify_calibration_alignment(rows: list[Phase1SubsetRow]) -> None:
    missing: list[str] = []
    for row in rows:
        stem = Path(row.filename).stem
        single_path = CALIB_SINGLE / stem / f"{stem}_00.png"
        avg_path = CALIB_AVG / f"{stem}.png"
        if not single_path.exists() or not avg_path.exists():
            missing.append(f"{row.filename} -> single={single_path.exists()} avg={avg_path.exists()}")
    if missing:
        preview = "\n".join(missing[:10])
        raise ValidationStop(f"Calibration asset alignment failed for Phase 4 subset:\n{preview}")


def write_phase4_inputs(rows: list[Phase1SubsetRow]) -> tuple[Path, Path, Path]:
    inputs_dir = PHASE4_ROOT / "inputs"
    subset_dir = inputs_dir / "subset_clean_aligned"
    subset_dir.mkdir(parents=True, exist_ok=False)
    subset_csv = inputs_dir / "phase4_subset_sources.csv"
    manifest_csv = inputs_dir / "phase4_input_manifest.csv"
    subset_txt = inputs_dir / "phase4_subset_sources.txt"

    with subset_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "index",
                "original_filename",
                "phase4_image_path",
                "original_full_path",
                "parsed_freq_hz",
                "parsed_amp_db",
                "regime",
                "selection_reason",
                "alignment_note",
            ],
        )
        writer.writeheader()
        for row in rows:
            aligned_name = build_aligned_name(row.filename)
            shutil.copy2(row.full_path, subset_dir / aligned_name)
            writer.writerow(
                {
                    "index": row.index,
                    "original_filename": row.filename,
                    "phase4_image_path": aligned_name,
                    "original_full_path": str(row.full_path),
                    "parsed_freq_hz": f"{row.parsed_freq_hz:.1f}",
                    "parsed_amp_db": f"{row.parsed_amp_db:.1f}",
                    "regime": regime_from_freq(row.parsed_freq_hz),
                    "selection_reason": row.selection_reason,
                    "alignment_note": "Phase 1 image content copied under _00 single-shot-aligned key for calibration filename matching",
                }
            )

    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_path", "freq_hz", "amp_db", "regime", "group"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": build_aligned_name(row.filename),
                    "freq_hz": f"{row.parsed_freq_hz:.1f}",
                    "amp_db": f"{row.parsed_amp_db:.1f}",
                    "regime": regime_from_freq(row.parsed_freq_hz),
                    "group": row.selection_reason,
                }
            )

    subset_txt.write_text(
        "".join(f"{row.full_path} -> {build_aligned_name(row.filename)}\n" for row in rows),
        encoding="utf-8",
    )
    return subset_csv, manifest_csv, subset_dir


def build_common_command(input_root: Path, manifest_csv: Path, output_root: Path, cfg: FrozenConfig) -> list[str]:
    return [
        sys.executable,
        str(V32_PATH),
        "--input",
        str(input_root),
        "--output",
        str(output_root),
        "--manifest",
        str(manifest_csv),
        "--conditioning-mode",
        "manifest",
        "--rng-mode",
        "per_image",
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
        "--calib-single",
        str(CALIB_SINGLE),
        "--calib-avg",
        str(CALIB_AVG),
        "--calib-override",
        "--calib-max-pairs",
        str(cfg.calib_max_pairs),
        "--export-metrics",
        str(output_root / "metrics.csv"),
        "--export-summary",
        str(output_root / "summary.json"),
    ]


def verify_run_outputs(run_root: Path, label: str) -> tuple[Path, Path, Path, Path]:
    run_manifest_final = run_root / "run_manifest_final.json"
    per_image_params = run_root / "per_image_params.csv"
    calibration_summary = run_root / "calibration_summary.json"
    calibration_pairs = run_root / "calibration_pairs.csv"
    metrics_csv = run_root / "metrics.csv"
    summary_json = run_root / "summary.json"
    missing = [
        p
        for p in [run_manifest_final, per_image_params, calibration_summary, calibration_pairs, metrics_csv, summary_json]
        if not p.exists()
    ]
    if missing:
        raise ValidationStop(f"{label} run missing required artifacts: {missing}")
    return run_manifest_final, per_image_params, calibration_summary, calibration_pairs


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_row_map(rows: list[dict[str, str]], key_field: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[key_field]
        if key in out:
            raise ValidationStop(f"Duplicate key {key} in {key_field} map")
        out[key] = row
    return out


def compare_output_structure(global_rows: list[dict[str, str]], by_rows: list[dict[str, str]]) -> None:
    global_keys = {row["image_id"] for row in global_rows}
    by_keys = {row["image_id"] for row in by_rows}
    if global_keys != by_keys:
        raise ValidationStop(
            "Per-image ledger mismatch between global and by-regime runs:\n"
            f"missing_in_by_regime={sorted(global_keys - by_keys)}\n"
            f"extra_in_by_regime={sorted(by_keys - global_keys)}"
        )


def write_param_comparison_csv(
    global_rows: list[dict[str, str]],
    by_rows: list[dict[str, str]],
) -> tuple[Path, int, list[str]]:
    out_path = PHASE4_ROOT / "comparison" / "calibration_mode_comparison.csv"
    global_map = build_row_map(global_rows, "image_id")
    by_map = build_row_map(by_rows, "image_id")
    changed_keys: list[str] = []

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_key",
                "regime",
                "global_k",
                "global_peak",
                "global_sigma",
                "byregime_k",
                "byregime_peak",
                "byregime_sigma",
                "global_blur_length_px",
                "byregime_blur_length_px",
                "params_changed",
            ],
        )
        writer.writeheader()
        for key in sorted(global_map):
            g = global_map[key]
            b = by_map[key]
            changed = any(
                float(g[field]) != float(b[field])
                for field in ["speckle_k", "poisson_peak", "gauss_sigma"]
            )
            if changed:
                changed_keys.append(key)
            writer.writerow(
                {
                    "image_key": key,
                    "regime": g["regime"],
                    "global_k": g["speckle_k"],
                    "global_peak": g["poisson_peak"],
                    "global_sigma": g["gauss_sigma"],
                    "byregime_k": b["speckle_k"],
                    "byregime_peak": b["poisson_peak"],
                    "byregime_sigma": b["gauss_sigma"],
                    "global_blur_length_px": g["blur_length_px"],
                    "byregime_blur_length_px": b["blur_length_px"],
                    "params_changed": "yes" if changed else "no",
                }
            )
    return out_path, len(changed_keys), changed_keys


def summarize_pair_rows(pair_rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in pair_rows:
        counts[row["regime"]] = counts.get(row["regime"], 0) + 1
    return counts


def write_calibration_summary_comparison(
    global_summary: dict,
    by_summary: dict,
    global_pairs: list[dict[str, str]],
    by_pairs: list[dict[str, str]],
) -> tuple[Path, bool]:
    out_path = PHASE4_ROOT / "comparison" / "calibration_summary_comparison.md"
    global_pair_counts = summarize_pair_rows(global_pairs)
    by_pair_counts = summarize_pair_rows(by_pairs)
    by_regime_params = by_summary.get("regime_params") or {}
    distinct_regime_sets = len({json.dumps(v, sort_keys=True) for v in by_regime_params.values()}) > 1 if by_regime_params else False

    text = f"""# Calibration Summary Comparison

## Global
- mode: `{global_summary.get('calibration_mode')}`
- pair_count: `{global_summary.get('pair_count')}`
- global_params: `{json.dumps(global_summary.get('global_params'), sort_keys=True)}`
- regime distribution in calibration pairs: `{json.dumps(global_pair_counts, sort_keys=True)}`
- clip_tracker: `{json.dumps(global_summary.get('clip_tracker'), sort_keys=True)}`

## By-Regime
- mode: `{by_summary.get('calibration_mode')}`
- pair_count: `{by_summary.get('pair_count')}`
- global_params: `{json.dumps(by_summary.get('global_params'), sort_keys=True)}`
- regime_params: `{json.dumps(by_regime_params, sort_keys=True)}`
- regime distribution in calibration pairs: `{json.dumps(by_pair_counts, sort_keys=True)}`
- clip_tracker: `{json.dumps(by_summary.get('clip_tracker'), sort_keys=True)}`

## Comparison
- by-regime parameter blocks present: `{len(by_regime_params)}`
- distinct by-regime parameter sets: `{"yes" if distinct_regime_sets else "no"}`
"""
    out_path.write_text(text, encoding="utf-8")
    return out_path, distinct_regime_sets


def mean_of(rows: list[dict[str, str]], field: str) -> float:
    vals = [float(row[field]) for row in rows if row.get(field, "") not in ("", None)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def write_generator_metric_comparison(
    global_rows: list[dict[str, str]],
    by_rows: list[dict[str, str]],
    global_summary: dict,
    by_summary: dict,
) -> Path:
    out_path = PHASE4_ROOT / "comparison" / "generator_metric_comparison.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run",
                "image_count",
                "mean_psnr",
                "mean_ssim",
                "mean_edgef1",
                "mean_mpi",
                "mean_pcs",
                "pair_count",
                "k_clips",
                "peak_clips",
                "sigma_clips",
            ],
        )
        writer.writeheader()
        for label, rows, summary in [
            ("global", global_rows, global_summary),
            ("by_regime", by_rows, by_summary),
        ]:
            clips = summary.get("clip_tracker") or {}
            writer.writerow(
                {
                    "run": label,
                    "image_count": len(rows),
                    "mean_psnr": f"{mean_of(rows, 'psnr'):.6f}",
                    "mean_ssim": f"{mean_of(rows, 'ssim'):.6f}",
                    "mean_edgef1": f"{mean_of(rows, 'edgef1'):.6f}",
                    "mean_mpi": f"{mean_of(rows, 'mpi'):.6f}",
                    "mean_pcs": f"{mean_of(rows, 'pcs'):.6f}",
                    "pair_count": summary.get("pair_count"),
                    "k_clips": clips.get("k_hits"),
                    "peak_clips": clips.get("peak_hits"),
                    "sigma_clips": clips.get("sigma_hits"),
                }
            )
    return out_path


def write_inputs_used(
    subset_csv: Path,
    manifest_csv: Path,
    cmd_global: list[str],
    cmd_by_regime: list[str],
) -> Path:
    out_path = PHASE4_ROOT / "reports" / "phase4_inputs_used.txt"
    lines = [
        f"subset_csv={subset_csv}",
        f"manifest_csv={manifest_csv}",
        f"script={V32_PATH}",
        f"calib_single={CALIB_SINGLE}",
        f"calib_avg={CALIB_AVG}",
        "regime_rule=mid if freq_hz <= 680 else high",
        "subset_alignment_note=Phase 1 image content copied under _00 single-shot-aligned keys for calibration matching",
        f"command_global={subprocess.list2cmdline(cmd_global)}",
        f"command_by_regime={subprocess.list2cmdline(cmd_by_regime)}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_report(
    subset_csv: Path,
    manifest_csv: Path,
    inputs_used_txt: Path,
    global_run_manifest: Path,
    by_run_manifest: Path,
    global_summary_json: Path,
    by_summary_json: Path,
    comparison_csv: Path,
    metric_csv: Path,
    calibration_summary_md: Path,
    images_compared: int,
    changed_count: int,
    distinct_regime_sets: bool,
    global_summary: dict,
    by_summary: dict,
) -> tuple[Path, str]:
    report_path = PHASE4_ROOT / "reports" / "phase4_calibration_report.md"
    by_regime_params = by_summary.get("regime_params") or {}
    if changed_count > 0 and len(by_regime_params) >= 2 and distinct_regime_sets:
        verdict = "v3.2 supports reproducible regime-aware calibration"
    elif len(by_regime_params) >= 2:
        verdict = "v3.2 exposes regime-aware calibration behavior, but with limited practical differentiation"
    else:
        verdict = "v3.2 by-regime calibration is not yet sufficiently supported"

    text = f"""# Phase 4 Calibration Report

## 1. Objective
The objective of Phase 4 was global vs by-regime calibration validation in `v3.2`, using manifest-driven conditioning and deterministic per-image RNG.

## 2. Frozen Inputs
- subset: `{subset_csv}`
- manifest: `{manifest_csv}`
- script path: `{V32_PATH}`
- calibration sources: `single={CALIB_SINGLE}`, `avg={CALIB_AVG}`
- seed: `123`
- global run path: `{PHASE4_ROOT / 'run_global'}`
- by-regime run path: `{PHASE4_ROOT / 'run_by_regime'}`

## 3. Experimental Control
- The same image content, manifest, calibration sources, seed, RNG mode, blur behavior, output format, and generator script were used for both runs.
- The Phase 1 frozen subset content was copied under `_00` single-shot-aligned keys so the current `v3.2` calibration interface could assign manifest regimes to matched calibration singles without patching the script.
- The only experimental variable changed between runs was `--calibration-mode` (`global` vs `by-regime`).
- Full input/command provenance is recorded in `{inputs_used_txt}`.

## 4. Calibration Structure Comparison
- global summary: `{global_summary_json}`
- by-regime summary: `{by_summary_json}`
- comparison note: `{calibration_summary_md}`
- global pair_count: `{global_summary.get('pair_count')}`
- by-regime pair_count: `{by_summary.get('pair_count')}`
- by-regime parameter blocks: `{json.dumps(by_regime_params, sort_keys=True)}`

## 5. Per-Image Parameter Comparison
- compared images: `{images_compared}`
- images with changed resolved params: `{changed_count}`
- comparison CSV: `{comparison_csv}`
- fields compared per image: `speckle_k`, `poisson_peak`, `gauss_sigma`, `blur_length_px`
- regime labels affected resolved calibration parameters: `{"yes" if changed_count > 0 else "no"}`

## 6. Metric Comparison
- aggregate generator metrics CSV: `{metric_csv}`
- global mean PSNR/SSIM/EdgeF1: `{mean_of(load_csv(PHASE4_ROOT / 'run_global' / 'per_image_params.csv'), 'psnr'):.4f}`, `{mean_of(load_csv(PHASE4_ROOT / 'run_global' / 'per_image_params.csv'), 'ssim'):.4f}`, `{mean_of(load_csv(PHASE4_ROOT / 'run_global' / 'per_image_params.csv'), 'edgef1'):.4f}`
- by-regime mean PSNR/SSIM/EdgeF1: `{mean_of(load_csv(PHASE4_ROOT / 'run_by_regime' / 'per_image_params.csv'), 'psnr'):.4f}`, `{mean_of(load_csv(PHASE4_ROOT / 'run_by_regime' / 'per_image_params.csv'), 'ssim'):.4f}`, `{mean_of(load_csv(PHASE4_ROOT / 'run_by_regime' / 'per_image_params.csv'), 'edgef1'):.4f}`
- These metrics are secondary evidence only and are not interpreted here as an optimization contest.

## 7. Interpretation
The Phase 4 claim is methodological: whether `v3.2` exposes a structured and traceable regime-aware calibration path. The primary evidence is the calibration summaries, pair records, and per-image resolved parameter ledger, not small generator-side metric differences.

## 8. Final Verdict
`{verdict}`
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path, verdict


def main() -> None:
    if not V32_PATH.exists():
        raise ValidationStop(f"Missing frozen v3.2 script: {V32_PATH}")
    if not CALIB_SINGLE.exists() or not CALIB_AVG.exists():
        raise ValidationStop("Calibration sources are missing for Phase 4")

    phase1_rows = load_phase1_subset()
    cfg = load_phase1_config()
    verify_calibration_alignment(phase1_rows)
    prepare_root()
    subset_csv, manifest_csv, subset_dir = write_phase4_inputs(phase1_rows)

    run_global = PHASE4_ROOT / "run_global"
    run_by_regime = PHASE4_ROOT / "run_by_regime"
    cmd_global = build_common_command(subset_dir, manifest_csv, run_global, cfg) + ["--calibration-mode", "global"]
    cmd_by_regime = build_common_command(subset_dir, manifest_csv, run_by_regime, cfg) + ["--calibration-mode", "by-regime"]
    inputs_used_txt = write_inputs_used(subset_csv, manifest_csv, cmd_global, cmd_by_regime)

    run_command(cmd_global, Path.cwd())
    run_command(cmd_by_regime, Path.cwd())

    global_run_manifest, global_params_csv, global_summary_json, global_pairs_csv = verify_run_outputs(run_global, "Global")
    by_run_manifest, by_params_csv, by_summary_json, by_pairs_csv = verify_run_outputs(run_by_regime, "By-regime")

    global_rows = load_csv(global_params_csv)
    by_rows = load_csv(by_params_csv)
    global_pairs = load_csv(global_pairs_csv)
    by_pairs = load_csv(by_pairs_csv)
    global_summary = load_json(global_summary_json)
    by_summary = load_json(by_summary_json)

    compare_output_structure(global_rows, by_rows)
    comparison_csv, changed_count, changed_keys = write_param_comparison_csv(global_rows, by_rows)
    calibration_summary_md, distinct_regime_sets = write_calibration_summary_comparison(
        global_summary, by_summary, global_pairs, by_pairs
    )
    metric_csv = write_generator_metric_comparison(global_rows, by_rows, global_summary, by_summary)
    report_path, verdict = write_report(
        subset_csv,
        manifest_csv,
        inputs_used_txt,
        global_run_manifest,
        by_run_manifest,
        global_summary_json,
        by_summary_json,
        comparison_csv,
        metric_csv,
        calibration_summary_md,
        len(global_rows),
        changed_count,
        distinct_regime_sets,
        global_summary,
        by_summary,
    )

    summary = {
        "output_root": str(PHASE4_ROOT),
        "phase4_manifest": str(manifest_csv),
        "global_run_summary": str(global_summary_json),
        "by_regime_run_summary": str(by_summary_json),
        "calibration_mode_comparison_csv": str(comparison_csv),
        "generator_metric_comparison_csv": str(metric_csv),
        "phase4_calibration_report_md": str(report_path),
        "images_compared": len(global_rows),
        "changed_resolved_params": changed_count,
        "changed_keys_preview": changed_keys[:10],
        "final_verdict": verdict,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValidationStop as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
