#!/usr/bin/env python3
"""
Generate pseudo-noisy datasets (profiles × seeds × instruments)
Uses make_pseudo_noisy_plus.py with --profile and --bitdepth 16
"""

import os, sys, subprocess
from pathlib import Path
import argparse

# --- Python interpreter selection (future-proof) ---
VENV_PY = Path(os.environ.get("ESPI_PY", sys.executable))
if not VENV_PY.exists():
    raise RuntimeError(f"Python interpreter not found: {VENV_PY}")
print(f"[INFO] Using python: {VENV_PY}")

def main():
    ap = argparse.ArgumentParser(description="Generate pseudo-noisy datasets")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = ap.parse_args()

    # === Paths ===
    AVG_PATHS = {
        "W01": "C:\\ESPI\\data\\wood_Averaged\\W01_ESPI_90db-Averaged",
        "W02": "C:\\ESPI\\data\\wood_Averaged\\W02_ESPI_90db-Averaged", 
        "W03": "C:\\ESPI\\data\\wood_Averaged\\W03_ESPI_90db-Averaged"
    }
    
    PSEUDO_ROOT = Path("C:\\ESPI_TEMP\\pseudo")
    METRICS_ROOT = Path("C:\\ESPI_TEMP\\pseudo\\metrics")
    PSEUDO_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)

    # === Profiles & Seeds ===
    PROFILES = ["lite", "mid", "heavy"]
    SEEDS = [1, 2, 3]
    
    # === Generate ===
    total_jobs = len(AVG_PATHS) * len(PROFILES) * len(SEEDS)
    current_job = 0
    
    for inst, avg_path in AVG_PATHS.items():
        if not Path(avg_path).exists():
            print(f"[WARN] Average path not found: {avg_path}")
            continue
            
        for profile in PROFILES:
            for seed in SEEDS:
                current_job += 1
                print(f"\n[{current_job}/{total_jobs}] Generating: {inst} - {profile} - seed {seed}")
                
                # Output paths
                out_dir = PSEUDO_ROOT / f"{inst}" / f"Pseudo_{profile}_s{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)
                metrics_csv = METRICS_ROOT / f"{inst}_{profile}_s{seed}.csv"
                
                # Command
                cmd = [
                    str(VENV_PY),
                    "C:\\ESPI_DnCNN\\make_pseudo_noisy_plus.py",
                    "--input", avg_path,
                    "--output", str(out_dir),
                    "--profile", profile,
                    "--seed", str(seed),
                    "--bitdepth", "16",
                    "--export-metrics", str(metrics_csv)
                ]
                
                print(f"CMD> {' '.join(f'\"{c}\"' if ' ' in c else c for c in cmd)}")
                
                if not args.dry_run:
                    try:
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(f"[OK] Generated: {out_dir}")
                        if result.stdout:
                            print(f"STDOUT: {result.stdout.strip()}")
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] Failed: {e}")
                        if e.stderr:
                            print(f"STDERR: {e.stderr}")
                        continue
                else:
                    print(f"[DRY-RUN] Would create: {out_dir}")
                    print(f"[DRY-RUN] Would create: {metrics_csv}")

    print(f"\n[COMPLETE] Pseudo-noisy generation {'simulated' if args.dry_run else 'completed'}!")
    print(f"Output directory: {PSEUDO_ROOT}")
    print(f"Metrics directory: {METRICS_ROOT}")

if __name__ == "__main__":
    main()
