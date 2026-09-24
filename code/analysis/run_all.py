"""Reproduce every Section 10 result of the dataset report (Problems 1 and 3-18).

Usage (from the repository root):  python code/analysis/run_all.py
CPU only, about 50 minutes (the rolling-origin random forests take about 40). Outputs are
written to results/analysis/{csv,tables,figures}. Problem 2 (neural nowcast benchmark) is
reproduced by code/validation_r3.py.
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# order matters: later studies read the CSV files written by earlier ones
STEPS = [
    "study_extremes", "study_flood_risk", "study_bridge", "study_clustering", "study_sufficiency",
    "study_importance", "study_usecases", "study_bridge_impact",
    "figs_hazard", "figs_ml", "figs_redo", "figs_usecases", "figs_bridge_impact", "figs_appendix",
    "make_tables", "make_tables_bridge",
]


def main() -> None:
    for step in STEPS:
        t0 = time.time()
        print(f"[{step}]", flush=True)
        runpy.run_path(str(HERE / f"{step}.py"), run_name="__main__")
        print(f"[{step}] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
