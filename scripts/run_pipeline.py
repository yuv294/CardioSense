"""Run the whole pipeline for one or both cohorts.

    python run_pipeline.py                     # both cohorts, everything
    python run_pipeline.py --cohort nhanes     # one cohort
    python run_pipeline.py --skip-nested       # skip the slow nested CV

Order: prepare -> train -> evaluate -> export -> ablation -> nested CV (+ re-export) -> figures.
NHANES raw files are downloaded first if missing.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from config import COHORTS, DATA_RAW

HERE = Path(__file__).resolve().parent


def step(*args):
    t = time.time()
    print(f"\n>>> python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", *args], cwd=HERE, check=True)
    print(f"<<< done in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORTS)
    ap.add_argument("--skip-nested", action="store_true")
    a = ap.parse_args()
    cohorts = [a.cohort] if a.cohort else COHORTS
    if "nhanes" in cohorts and not (DATA_RAW / "nhanes" / "mort").exists():
        step("nhanes_download.py")
    for c in cohorts:
        for s in ("prepare.py", "train.py", "evaluate.py", "export.py", "ablation.py"):
            step(s, "--cohort", c)
    if not a.skip_nested:
        for c in cohorts:
            step("nested_cv.py", "--cohort", c)
            step("export.py", "--cohort", c)     # adds the nested-CV summary for the app
    step("figures.py")


if __name__ == "__main__":
    main()
