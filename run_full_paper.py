"""Explicit, resumable launcher for the five-seed, 40-epoch protocol."""
import argparse
import importlib.util
import os
import sys

if importlib.util.find_spec("xgboost") is None:
    raise RuntimeError("XGBoost is required. Install requirements-reproduction.txt before FULL_PAPER_RUN.")

from paper_reproduction_core import Config, run_suite

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True, help="Stable identifier; reuse it to resume the same run")
parser.add_argument("--implementation-id", default="minimal-diff-notebook")
parser.add_argument("--smoke", action="store_true", help="Run the short non-paper-valid preflight suite")
args = parser.parse_args()
mode = "SMOKE_TEST" if args.smoke else "FULL_PAPER_RUN"
os.environ["DG_HETERO_RUN_MODE"] = mode
results = run_suite(cfg=Config(run_mode=mode), run_id=args.run_id, implementation_id=args.implementation_id)
print(f"{mode} complete: {len(results)} aggregate result files", flush=True)
