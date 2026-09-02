"""One-command entry point for the portable reviewer package."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and run the DG-Hetero-GNN reproduction once.")
    parser.add_argument("--full", action="store_true", help="Run the complete 40-epoch/five-seed protocol; default is smoke validation.")
    parser.add_argument("--run-id", default="review-one-time-run")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if not args.skip_tests:
        subprocess.run([sys.executable, "-u", "reproduction_tests.py"], cwd=root, env=env, check=True)
    command = [sys.executable, "-u", "run_full_paper.py", "--run-id", args.run_id,
               "--implementation-id", "portable-review-package"]
    if not args.full:
        command.append("--smoke")
    print("Launching:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
