# DG-Hetero-GNN portable review package

Open `DG_Hetero_GNN_Review.ipynb` as the single review entry point. It explains the specification, data integrity checks, leakage controls, architecture, experiment protocol, results, tables, and figures in execution order.

The package includes the paper, four hash-verified processed graph archives, reviewed Python engine, validation tests, full-run artifacts, and pinned dependencies. It does not require the original workspace layout or a network download after dependencies are installed.

## Setup

```powershell
python -m pip install -r requirements-reproduction.txt
```

## One-time smoke validation

```powershell
python -u run_review_once.py 2>&1 | Tee-Object review-smoke.log
```

## One-time complete reproduction

```powershell
python -u run_review_once.py --full --run-id reviewer-full-run 2>&1 | Tee-Object reviewer-full-run.log
```

The full command runs tests first, prints live scenario/model/seed/epoch/GPU progress, writes atomic checkpoints, and resumes when invoked again with the same run ID.

Included completed results are under `artifacts_runs/minimal-diff-notebook/FULL_PAPER_RUN/paper-full-20260830`. The review notebook can rerender all tables and figures from those saved prediction artifacts without retraining.
