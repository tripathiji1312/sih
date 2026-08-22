#!/usr/bin/env python3
"""
Train all models: evidential classifier + OOD detector.
Usage:
  python scripts/train_models.py --quick           # smoke test 3 epochs
  python scripts/train_models.py                   # default 60 epochs
  python scripts/train_models.py --epochs 100 --lr 5e-4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--full", action="store_true")
    args, unknown = p.parse_known_args()

    # Build train command — pass through any extra args (e.g. --wandb --wandb-project ...)
    cmd = [sys.executable, "-m", "backend.ml.training.train_evidential"]
    if args.quick:
        cmd.append("--quick")
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.lr:
        cmd += ["--lr", str(args.lr)]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    cmd += unknown
    # If WANDB_API_KEY is set in env, auto-enable wandb unless explicitly disabled
    import os
    if os.environ.get("WANDB_API_KEY") and "--wandb" not in cmd and "--no-wandb" not in cmd and "--wandb-mode" not in " ".join(cmd):
        # Do not auto-add — let user decide; but hint
        print("[train_models] Tip: WANDB_API_KEY detected — add --wandb to push artifacts to W&B")
    print(f"[train_models] Running: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    if ret != 0:
        sys.exit(ret)

    # Then calibrate OOD
    print("[train_models] Calibrating OOD detector...")
    cmd2 = [sys.executable, "-m", "backend.ml.training.calibrate_ood"]
    ret2 = subprocess.call(cmd2)
    if ret2 != 0:
        print("[train_models] OOD calibration failed, but training succeeded")
    else:
        print("[train_models] All done.")
