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
    # W&B is compulsory — warn if key missing (train will fail fast)
    import os
    if not os.environ.get("WANDB_API_KEY") and "--wandb-mode" not in " ".join(cmd):
        print("[train_models] WARN: WANDB_API_KEY not set but W&B is compulsory (mode=online). Training will fail — set Kaggle Secret WANDB_API_KEY or use --wandb-mode offline for debugging.")
    print(f"[train_models] Running: {' '.join(cmd)} (W&B compulsory)")
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
