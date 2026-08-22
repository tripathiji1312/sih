#!/usr/bin/env python3
"""
Wrapper script: generate all training data.
Usage:
  python scripts/generate_all_data.py --quick      # smoke test (2 min missions)
  python scripts/generate_all_data.py              # fast default (10 min missions)
  python scripts/generate_all_data.py --full       # full 60 min missions (Kaggle)
  python scripts/generate_all_data.py --full --seed 42
"""
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml.training.generate_training_data import generate_all
from backend.config import TRAINING_DATA_ROOT, MODELS_ROOT

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration-min", type=float, default=None)
    args = p.parse_args()

    if args.quick:
        scenarios = {
            "healthy": {"count": 2, "duration_min": 2, "profiles": ["cruise"], "altitudes": [3000], "ambient_temps": [15]},
            "cooling_degradation": {"count": 2, "severities": [0.5], "onset": "random"},
            "lubrication_fault": {"count": 1, "severities": [0.4], "onset": "random"},
            "misfire_single_cyl": {"count": 2, "severities": [0.5], "onset": "random"},
            "injector_clog": {"count": 1, "severities": [0.4], "onset": "random"},
            "sensor_faults": {"count": 1, "types": ["drift"], "sensors": ["oil_pressure"]},
            "ood_scenarios": {"count": 1, "conditions": ["combined_faults"]},
        }
        generate_all(scenarios=scenarios, seed=args.seed, duration_override_min=args.duration_min)
    else:
        generate_all(full=args.full, seed=args.seed, duration_override_min=args.duration_min)
