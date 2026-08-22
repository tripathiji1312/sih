"""
OOD calibration — fits Mahalanobis detector on healthy windows.
Run standalone or called from train pipeline.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from backend.config import TRAINING_DATA_ROOT, MODELS_ROOT, OOD
from backend.ml.models.ood_detector import OODDetector


def main():
    p = argparse.ArgumentParser(description="Calibrate OOD detector")
    p.add_argument("--data", type=str, default=str(TRAINING_DATA_ROOT / "windows.npz"))
    p.add_argument("--output", type=str, default=str(MODELS_ROOT / "ood_stats.npz"))
    p.add_argument("--ood-data", type=str, default=str(TRAINING_DATA_ROOT / "ood_windows.npz"))
    p.add_argument("--threshold", type=float, default=OOD["threshold_percentile"])
    p.add_argument("--shrinkage", type=float, default=OOD["shrinkage"])
    args = p.parse_args()

    data_path = Path(args.data)
    if data_path.is_dir():
        data_path = data_path / "windows.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}. Generate first.")

    data = np.load(data_path)
    X = data["X"]  # (N,14,30)
    y = data["y"]
    # Only healthy for fitting
    from backend.config import FAULT_TO_IDX
    healthy_mask = (y == FAULT_TO_IDX["healthy"])
    X_healthy = X[healthy_mask]
    print(f"[ood] Healthy windows for fitting: {len(X_healthy)} / {len(X)}")

    if len(X_healthy) < 10:
        raise ValueError("Not enough healthy windows to fit OOD detector")

    detector = OODDetector(stats_path=Path(args.output))
    stats = detector.fit(X_healthy, threshold_percentile=args.threshold, shrinkage=args.shrinkage, pca_components=OOD.get("pca_components"))

    print(f"[ood] Fitted: mean_shape={stats['mean'].shape} threshold={stats['threshold']:.3f}")
    print(f"[ood] Saved to {args.output}")

    # Evaluate AUROC if OOD data exists
    ood_path = Path(args.ood_data)
    if ood_path.exists():
        ood_data = np.load(ood_path)
        X_ood = ood_data["X"]
        # Score both
        healthy_scores = detector.batch_scores(X_healthy[: min(len(X_healthy), len(X_ood))])
        ood_scores = detector.batch_scores(X_ood)
        from backend.ml.training.metrics import ood_auroc
        auroc = ood_auroc(healthy_scores, ood_scores)
        print(f"[ood] Healthy mean score: {healthy_scores.mean():.3f} ± {healthy_scores.std():.3f}")
        print(f"[ood] OOD mean score: {ood_scores.mean():.3f} ± {ood_scores.std():.3f}")
        print(f"[ood] AUROC (healthy vs OOD): {auroc:.3f}")

        # Also check fault detection as OOD? Use non-healthy as quasi-OOD
        non_healthy = X[~healthy_mask]
        if len(non_healthy) > 0:
            fault_scores = detector.batch_scores(non_healthy[: min(len(non_healthy), 500)])
            print(f"[ood] Fault mean score: {fault_scores.mean():.3f} (should be > healthy)")

    print("[ood] Done.")


if __name__ == "__main__":
    main()
