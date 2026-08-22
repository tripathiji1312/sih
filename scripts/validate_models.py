#!/usr/bin/env python3
"""
Validate trained models: sanity checks on all fault types, uncertainty, OOD.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from backend.config import MODELS_ROOT, TRAINING_DATA_ROOT, FAULT_CLASSES, FAULT_TO_IDX
from backend.ml.models.evidential_model import EvidentialFaultClassifier
from backend.ml.models.ood_detector import OODDetector
from backend.ml.inference import EvidentialModelInference

def main():
    print("=== Validate Models ===")
    # Check files exist
    for f in [MODELS_ROOT / "evidential_model.pt", MODELS_ROOT / "evidential_model.onnx", MODELS_ROOT / "ood_stats.npz", MODELS_ROOT / "residual_stats.npz"]:
        print(f"{'OK' if f.exists() else 'MISSING'}: {f} ({f.stat().st_size/1024:.1f} KB)" if f.exists() else f"{'MISSING'}: {f}")

    # Load data for quick test
    window_file = TRAINING_DATA_ROOT / "windows.npz"
    if not window_file.exists():
        print("[validate] No windows.npz — generate data first")
        return
    data = np.load(window_file)
    X, y = data["X"], data["y"]
    print(f"[validate] Data: X={X.shape} y dist={np.bincount(y)}")

    # Inference test
    infer = EvidentialModelInference()
    print(f"[validate] Inference loaded: {infer.loaded}")
    if infer.loaded:
        # Test per-class
        for cls_idx in range(len(FAULT_CLASSES)):
            mask = (y == cls_idx)
            if mask.sum() == 0:
                continue
            sample = X[mask][0]  # (14,30)
            out = infer.infer(sample)
            print(f"  Class {FAULT_CLASSES[cls_idx]:20s} -> pred={out['predicted_label']:20s} fault_prob={out['fault_probability']:.2f} epistemic={out['epistemic_uncertainty']:.3f} conf={out['confidence']:.3f}")

        # Healthy should have low epistemic, faults higher
        # OOD test
        ood_path = TRAINING_DATA_ROOT / "ood_windows.npz"
        if ood_path.exists():
            ood = np.load(ood_path)
            Xo = ood["X"]
            if len(Xo) > 0:
                out_ood = infer.infer(Xo[0])
                print(f"  OOD sample -> pred={out_ood['predicted_label']} epistemic={out_ood['epistemic_uncertainty']:.3f} (should be high)")

    # OOD detector test
    det = OODDetector()
    if det._loaded:
        # healthy vs fault
        healthy = X[y == FAULT_TO_IDX["healthy"]]
        fault = X[y != FAULT_TO_IDX["healthy"]]
        if len(healthy) > 0 and len(fault) > 0:
            hs = det.batch_scores(healthy[:200])
            fs = det.batch_scores(fault[:200])
            print(f"[validate] OOD scores: healthy mean={hs.mean():.2f} fault mean={fs.mean():.2f}")
            # AUROC
            from backend.ml.training.metrics import ood_auroc
            print(f"[validate] Fault vs healthy AUROC: {ood_auroc(hs, fs):.3f}")
        print(f"[validate] OOD threshold={det.threshold:.2f}")

    print("=== Done ===")

if __name__ == "__main__":
    main()
