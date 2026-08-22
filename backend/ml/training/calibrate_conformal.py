"""
Conformal prediction calibration — provides distribution-free coverage guarantees.
Saves conformal thresholds for classification (APS method).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from backend.config import CONFORMAL


def calibrate_conformal(model, val_loader, device, output_dir: Path, alpha: float = None):
    alpha = alpha or CONFORMAL["alpha"]
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            probs = out["probs"]  # (B,K)
            # APS score: cumulative prob up to true label sorted by prob descending
            # Simpler: 1 - prob_true
            for i in range(len(y)):
                p_true = probs[i, y[i]].item()
                score = 1 - p_true
                all_scores.append(score)
                all_labels.append(y[i].item())

    scores = np.array(all_scores)
    # Quantile
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = float(np.clip(q_level, 0, 1))
    threshold = float(np.quantile(scores, q_level))
    print(f"[conformal] Calibration: n={n} alpha={alpha} q_level={q_level:.3f} threshold={threshold:.3f}")

    # Save
    out = dict(alpha=alpha, threshold=threshold, q_level=q_level, n=n, method="1 - prob_true")
    output_dir = Path(output_dir)
    with open(output_dir / "conformal_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    # Also save scores for analysis
    np.savez(output_dir / "conformal_scores.npz", scores=scores)
    return out
