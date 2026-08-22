"""
OOD detector — Mahalanobis distance from healthy training distribution.
Fitted on healthy residual windows (flattened to vector).
Provides threshold at 99th percentile + optional PCA.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

from backend.config import MODELS_ROOT, OOD


class OODDetector:
    def __init__(self, stats_path: Path = None):
        self.stats_path = stats_path or (MODELS_ROOT / "ood_stats.npz")
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: float = 100.0
        self.pca_components: Optional[np.ndarray] = None  # (orig_dim, k)
        self.pca_mean: Optional[np.ndarray] = None
        self._loaded = False
        self.try_load()

    def try_load(self):
        try:
            if self.stats_path.exists():
                data = np.load(self.stats_path)
                self.mean = data["mean"]
                self.cov_inv = data["cov_inv"]
                self.threshold = float(data["threshold"])
                if "pca_components" in data:
                    self.pca_components = data["pca_components"]
                    self.pca_mean = data["pca_mean"]
                self._loaded = True
        except Exception:
            pass

    def fit(self, X_healthy: np.ndarray, threshold_percentile: float = 99.0, shrinkage: float = 1e-4, pca_components: int = None) -> dict:
        """
        X_healthy: (N, 14, 30) or (N, D) — healthy windows
        Returns stats dict and saves to disk.
        """
        if X_healthy.ndim == 3:
            X_flat = X_healthy.reshape(len(X_healthy), -1)  # (N, 420)
        else:
            X_flat = X_healthy

        self.pca_mean = None
        self.pca_components = None

        if pca_components is not None and pca_components < X_flat.shape[1]:
            # PCA via SVD
            self.pca_mean = X_flat.mean(axis=0)
            X_centered = X_flat - self.pca_mean
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.pca_components = Vt[:pca_components].T  # (D, k)
            X_flat = X_centered @ self.pca_components  # (N, k)

        mean = X_flat.mean(axis=0)
        cov = np.cov(X_flat.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # shrinkage regularization
        cov = cov + shrinkage * np.eye(cov.shape[0])
        # Use pinv for stability
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        distances = np.array([self._mahalanobis(x, mean, cov_inv) for x in X_flat])
        threshold = float(np.percentile(distances, threshold_percentile))

        self.mean = mean
        self.cov_inv = cov_inv
        self.threshold = threshold
        self._loaded = True

        # Save
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = dict(mean=mean, cov_inv=cov_inv, threshold=np.array(threshold))
        if self.pca_components is not None:
            save_dict["pca_components"] = self.pca_components
            save_dict["pca_mean"] = self.pca_mean
        np.savez(self.stats_path, **save_dict)

        return {"mean": mean, "cov_inv": cov_inv, "threshold": threshold, "distances": distances}

    def _mahalanobis(self, x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> float:
        diff = x - mean
        return float(np.sqrt(diff @ cov_inv @ diff))

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x.reshape(-1)
        if self.pca_components is not None:
            x = (x - self.pca_mean) @ self.pca_components
        return x

    def score(self, x: np.ndarray) -> float:
        """Mahalanobis distance score."""
        if not self._loaded or self.mean is None:
            return 0.0
        x = self._prepare(np.asarray(x, dtype=float))
        return self._mahalanobis(x, self.mean, self.cov_inv)

    def check(self, x: np.ndarray) -> dict:
        """
        x: residual vector (14,) or window flattened (420,)
        Returns dict with distance, is_ood, confidence.
        """
        dist = self.score(x)
        is_ood = dist > self.threshold
        # confidence proxy: 1 - sigmoid((dist - threshold)/ scale)
        # simple linear mapping
        if self.threshold > 0:
            # normalize: distance / threshold
            ratio = dist / (self.threshold + 1e-6)
            confidence = float(np.clip(1.0 - (ratio - 0.8) * 2.0, 0.0, 1.0)) if is_ood else 1.0
        else:
            confidence = 0.5
        return {
            "mahalanobis_distance": float(dist),
            "threshold": float(self.threshold),
            "is_ood": bool(is_ood),
            "confidence": float(confidence),
        }

    def batch_scores(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        return np.array([self.score(x) for x in X])
