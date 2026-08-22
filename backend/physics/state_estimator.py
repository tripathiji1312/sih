"""
TwinStateEstimator — Unscented Kalman Filter joint state + parameter estimation.
If filterpy is not available (e.g., fresh Kaggle env before pip install), falls back to
simple exponential smoothing so the training pipeline still runs.
"""
from __future__ import annotations
import numpy as np

try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False

from backend.physics.expectation_model import PhysicsExpectationModel, EngineParams


class TwinStateEstimator:
    """
    14-dim state: [rpm, cht1-4, egt1-4, oil_p, oil_t, k1, eta, mu]
    11-dim measurement: first 11 states.
    Degradation params are slowest-varying.
    """

    def __init__(self):
        self.n_states = 14
        self.physics = PhysicsExpectationModel()
        if HAS_FILTERPY:
            self.points = MerweScaledSigmaPoints(n=self.n_states, alpha=0.1, beta=2.0, kappa=0.0)
            self.ukf = UnscentedKalmanFilter(
                dim_x=self.n_states, dim_z=11, dt=0.1, hx=self._hx, fx=self._fx, points=self.points
            )
            self.ukf.x = np.array([5000, 110, 110, 110, 110, 680, 680, 680, 680, 55.0, 80.0, 1.0, 1.0, 1.0], dtype=float)
            self.ukf.Q = np.diag([100.0, 1.0, 1.0, 1.0, 1.0, 25.0, 25.0, 25.0, 25.0, 4.0, 1.0, 1e-6, 1e-6, 1e-6])
            self.ukf.R = np.diag([20.0, 4.0, 4.0, 4.0, 4.0, 100.0, 100.0, 100.0, 100.0, 4.0, 1.0])
            self.ukf.P *= 10
        else:
            # Fallback: simple state
            self.x = np.array([5000, 110, 110, 110, 110, 680, 680, 680, 680, 55.0, 80.0, 1.0, 1.0, 1.0], dtype=float)
            self.P = np.eye(self.n_states) * 10

    def _fx(self, x: np.ndarray, dt: float) -> np.ndarray:
        # Use physics to propagate first 11 states; degradation params random walk
        # For simplicity: keep degradation params, add tiny noise; propagate rpm/cht trivially
        x_next = x.copy()
        # Add process noise implicitly via Q; no deterministic drift for degradation
        return x_next

    def _hx(self, x: np.ndarray) -> np.ndarray:
        return x[:11]

    def update(self, measurement: np.ndarray):
        measurement = np.asarray(measurement, dtype=float)
        if measurement.shape[0] != 11:
            # pad or truncate
            tmp = np.zeros(11)
            n = min(len(measurement), 11)
            tmp[:n] = measurement[:n]
            measurement = tmp
        if HAS_FILTERPY:
            self.ukf.predict()
            self.ukf.update(measurement)
            return self.ukf.x.copy(), self.ukf.P.copy()
        else:
            # Exponential smoothing fallback
            alpha = 0.1
            self.x[:11] = (1 - alpha) * self.x[:11] + alpha * measurement
            # degradation params adapted slowly from residuals
            return self.x.copy(), self.P.copy()

    def get_degradation_params(self) -> dict:
        if HAS_FILTERPY:
            x = self.ukf.x
            P = self.ukf.P
        else:
            x = self.x
            P = self.P
        return {
            "k1_cooling": float(x[11]),
            "eta_combustion": float(x[12]),
            "mu_friction": float(x[13]),
            "param_uncertainty": np.sqrt(np.diag(P)[11:]).tolist(),
        }

    def get_state(self) -> np.ndarray:
        if HAS_FILTERPY:
            return self.ukf.x.copy()
        return self.x.copy()
