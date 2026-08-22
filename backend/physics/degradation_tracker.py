"""
EMA-based degradation parameter tracker.

Replaces UKF state estimation for hackathon execution.
Inverts thermodynamic & mechanical physics equations from frame residuals
and applies exponential moving average (EMA) smoothing to estimate:
- k1_cooling: cooling efficiency (nominal 1.0)
- eta_combustion: combustion efficiency (nominal 1.0)
- mu_friction: friction coefficient multiplier (nominal 1.0)
- injector_health: per-cylinder health list [1.0, 1.0, 1.0, 1.0]
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np


class DegradationTracker:
    """
    Tracks slow-varying engine health degradation parameters using residual-driven
    physics inversion and exponential moving average (EMA) filtering.
    """

    def __init__(self, alpha: float = 0.995):
        """
        Args:
            alpha: Smoothing factor (0.995 gives ~200 tick memory horizon at 10Hz = 20s).
        """
        self.alpha = alpha

        # State estimates initialized to nominal healthy engine (1.0)
        self.k1_cooling: float = 1.0
        self.eta_combustion: float = 1.0
        self.mu_friction: float = 1.0
        self.injector_health: np.ndarray = np.ones(4, dtype=np.float64)

    def update(
        self,
        frame_dict: Dict[str, Any],
        expected_dict: Dict[str, Any],
        raw_residuals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Updates degradation parameter estimates based on current frame residuals.

        Args:
            frame_dict: Dictionary containing measured sensor values.
            expected_dict: Dictionary containing expected physics model predictions.
            raw_residuals: Dictionary containing (measured - expected) values.

        Returns:
            Dict containing current smoothed degradation parameter estimates.
        """
        # 1. Invert CHT elevation relative to nominal balance (approx 105°C) to estimate cooling efficiency loss (k1 < 1.0)
        avg_cht = float(np.mean(getattr(frame_dict, "cht_c", frame_dict.get("cht_c", [110.0] * 4))))
        # Nominal cruise CHT is ~105-110°C; severe cooling failure reaches ~135-150°C
        k1_instant = 1.0 - (max(0.0, avg_cht - 108.0) / 45.0)
        k1_instant = float(np.clip(k1_instant, 0.3, 1.0))

        # 2. Invert EGT & Fuel Flow to estimate combustion efficiency & injector health
        egt_vals = getattr(frame_dict, "egt_c", frame_dict.get("egt_c", [680.0] * 4))
        avg_egt = float(np.mean(egt_vals))
        # Elevated overall EGT -> lower combustion efficiency
        eta_instant = 1.0 - (max(0.0, avg_egt - 680.0) / 250.0)
        eta_instant = float(np.clip(eta_instant, 0.4, 1.0))

        # Per-cylinder EGT elevation relative to average -> individual injector clog/health drop
        inj_instant = np.ones(4, dtype=np.float64)
        for i in range(min(4, len(egt_vals))):
            rel_egt = egt_vals[i] - avg_egt
            if rel_egt > 8.0:
                inj_instant[i] = max(0.3, 1.0 - (rel_egt / 60.0))

        # 3. Invert Oil Temp & Vibration to estimate mechanical friction
        oil_temp = float(getattr(frame_dict, "oil_temp_c", frame_dict.get("oil_temp_c", 80.0)))
        vib_g = float(getattr(frame_dict, "vibration_g", frame_dict.get("vibration_g", 1.0)))

        mu_instant = 1.0 + (max(0.0, oil_temp - 85.0) / 30.0) + (max(0.0, vib_g - 1.6) / 3.0)
        mu_instant = float(np.clip(mu_instant, 1.0, 2.5))

        # 4. Apply EMA filter
        self.k1_cooling = self.alpha * self.k1_cooling + (1.0 - self.alpha) * k1_instant
        self.eta_combustion = self.alpha * self.eta_combustion + (1.0 - self.alpha) * eta_instant
        self.mu_friction = self.alpha * self.mu_friction + (1.0 - self.alpha) * mu_instant
        self.injector_health = self.alpha * self.injector_health + (1.0 - self.alpha) * inj_instant

        return self.get_estimate()

    def get_estimate(self) -> Dict[str, Any]:
        """Returns the current degradation parameter estimates matching WebSocket contract schema."""
        return {
            "k1_cooling": round(float(self.k1_cooling), 4),
            "eta_combustion": round(float(self.eta_combustion), 4),
            "mu_friction": round(float(self.mu_friction), 4),
            "injector_health": [round(float(h), 4) for h in self.injector_health],
        }

    def reset(self):
        """Resets tracker to initial nominal state."""
        self.k1_cooling = 1.0
        self.eta_combustion = 1.0
        self.mu_friction = 1.0
        self.injector_health = np.ones(4, dtype=np.float64)
