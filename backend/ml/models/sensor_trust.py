"""
Sensor trust evaluator — per-sensor agreement with physics + correlated sensors.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


class SensorTrustEvaluator:
    CORRELATION_GROUPS = {
        "thermal": ["cht_1", "cht_2", "cht_3", "cht_4", "oil_temp", "egt_1"],
        "lubrication": ["oil_pressure", "oil_temp", "rpm"],
        "combustion": ["egt_1", "egt_2", "egt_3", "egt_4", "fuel_flow"],
        "mechanical": ["rpm", "vibration", "oil_pressure"],
    }

    # Residual thresholds for trust (z-score magnitude where trust drops to 0)
    THRESHOLDS = {
        "rpm": 2.5,
        "cht_1": 2.0, "cht_2": 2.0, "cht_3": 2.0, "cht_4": 2.0,
        "egt_1": 2.5, "egt_2": 2.5, "egt_3": 2.5, "egt_4": 2.5,
        "oil_pressure": 2.0, "oil_p": 2.0,
        "oil_temp": 2.0, "oil_t": 2.0,
        "fuel_flow": 2.0,
        "vibration": 2.5,
        "batt_v": 2.0,
        "default": 2.5,
    }

    def _get_threshold(self, sensor: str) -> float:
        return self.THRESHOLDS.get(sensor, self.THRESHOLDS["default"])

    def _get_correlated(self, sensor: str):
        for group, members in self.CORRELATION_GROUPS.items():
            if sensor in members:
                return [m for m in members if m != sensor]
        return []

    def evaluate_trust(self, residuals: Dict[str, float], physics_state: dict = None) -> Dict[str, float]:
        trust_scores: Dict[str, float] = {}
        for sensor, resid in residuals.items():
            correlated = self._get_correlated(sensor)
            thresh = self._get_threshold(sensor)
            physics_agreement = 1.0 - min(1.0, abs(resid) / thresh)
            physics_agreement = float(np.clip(physics_agreement, 0, 1))
            if correlated:
                # find correlated residuals that exist
                corr_vals = [residuals[s] for s in correlated if s in residuals]
                if corr_vals:
                    # if sensor diverges but correlated don't, trust low
                    mean_corr = float(np.mean(corr_vals))
                    corr_agreement = 1.0 - min(1.0, abs(resid - mean_corr) / thresh)
                    corr_agreement = float(np.clip(corr_agreement, 0, 1))
                    # Combine: trust is high only if both agree; if physics agrees but correlated disagrees -> medium
                    # Use min for strict
                    trust = float(physics_agreement * 0.6 + corr_agreement * 0.4)
                else:
                    trust = physics_agreement
            else:
                trust = physics_agreement
            trust_scores[sensor] = float(np.clip(trust, 0, 1))
        return trust_scores

    def arbitrate(self, trust_scores: Dict[str, float], residuals: Dict[str, float]) -> str:
        n_untrusted = sum(1 for t in trust_scores.values() if t < 0.3)
        if n_untrusted == 0:
            # check if any residual large
            if all(abs(r) < 0.7 for r in residuals.values()):
                return "normal"
            return "engine_fault"
        if n_untrusted == 1:
            return "sensor_fault"
        if n_untrusted >= 2:
            if self._physically_consistent(residuals):
                return "engine_fault"
            else:
                return "sensor_fault"
        return "uncertain"

    def _physically_consistent(self, residuals: Dict[str, float]) -> bool:
        # Simple consistency: CHTs should have same sign if cooling fault; EGTs similar
        cht_vals = [residuals.get(f"cht_{i}", 0) for i in range(1, 5)]
        egt_vals = [residuals.get(f"egt_{i}", 0) for i in range(1, 5)]
        # Check sign consistency
        cht_signs = [np.sign(v) for v in cht_vals if abs(v) > 0.5]
        egt_signs = [np.sign(v) for v in egt_vals if abs(v) > 0.5]
        cht_consistent = len(set(cht_signs)) <= 1 if cht_signs else True
        egt_consistent = len(set(egt_signs)) <= 1 if egt_signs else True
        # If majority consistent, engine fault
        return cht_consistent and egt_consistent
