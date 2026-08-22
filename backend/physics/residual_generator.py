"""
Residual generator: measured − expected → normalized 14-dim vector.
Handles z-score normalization using stats from healthy training data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from backend.config import RESIDUAL_CHANNELS, RESIDUAL_STATS_FILE
from backend.physics.expectation_model import PhysicsExpectationModel, EngineParams
from backend.simulator.engine_simulator import SensorFrame


@dataclass
class ResidualVector:
    timestamp: float
    # raw residuals
    rpm_residual: float
    cht_residuals: list[float]
    egt_residuals: list[float]
    oil_p_residual: float
    oil_t_residual: float
    fuel_flow_residual: float
    vibration_residual: float
    batt_v_residual: float
    # normalized 14-dim
    normalized: list[float]

    def to_vector(self) -> np.ndarray:
        return np.array(self.normalized, dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "rpm": self.rpm_residual,
            "cht_1": self.cht_residuals[0],
            "cht_2": self.cht_residuals[1],
            "cht_3": self.cht_residuals[2],
            "cht_4": self.cht_residuals[3],
            "egt_1": self.egt_residuals[0],
            "egt_2": self.egt_residuals[1],
            "egt_3": self.egt_residuals[2],
            "egt_4": self.egt_residuals[3],
            "oil_p": self.oil_p_residual,
            "oil_t": self.oil_t_residual,
            "fuel_flow": self.fuel_flow_residual,
            "vibration": self.vibration_residual,
            "batt_v": self.batt_v_residual,
        }


class ResidualGenerator:
    def __init__(self, physics_model: Optional[PhysicsExpectationModel] = None):
        self.physics = physics_model or PhysicsExpectationModel()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._try_load_stats()

    def _try_load_stats(self):
        try:
            if RESIDUAL_STATS_FILE.exists():
                data = np.load(RESIDUAL_STATS_FILE)
                self.mean = data["mean"]
                self.std = data["std"]
        except Exception:
            pass

    def load_stats(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        # avoid div by zero
        self.std = np.maximum(self.std, 1e-6)

    def compute(self, frame: SensorFrame, expected: dict, params: EngineParams = None) -> ResidualVector:
        if params is None:
            params = EngineParams()
        # raw residuals
        rpm_r = frame.rpm - expected["rpm"]
        cht_r = [m - e for m, e in zip(frame.cht_c, expected["cht_c"])]
        egt_r = [m - e for m, e in zip(frame.egt_c, expected["egt_c"])]
        oil_p_r = frame.oil_pressure_psi - expected["oil_pressure_psi"]
        oil_t_r = frame.oil_temp_c - expected["oil_temp_c"]
        fuel_r = frame.fuel_flow_lph - expected["fuel_flow_lph"]
        vib_r = frame.vibration_g - expected["vibration_g"]
        batt_r = frame.batt_voltage - expected["batt_voltage"]

        raw_14 = np.array([rpm_r] + cht_r + egt_r + [oil_p_r, oil_t_r, fuel_r, vib_r, batt_r], dtype=float)

        if self.mean is not None and self.std is not None:
            norm = (raw_14 - self.mean) / self.std
        else:
            # fallback: simple scaling by healthy envelope (approx)
            scales = np.array([50, 5, 5, 5, 5, 15, 15, 15, 15, 5, 3, 2, 0.5, 0.5])
            norm = raw_14 / scales

        return ResidualVector(
            timestamp=frame.timestamp,
            rpm_residual=float(rpm_r),
            cht_residuals=[float(x) for x in cht_r],
            egt_residuals=[float(x) for x in egt_r],
            oil_p_residual=float(oil_p_r),
            oil_t_residual=float(oil_t_r),
            fuel_flow_residual=float(fuel_r),
            vibration_residual=float(vib_r),
            batt_v_residual=float(batt_r),
            normalized=norm.tolist(),
        )

    def compute_raw_vector(self, frame: SensorFrame, expected: dict) -> np.ndarray:
        rv = self.compute(frame, expected)
        return np.array([rv.rpm_residual] + rv.cht_residuals + rv.egt_residuals + [rv.oil_p_residual, rv.oil_t_residual, rv.fuel_flow_residual, rv.vibration_residual, rv.batt_v_residual])

    @staticmethod
    def compute_stats(residuals_14_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.stack(residuals_14_list, axis=0)  # (N, 14)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-6
        return mean, std

    def save_stats(self, mean: np.ndarray, std: np.ndarray):
        RESIDUAL_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(RESIDUAL_STATS_FILE, mean=mean, std=std)
        self.load_stats(mean, std)


# Helper for offline generation: compute residuals from frames using nominal physics
def compute_residuals_for_frames(frames: list[SensorFrame], physics: PhysicsExpectationModel = None, params: EngineParams = None) -> list[np.ndarray]:
    physics = physics or PhysicsExpectationModel()
    params = params or EngineParams()
    gen = ResidualGenerator(physics)
    # Use simple scales (no stats yet) — will re-normalize after stats are computed
    residuals = []
    for f in frames:
        expected = physics.predict_all(f, params)
        rv = gen.compute(f, expected, params)
        residuals.append(np.array(rv.normalized))
    return residuals
