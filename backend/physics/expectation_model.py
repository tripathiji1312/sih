"""
Physics expectation model — causal skeleton (archi.md 4.1)
Simplified but preserves correct causal gradients for residual generation.
"""
from dataclasses import dataclass, field
import numpy as np

from backend.config import PHYSICS, DT_S


@dataclass
class EngineParams:
    k1_cooling: float = 1.0
    eta_combustion: float = 1.0
    mu_friction: float = 1.0
    inj_health: np.ndarray = field(default_factory=lambda: np.ones(4))

    def __post_init__(self):
        if isinstance(self.inj_health, list):
            self.inj_health = np.array(self.inj_health, dtype=float)
        if self.inj_health is None:
            self.inj_health = np.ones(4)


class PhysicsExpectationModel:
    TAU_RPM = PHYSICS["tau_rpm"]
    T_COMBUSTION = PHYSICS["t_combustion"]
    K2_AMBIENT_LOSS = PHYSICS["k2_ambient_loss"]
    K3_AIRSPEED_COOL = PHYSICS["k3_airspeed_cool"]
    OIL_VISCOSITY_REF = PHYSICS["oil_viscosity_ref"]

    def predict_rpm(self, rpm_current: float, rpm_commanded: float, load_fraction: float) -> float:
        tau = self.TAU_RPM * (1.0 + load_fraction * 0.5)
        return rpm_current + (rpm_commanded - rpm_current) * DT_S / tau

    def predict_cht(self, cht_current: float, rpm: float, ambient_temp_c: float, airspeed_ms: float, params: EngineParams) -> float:
        dt = DT_S
        t_effective = self.T_COMBUSTION * params.eta_combustion
        heat_input = (t_effective - cht_current) * (rpm / 5800.0) ** 2 * 0.02
        ambient_loss = (cht_current - ambient_temp_c) * self.K2_AMBIENT_LOSS * params.k1_cooling
        airspeed_loss = (cht_current - ambient_temp_c) * self.K3_AIRSPEED_COOL * airspeed_ms * params.k1_cooling
        dcht = heat_input - ambient_loss - airspeed_loss
        return cht_current + dcht * dt

    def predict_egt(self, rpm: float, mixture_ratio: float, inj_health_cyl: float, params: EngineParams) -> float:
        base_egt = 650.0 + (rpm - 5000) * 0.05
        mixture_effect = (mixture_ratio - 1.0) * 100.0
        injector_effect = (1.0 - inj_health_cyl) * 150.0
        efficiency_effect = (1.0 - params.eta_combustion) * 200.0
        return base_egt + mixture_effect + injector_effect + efficiency_effect

    def predict_oil_pressure(self, rpm: float, oil_temp_c: float) -> float:
        pump_pressure = 2.0 + rpm * 0.012
        temp_factor = max(0.3, 1.0 - (oil_temp_c - 50.0) * 0.004)
        return pump_pressure * temp_factor

    def predict_fuel_flow(self, rpm: float, inj_health_avg: float) -> float:
        displacement_l = 1.352
        ve = 0.85 * inj_health_avg
        cycles_per_sec = rpm / 60.0 / 2.0
        fuel_per_cycle_l = 3.0e-5
        return displacement_l * ve * cycles_per_sec * fuel_per_cycle_l * 3600

    def predict_all(self, frame, params: EngineParams) -> dict:
        """
        Predict expected sensor values given current frame context and degradation params.
        `frame` is a SensorFrame (uses its rpm, altitude, oat, airspeed as context).
        Returns dict with expected values.
        """
        rpm_exp = self.predict_rpm(frame.rpm, frame.rpm, load_fraction=0.65)
        # Use current CHT as starting point for one-step prediction
        cht_exp = []
        for i in range(4):
            cht_exp.append(self.predict_cht(frame.cht_c[i], frame.rpm, frame.oat_c, frame.airspeed_ms, params))
        egt_exp = []
        for i in range(4):
            egt_exp.append(self.predict_egt(frame.rpm, 1.0, float(params.inj_health[i]), params))
        oil_p_exp = self.predict_oil_pressure(frame.rpm, frame.oil_temp_c)
        oil_t_exp = frame.oil_temp_c  # quasi-static, no one-step model; use measured
        fuel_exp = self.predict_fuel_flow(frame.rpm, float(np.mean(params.inj_health)))
        vib_exp = 0.8 + (frame.rpm / 5800) * 0.8  # baseline
        return dict(
            rpm=rpm_exp,
            cht_c=cht_exp,
            egt_c=egt_exp,
            oil_pressure_psi=oil_p_exp,
            oil_temp_c=oil_t_exp,
            fuel_flow_lph=fuel_exp,
            vibration_g=vib_exp,
            batt_voltage=13.8,
        )

    def predict_all_vector(self, frame, params: EngineParams) -> np.ndarray:
        d = self.predict_all(frame, params)
        # 14-dim to match residual channels
        return np.array(
            [d["rpm"]] + d["cht_c"] + d["egt_c"] + [d["oil_pressure_psi"], d["oil_temp_c"], d["fuel_flow_lph"], d["vibration_g"], d["batt_voltage"]],
            dtype=np.float64,
        )
