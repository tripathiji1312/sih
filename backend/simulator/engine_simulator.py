"""
Physics-based Rotax-912-class sensor simulator.
Generates realistic sensor frames at 10 Hz with optional fault injection.
Used for both live demo and offline training data generation.
"""
from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

import numpy as np

from backend.config import ROTAX_912, DT_S
from .mission_profiles import get_mission_profile


@dataclass
class SensorFrame:
    timestamp: float
    mission_time_s: float
    rpm: float
    cht_c: List[float]  # 4 cylinders
    egt_c: List[float]  # 4 cylinders
    oil_pressure_psi: float
    oil_temp_c: float
    fuel_flow_lph: float
    vibration_g: float
    batt_voltage: float
    inj_timing_deg: float
    altitude_m: float
    oat_c: float
    airspeed_ms: float
    fault_injected: Optional[str] = None

    def to_vector(self) -> np.ndarray:
        """Flat vector for UKF / residual: 11 dims (excludes metadata)."""
        return np.array(
            [self.rpm]
            + self.cht_c
            + self.egt_c
            + [self.oil_pressure_psi, self.oil_temp_c],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class EngineSimulator:
    """
    Simplified but causally-correct thermodynamic + mechanical model.
    State is integrated with Euler step at DT_S.
    Faults are injected via multiplicative degradation parameters.
    """

    # Physics constants (tuned to produce Rotax-912 envelope)
    TAU_RPM = 0.8
    T_COMB = 2100.0
    K2_AMBIENT = 0.15
    K3_AIRSPEED = 0.008

    def __init__(self, seed: int = 0, mission_profile: str = "cruise", altitude_m: float = 3000, oat_c: float = 15):
        self.rng = np.random.default_rng(seed)
        self.mission_profile_name = mission_profile
        self.mission_fn = get_mission_profile(mission_profile)
        self.altitude_m = altitude_m
        self.oat_c = oat_c

        # Engine state
        self.mission_time_s = 0.0
        self.rpm = 4800.0
        self.cht_c = np.array([110.0, 110.0, 110.0, 110.0])
        self.egt_c = np.array([680.0, 680.0, 680.0, 680.0])
        self.oil_temp_c = 80.0
        self.oil_pressure_psi = 55.0
        self.fuel_flow_lph = 18.0
        self.vibration_g = 1.2
        self.batt_voltage = 13.8
        self.inj_timing_deg = 28.0

        # Degradation params (fault injection handles these)
        self.k1_cooling = 1.0
        self.eta_combustion = 1.0
        self.mu_friction = 1.0
        self.inj_health = np.ones(4)
        self.fault_label: Optional[str] = None

        # Sensor fault overrides: sensor_name -> fault_spec
        self.sensor_faults: Dict[str, dict] = {}

        # For deterministic mission variation
        self._base_altitude = altitude_m
        self._base_oat = oat_c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_mission_profile(self, name: str, altitude_m: float = None, oat_c: float = None):
        self.mission_profile_name = name
        self.mission_fn = get_mission_profile(name)
        if altitude_m is not None:
            self._base_altitude = altitude_m
        if oat_c is not None:
            self._base_oat = oat_c

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mission_time_s = 0.0
        self.rpm = 4800.0
        self.cht_c = np.array([110.0, 110.0, 110.0, 110.0])
        self.egt_c = np.array([680.0, 680.0, 680.0, 680.0])
        self.oil_temp_c = 80.0
        self.oil_pressure_psi = 55.0
        self.fuel_flow_lph = 18.0
        self.vibration_g = 1.2
        self.k1_cooling = 1.0
        self.eta_combustion = 1.0
        self.mu_friction = 1.0
        self.inj_health = np.ones(4)
        self.fault_label = None
        self.sensor_faults.clear()

    def step(self) -> SensorFrame:
        """Advance one DT_S and return a SensorFrame."""
        # Mission context
        ctx = self.mission_fn(
            self.mission_time_s, altitude_m=self._base_altitude, oat_c=self._base_oat
        )
        rpm_cmd = float(np.clip(ctx["rpm_command"], ROTAX_912["min_rpm"], ROTAX_912["max_rpm"]))
        alt = float(ctx["altitude_m"])
        oat = float(ctx["oat_c"])
        airspeed = float(ctx["airspeed_ms"])
        load_frac = float(ctx.get("load_fraction", 0.65))
        mixture = float(ctx.get("mixture_ratio", 1.0))

        # RPM first-order lag
        tau = self.TAU_RPM * (1.0 + load_frac * 0.5)
        self.rpm += (rpm_cmd - self.rpm) * DT_S / tau
        self.rpm += self.rng.normal(0, 3.0)  # sensor noise
        self.rpm = float(np.clip(self.rpm, 1500, 6000))

        # CHT dynamics per cylinder
        for i in range(4):
            t_eff = self.T_COMB * self.eta_combustion
            # per-cylinder combustion variation via injector health
            cyl_comb = t_eff * (0.9 + 0.1 * self.inj_health[i])
            heat_input = (cyl_comb - self.cht_c[i]) * (self.rpm / 5800.0) ** 2 * 0.02
            ambient_loss = (self.cht_c[i] - oat) * self.K2_AMBIENT * self.k1_cooling
            airspeed_loss = (self.cht_c[i] - oat) * self.K3_AIRSPEED * airspeed * self.k1_cooling
            dcht = heat_input - ambient_loss - airspeed_loss
            self.cht_c[i] += dcht * DT_S
            self.cht_c[i] += self.rng.normal(0, 0.15)
            # vibration / friction adds heat
            self.cht_c[i] += (self.mu_friction - 1.0) * 8.0 * DT_S

        # EGT per cylinder
        for i in range(4):
            base = 650.0 + (self.rpm - 5000) * 0.05
            mix_eff = (mixture - 1.0) * 100.0
            inj_eff = (1.0 - self.inj_health[i]) * 150.0
            eta_eff = (1.0 - self.eta_combustion) * 200.0
            self.egt_c[i] = base + mix_eff + inj_eff + eta_eff + self.rng.normal(0, 3.0)
            self.egt_c[i] = float(np.clip(self.egt_c[i], 400, 950))

        self.cht_c = np.clip(self.cht_c, 40, 180)
        # Oil temp: tracks CHT + friction
        target_oil_t = 75.0 + np.mean(self.cht_c) * 0.15 + (self.mu_friction - 1.0) * 30.0
        self.oil_temp_c += (target_oil_t - self.oil_temp_c) * 0.01
        self.oil_temp_c += self.rng.normal(0, 0.2)
        self.oil_temp_c = float(np.clip(self.oil_temp_c, 30, 150))

        # Oil pressure: RPM-dependent pump minus viscosity
        pump = 2.0 + self.rpm * 0.012
        temp_factor = max(0.3, 1.0 - (self.oil_temp_c - 50.0) * 0.004)
        # lubrication fault reduces pressure
        lub_factor = 1.0
        if self.fault_label == "lubrication_fault":
            # severity encoded in mu_friction / special field; use k1 as proxy if needed
            # Instead we store lub severity separately; fallback to mu_friction
            lub_factor = 1.0 - getattr(self, "_lub_severity", 0.0) * 0.9
        self.oil_pressure_psi = pump * temp_factor * lub_factor + self.rng.normal(0, 0.5)
        self.oil_pressure_psi = float(np.clip(self.oil_pressure_psi, 5, 95))

        # Fuel flow
        ve = 0.85 * float(np.mean(self.inj_health))
        cycles = self.rpm / 60.0 / 2.0
        fuel_per_cycle = 3.0e-5
        displacement = 1.352
        self.fuel_flow_lph = displacement * ve * cycles * fuel_per_cycle * 3600 + self.rng.normal(0, 0.3)
        self.fuel_flow_lph = float(np.clip(self.fuel_flow_lph, 5, 50))

        # Vibration: base + RPM + misfire + friction
        base_vib = 0.8 + (self.rpm / 5800) * 0.8
        misfire_vib = 0.0
        if self.fault_label and "misfire" in self.fault_label:
            misfire_vib = getattr(self, "_misfire_severity", 0.3) * 4.0
        friction_vib = (self.mu_friction - 1.0) * 2.0
        self.vibration_g = base_vib + misfire_vib + friction_vib + abs(self.rng.normal(0, 0.12))
        self.vibration_g = float(np.clip(self.vibration_g, 0.2, 15.0))

        # Battery
        self.batt_voltage = 13.8 + self.rng.normal(0, 0.05)
        self.batt_voltage = float(np.clip(self.batt_voltage, 11.5, 15.0))

        self.mission_time_s += DT_S
        ts = time.time()

        frame = SensorFrame(
            timestamp=ts,
            mission_time_s=self.mission_time_s,
            rpm=float(self.rpm),
            cht_c=self.cht_c.tolist(),
            egt_c=self.egt_c.tolist(),
            oil_pressure_psi=float(self.oil_pressure_psi),
            oil_temp_c=float(self.oil_temp_c),
            fuel_flow_lph=float(self.fuel_flow_lph),
            vibration_g=float(self.vibration_g),
            batt_voltage=float(self.batt_voltage),
            inj_timing_deg=float(self.inj_timing_deg),
            altitude_m=float(alt),
            oat_c=float(oat),
            airspeed_ms=float(airspeed),
            fault_injected=self.fault_label,
        )

        # Apply sensor faults as overrides on the frame
        frame = self._apply_sensor_faults(frame)
        return frame

    def get_frame(self) -> SensorFrame:
        return self.step()

    def run_mission(self, duration_s: float, fault_onset_s: Optional[float] = None, fault_type: Optional[str] = None, fault_severity: float = 0.5, fault_kwargs: dict = None) -> list[SensorFrame]:
        """
        Run a full mission and return list of frames.
        If fault_type is given, inject at fault_onset_s.
        """
        frames: list[SensorFrame] = []
        steps = int(duration_s / DT_S)
        if fault_onset_s is None and fault_type is not None:
            fault_onset_s = duration_s * 0.5
        onset_step = int(fault_onset_s / DT_S) if fault_onset_s is not None else None

        for step in range(steps):
            if fault_type is not None and onset_step is not None and step == onset_step:
                self.inject_fault(fault_type, fault_severity, **(fault_kwargs or {}))
            frames.append(self.step())
        return frames

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------
    def inject_fault(self, fault_type: str, severity: float = 0.5, **kwargs):
        """
        fault_type: one of FAULT_CLASSES (or sensor fault types)
        severity: 0-1
        """
        self.fault_label = fault_type
        severity = float(np.clip(severity, 0.0, 1.0))

        if fault_type == "cooling_degradation":
            # k1_cooling degrades from 1.0 towards 0.3
            self.k1_cooling = 1.0 - severity * 0.7
        elif fault_type == "lubrication_fault":
            self._lub_severity = severity
            self.mu_friction = 1.0 + severity * 0.6
        elif fault_type.startswith("misfire"):
            # misfire_cyl1 etc. or generic misfire
            cyl = kwargs.get("cylinder", 1)
            if "_" in fault_type:
                try:
                    cyl = int(fault_type.split("cyl")[-1])
                except Exception:
                    pass
            idx = int(np.clip(cyl, 1, 4)) - 1
            self._misfire_severity = severity
            self._misfire_cyl = idx
            self.eta_combustion = 1.0  # keep global, misfire handled via vibration + EGT logic
            # For residual generation: degrade injector/ combustion for that cylinder
            self.inj_health[idx] = 1.0 - severity * 0.5
            # Also bump EGT for that cylinder via special offset handled in step
            self._misfire_egt_offset = severity * 120
        elif fault_type == "injector_clog":
            cyl = kwargs.get("cylinder", int(self.rng.integers(1, 5)))
            idx = int(np.clip(cyl, 1, 4)) - 1
            self.inj_health[idx] = 1.0 - severity * 0.6
        elif fault_type == "sensor_fault":
            sensor = kwargs.get("sensor", "oil_pressure")
            mode = kwargs.get("mode", "drift")
            self._inject_sensor_fault(sensor, mode, severity)
        else:
            # generic: treat as cooling
            self.k1_cooling = 1.0 - severity * 0.5

    def clear_fault(self):
        self.k1_cooling = 1.0
        self.eta_combustion = 1.0
        self.mu_friction = 1.0
        self.inj_health = np.ones(4)
        self.fault_label = None
        self.sensor_faults.clear()
        for attr in ["_lub_severity", "_misfire_severity", "_misfire_cyl", "_misfire_egt_offset"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def _inject_sensor_fault(self, sensor: str, mode: str, severity: float):
        self.sensor_faults[sensor] = dict(mode=mode, severity=severity, t0=self.mission_time_s)

    def _apply_sensor_faults(self, frame: SensorFrame) -> SensorFrame:
        for sensor, spec in self.sensor_faults.items():
            mode = spec["mode"]
            sev = spec["severity"]
            dt = frame.mission_time_s - spec["t0"]
            if sensor == "oil_pressure":
                if mode == "drift":
                    frame.oil_pressure_psi += sev * dt * 0.5
                elif mode == "stuck":
                    frame.oil_pressure_psi = 55.0
                elif mode == "noise":
                    frame.oil_pressure_psi += self.rng.normal(0, sev * 10)
                elif mode == "spike":
                    if self.rng.random() < 0.02:
                        frame.oil_pressure_psi += sev * 40 * (1 if self.rng.random() > 0.5 else -1)
            elif sensor.startswith("cht"):
                try:
                    idx = int(sensor.split("_")[1]) - 1
                except Exception:
                    idx = 0
                if mode == "drift":
                    frame.cht_c[idx] += sev * dt * 0.3
                elif mode == "stuck":
                    frame.cht_c[idx] = 110.0
                elif mode == "spike":
                    if self.rng.random() < 0.02:
                        frame.cht_c[idx] += sev * 30
            elif sensor.startswith("egt"):
                try:
                    idx = int(sensor.split("_")[1]) - 1
                except Exception:
                    idx = 0
                if mode == "drift":
                    frame.egt_c[idx] += sev * dt * 1.0
                elif mode == "stuck":
                    frame.egt_c[idx] = 680.0
            elif sensor == "rpm":
                if mode == "noise":
                    frame.rpm += self.rng.normal(0, sev * 200)
                elif mode == "stuck":
                    frame.rpm = 4800
            elif sensor == "fuel_flow":
                if mode == "drift":
                    frame.fuel_flow_lph += sev * dt * 0.05
                elif mode == "noise":
                    frame.fuel_flow_lph += self.rng.normal(0, sev * 5)
        # misfire EGT offset
        if hasattr(self, "_misfire_egt_offset") and hasattr(self, "_misfire_cyl"):
            idx = self._misfire_cyl
            frame.egt_c[idx] += self._misfire_egt_offset
        return frame

    def available_missions(self):
        from .mission_profiles import MISSION_PROFILES
        return list(MISSION_PROFILES.keys())
