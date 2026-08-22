"""
Fault injector — thin wrapper around EngineSimulator.inject_fault for demo/WebSocket API.
Also provides helper to inject random combined faults for OOD generation.
"""
from __future__ import annotations
import random
from typing import Optional
import numpy as np

from .engine_simulator import EngineSimulator


class FaultInjector:
    def __init__(self, simulator: EngineSimulator):
        self.sim = simulator

    def inject(self, fault_type: str, severity: float = 0.5, **kwargs):
        self.sim.inject_fault(fault_type, severity, **kwargs)

    def clear(self):
        self.sim.clear_fault()

    def inject_random_combined(self, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        faults = ["cooling_degradation", "lubrication_fault", "injector_clog"]
        chosen = rng.choice(faults, size=2, replace=False)
        for f in chosen:
            self.sim.inject_fault(f, severity=float(rng.uniform(0.3, 0.6)))
        self.sim.fault_label = "combined_faults"

    def inject_sensor_fault(self, sensor: str, mode: str = "drift", severity: float = 0.5):
        self.sim.inject_fault("sensor_fault", severity, sensor=sensor, mode=mode)
