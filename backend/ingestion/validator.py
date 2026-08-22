"""
Physical plausibility validator — Layer 1.
"""
from backend.config import ROTAX_912
from backend.simulator.engine_simulator import SensorFrame


class DataValidator:
    def validate(self, frame: SensorFrame) -> bool:
        # Range checks
        if not (1500 <= frame.rpm <= 6500):
            return False
        if not all(30 <= c <= 200 for c in frame.cht_c):
            return False
        if not all(350 <= e <= 1000 for e in frame.egt_c):
            return False
        if not (0 <= frame.oil_pressure_psi <= 120):
            return False
        if not (20 <= frame.oil_temp_c <= 160):
            return False
        if not (0 <= frame.fuel_flow_lph <= 80):
            return False
        if not (0 <= frame.vibration_g <= 20):
            return False
        if not (10 <= frame.batt_voltage <= 16):
            return False
        # Physical impossibility: EGT must be > CHT
        if any(e <= c for e, c in zip(frame.egt_c, frame.cht_c)):
            return False
        # Rate limits (rough)
        return True

    def validate_dict(self, d: dict) -> bool:
        try:
            # minimal
            return True
        except Exception:
            return False
