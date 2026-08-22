"""FMEA rules engine — maps residual patterns to fault causes."""
from __future__ import annotations
import numpy as np
from typing import Optional

class FMEAAttributionEngine:
    RULES = [
        {
            "name": "cooling_degradation",
            "subsystem": "cooling",
            "condition": lambda r: (
                np.mean([abs(r.get(f"cht_{i}", 0)) for i in range(1,5)]) > 0.7 and abs(r.get("oil_p", 0)) < 0.4 and abs(r.get("rpm", 0)) < 0.4
            ),
            "explanation": "CHT residuals elevated across all cylinders while oil pressure and RPM remain nominal — indicates cooling system efficiency loss.",
            "action": "Reduce power to 75%, increase airspeed for cooling. Schedule cooling system inspection within next 10 flight hours."
        },
        {
            "name": "lubrication_fault",
            "subsystem": "lubrication",
            "condition": lambda r: (r.get("oil_p", 0) < -0.6 and r.get("oil_t", 0) > 0.5 and abs(r.get("rpm", 0)) < 0.4),
            "explanation": "Oil pressure below expected with concurrent oil temperature elevation — indicates lubrication system degradation.",
            "action": "CRITICAL: Reduce power immediately. Monitor oil pressure. Prepare for precautionary landing."
        },
    ]

    def attribute(self, residuals: dict, arbitration: str = None, trust_scores: dict = None) -> Optional[dict]:
        if arbitration == "sensor_fault" and trust_scores:
            untrusted = [s for s, t in trust_scores.items() if t < 0.3]
            if untrusted:
                return dict(
                    subsystem="sensors", fault_type=f"sensor_fault_{untrusted[0]}",
                    causal_evidence=f"Sensor {untrusted[0]} diverges from physics and correlated sensors — sensor or wiring fault.",
                    recommended_action="Verify sensor wiring and connection.",
                    confidence=0.85, severity="CAUTION",
                )
        for rule in self.RULES:
            try:
                if rule["condition"](residuals):
                    return dict(
                        subsystem=rule["subsystem"], fault_type=rule["name"],
                        causal_evidence=rule["explanation"],
                        recommended_action=rule["action"],
                        confidence=0.9, severity="WARNING" if rule["subsystem"] != "lubrication" else "CRITICAL",
                    )
            except Exception:
                continue
        # Misfire check per cylinder
        for i in range(1,5):
            egt_key = f"egt_{i}"
            if egt_key in residuals:
                other = [residuals.get(f"egt_{j}", 0) for j in range(1,5) if j != i]
                if residuals[egt_key] > 0.8 and (residuals[egt_key] - np.mean(other) > 0.5 if other else False):
                    return dict(
                        subsystem="ignition", fault_type=f"misfire_cylinder_{i}",
                        causal_evidence=f"EGT residual spike on cylinder {i} while other cylinders remain nominal — indicates misfire on this cylinder.",
                        recommended_action=f"Run mag check. Cylinder {i} misfire confirmed. Reduce power. Schedule inspection.",
                        confidence=0.88, severity="WARNING",
                    )
        return None
