"""Evidence aggregator — combines physics + ML + sensor trust -> Health Index."""
from __future__ import annotations
import numpy as np

class EvidenceAggregator:
    def compute(self, residuals: dict, ml_output: dict, trust_scores: dict, ood_result: dict, arbitration: str, watchdog_status: dict) -> dict:
        # Health 0-100: start 100, subtract penalties
        residual_penalties = sum(min(5, abs(v) * 8) for v in residuals.values()) / max(len(residuals), 1)
        # ML fault probability
        fault_prob = ml_output.get("fault_probability", 0.0) if isinstance(ml_output, dict) else 0.0
        epistemic = ml_output.get("epistemic_uncertainty", 0.5) if isinstance(ml_output, dict) else 0.5
        ood_pen = 10 if ood_result.get("is_ood", False) else 0
        trust_pen = (1 - np.mean(list(trust_scores.values()))) * 15 if trust_scores else 0
        watchdog_pen = 20 if watchdog_status.get("overall_status") == "CRITICAL_DATA_LOSS" else (10 if watchdog_status.get("overall_status") == "DATA_DEGRADED" else 0)

        health = 100 - residual_penalties - fault_prob * 35 - epistemic * 10 - ood_pen - trust_pen - watchdog_pen
        health = float(np.clip(health, 0, 100))

        # Confidence: low if epistemic high or OOD or untrusted sensors
        confidence = float(np.clip(1.0 - epistemic * 0.7 - (0.3 if ood_result.get("is_ood") else 0) - (0.2 if trust_pen > 5 else 0), 0, 1))

        subsystem_health = {
            "cooling": float(np.clip(1 - np.mean([abs(residuals.get(f"cht_{i}", 0)) for i in range(1,5)]) * 0.4, 0, 1)),
            "lubrication": float(np.clip(1 - abs(residuals.get("oil_p", 0)) * 0.5, 0, 1)),
            "combustion": float(np.clip(1 - np.mean([abs(residuals.get(f"egt_{i}", 0)) for i in range(1,5)]) * 0.3, 0, 1)),
            "fuel_system": float(np.clip(1 - abs(residuals.get("fuel_flow", 0)) * 0.5, 0, 1)),
            "mechanical": float(np.clip(1 - abs(residuals.get("vibration", 0)) * 0.3, 0, 1)),
        }

        return dict(
            health_index=health,
            confidence=confidence,
            subsystem_health=subsystem_health,
            is_physics_fallback=bool(epistemic > 0.7 or ood_result.get("is_ood", False)),
            fault_probability=fault_prob,
            epistemic_uncertainty=epistemic,
        )
