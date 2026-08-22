"""Pydantic schemas for API."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Literal

class SensorFrameSchema(BaseModel):
    timestamp: float
    mission_time_s: float
    rpm: float
    cht_c: List[float]
    egt_c: List[float]
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

class AlertSchema(BaseModel):
    timestamp: float
    severity: Literal["INFO", "CAUTION", "WARNING", "CRITICAL"]
    subsystem: str
    fault_type: str
    causal_evidence: str
    confidence: float
    recommended_action: str
    is_physics_fallback: bool = False

class HealthSchema(BaseModel):
    health_index: float
    confidence: float
    subsystem_health: Dict[str, float]
    is_physics_fallback: bool
