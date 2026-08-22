"""
Single-process FastAPI application — processing loop at 10 Hz.
Run: uvicorn backend.main:app --reload
"""
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.simulator.engine_simulator import EngineSimulator
from backend.simulator.fault_injector import FaultInjector
from backend.ingestion import DataValidator, SystemWatchdog, CircularBuffer
from backend.physics import PhysicsExpectationModel, TwinStateEstimator, ResidualGenerator, EngineParams
from backend.ml.inference import EvidentialModelInference
from backend.ml.models.ood_detector import OODDetector
from backend.ml.models.sensor_trust import SensorTrustEvaluator
from backend.fusion import FMEAAttributionEngine, EvidenceAggregator, DecisionArbiter
from backend.api.websocket import ConnectionManager
from backend.api.routes import router

app = FastAPI(title="Digital Twin — MALE UAV Aero-Piston Engine", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api")

# Singletons
simulator = EngineSimulator()
fault_injector = FaultInjector(simulator)
validator = DataValidator()
watchdog = SystemWatchdog()
buffer = CircularBuffer()
physics_model = PhysicsExpectationModel()
state_estimator = TwinStateEstimator()
residual_gen = ResidualGenerator(physics_model)
# Try to load stats if exists
try:
    residual_gen._try_load_stats()
except Exception:
    pass
ml_model = EvidentialModelInference()
ood_detector = OODDetector()
sensor_trust = SensorTrustEvaluator()
fmea_engine = FMEAAttributionEngine()
decision_arbiter = DecisionArbiter()
evidence_aggregator = EvidenceAggregator()
manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(processing_loop())

async def processing_loop():
    while True:
        tick_start = time.time()
        try:
            frame = simulator.get_frame()
            is_valid = validator.validate(frame)
            watchdog.on_frame(frame, valid=is_valid)
            watchdog_status = watchdog.system_health()
            buffer.append(frame)

            degradation_params = state_estimator.get_degradation_params()
            # For physics fallback, use EngineParams from degradation
            eng_params = EngineParams(
                k1_cooling=degradation_params.get("k1_cooling", 1.0),
                eta_combustion=degradation_params.get("eta_combustion", 1.0),
                mu_friction=degradation_params.get("mu_friction", 1.0),
            )
            expected = physics_model.predict_all(frame, eng_params)
            state_estimate, covariance = state_estimator.update(frame.to_vector())
            residuals = residual_gen.compute(frame, expected, eng_params)
            buffer.append_residual(residuals.to_vector())

            ml_output = {}
            ood_result = {"is_ood": False, "mahalanobis_distance": 0}
            trust_scores = {}
            arbitration = "normal"
            alert = None
            health = {"health_index": 100, "confidence": 1.0, "subsystem_health": {}, "is_physics_fallback": False}

            if buffer.is_full():
                window = buffer.get_ml_window()
                ml_output = ml_model.infer(window)
                ood_result = ood_detector.check(residuals.to_vector())
                trust_scores = sensor_trust.evaluate_trust(residuals.to_dict(), state_estimate)
                arbitration = decision_arbiter.arbitrate(trust_scores, residuals.to_dict())
                alert = fmea_engine.attribute(residuals.to_dict(), arbitration, trust_scores)
                health = evidence_aggregator.compute(residuals.to_dict(), ml_output, trust_scores, ood_result, arbitration, watchdog_status)

            await manager.broadcast({
                "type": "telemetry",
                "timestamp": frame.timestamp,
                "measured": frame.to_dict(),
                "expected": expected,
                "residuals": residuals.to_dict(),
                "ml_output": ml_output,
                "sensor_trust": trust_scores,
                "arbitration": arbitration,
                "alert": alert,
                "health": health,
                "watchdog": watchdog_status,
                "degradation_params": degradation_params,
            })
        except Exception as e:
            # Don't crash loop; log
            print(f"[processing_loop] error: {e}")
        elapsed = time.time() - tick_start
        await asyncio.sleep(max(0, 0.1 - elapsed))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            if data.get("command") == "inject_fault":
                fault_injector.inject(data["fault_type"], data.get("severity", 0.5))
            elif data.get("command") == "clear_fault":
                fault_injector.clear()
            elif data.get("command") == "set_mission":
                simulator.set_mission_profile(data["profile"])
    except WebSocketDisconnect:
        manager.disconnect(ws)
