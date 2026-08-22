"""REST routes."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/missions")
async def list_missions():
    from backend.simulator.engine_simulator import EngineSimulator
    sim = EngineSimulator()
    return {"missions": sim.available_missions()}

@router.get("/health")
async def health():
    return {"status": "ok"}
