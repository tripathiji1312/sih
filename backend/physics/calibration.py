"""
Physics calibration — tune constants so healthy residuals < 5% for all params.
Run: python -m backend.physics.calibration
"""
import numpy as np
from backend.simulator.engine_simulator import EngineSimulator
from backend.physics.expectation_model import PhysicsExpectationModel, EngineParams


def calibrate(num_frames: int = 2000, seed: int = 0):
    sim = EngineSimulator(seed=seed, mission_profile="cruise")
    physics = PhysicsExpectationModel()
    params = EngineParams()
    residuals = []
    for _ in range(num_frames):
        frame = sim.step()
        expected = physics.predict_all(frame, params)
        # simple residual magnitude
        r_rpm = abs(frame.rpm - expected["rpm"]) / 5000
        r_cht = np.mean([abs(m - e) / 130 for m, e in zip(frame.cht_c, expected["cht_c"])])
        r_egt = np.mean([abs(m - e) / 700 for m, e in zip(frame.egt_c, expected["egt_c"])])
        residuals.append((r_rpm, r_cht, r_egt))
    arr = np.array(residuals)
    print(f"Mean residuals: rpm={arr[:,0].mean():.4f} cht={arr[:,1].mean():.4f} egt={arr[:,2].mean():.4f}")
    print(f"Max residuals: rpm={arr[:,0].max():.4f} cht={arr[:,1].max():.4f} egt={arr[:,2].max():.4f}")
    ok = (arr.mean(axis=0) < 0.05).all()
    print(f"Calibration {'PASSED' if ok else 'FAILED'} (target <5%)")
    return arr


if __name__ == "__main__":
    calibrate()
