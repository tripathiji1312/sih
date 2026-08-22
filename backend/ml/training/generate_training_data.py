"""
Generate training data from physics simulator with fault injection.
Produces: data/training/windows.npz with X=(N,14,30), y=(N,), plus
data/models/residual_stats.npz and per-scenario breakdown.
SOTA: stratified scenario sampling, proper onset randomization, vectorized windows.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Local imports — allow running as `python -m backend.ml.training.generate_training_data`
from backend.simulator.engine_simulator import EngineSimulator
from backend.physics.expectation_model import PhysicsExpectationModel, EngineParams
from backend.physics.residual_generator import ResidualGenerator
from backend.config import (
    SCENARIOS, SCENARIOS_FULL, FAULT_TO_IDX, FAULT_CLASSES,
    WINDOW_SIZE, WINDOW_STRIDE, SAMPLE_RATE_HZ, DT_S,
    DATA_ROOT, TRAINING_DATA_ROOT, MODELS_ROOT, RESIDUAL_STATS_FILE,
)


def sliding_windows(residuals: np.ndarray, label: int, onset_idx: int = None, window: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE):
    """
    residuals: (T, 14) time series of normalized residuals
    label: fault class idx
    onset_idx: if fault, index where fault starts; windows before onset labeled healthy (0)
    Returns list of (window_array (14,30), label)
    """
    T = residuals.shape[0]
    windows = []
    labels = []
    for start in range(0, T - window + 1, stride):
        end = start + window
        w = residuals[start:end].T  # (14, 30)
        # Label logic: if onset inside window or after window start
        if onset_idx is not None:
            # If window ends before onset -> healthy
            if end <= onset_idx:
                lbl = FAULT_TO_IDX["healthy"]
            else:
                # majority after onset -> fault
                # For misfire/injector, fault may be subtle early; still fault
                lbl = label
        else:
            lbl = label
        windows.append(w)
        labels.append(lbl)
    return windows, labels


def generate_one_mission(
    profile: str, duration_s: float, fault_type: str = None, severity: float = 0.5,
    onset_s: float = None, seed: int = 0, altitude_m: float = 3000, oat_c: float = 15,
    physics: PhysicsExpectationModel = None, fault_kwargs: dict = None,
) -> tuple[np.ndarray, list[int], dict]:
    """
    Returns residuals (T,14) raw (before global normalization) and metadata.
    We compute raw residuals (measured - expected) in physical units then later normalize.
    For speed, we collect raw 14-dim vectors.
    """
    physics = physics or PhysicsExpectationModel()
    sim = EngineSimulator(seed=seed, mission_profile=profile, altitude_m=altitude_m, oat_c=oat_c)
    # Inject fault at onset_s if given
    onset_idx = None
    if fault_type is not None and fault_type != "healthy":
        if onset_s is None:
            onset_s = duration_s * random.uniform(0.3, 0.6)
        onset_idx = int(onset_s / DT_S)

    steps = int(duration_s / DT_S)
    raw_residuals = []
    gen_tmp = ResidualGenerator(physics)  # without stats -> uses fallback scaling? We want raw
    # For raw we compute difference without normalization; bypass gen's fallback
    # Instead compute manually: measured - expected (physical units) then we'll global normalize
    params_nominal = EngineParams()  # healthy expectation

    for step in range(steps):
        if fault_type is not None and fault_type != "healthy" and onset_idx is not None and step == onset_idx:
            # Inject with appropriate kwargs
            kw = fault_kwargs or {}
            # For injector/misfire pick cylinder if not specified
            if fault_type == "injector_clog" and "cylinder" not in kw:
                kw["cylinder"] = random.randint(1, 4)
            if fault_type.startswith("misfire") and "cylinder" not in kw:
                # Extract cyl from label if needed
                if fault_type.startswith("misfire_cyl"):
                    try:
                        kw["cylinder"] = int(fault_type.split("cyl")[-1])
                    except Exception:
                        kw["cylinder"] = random.randint(1, 4)
                else:
                    kw["cylinder"] = random.randint(1, 4)
                    fault_type = f"misfire_cyl{kw['cylinder']}"
            sim.inject_fault(fault_type, severity, **kw)

        frame = sim.step()
        expected = physics.predict_all(frame, params_nominal)
        # Raw residual vector (14 dims physical units)
        raw = np.array(
            [frame.rpm - expected["rpm"]]
            + [m - e for m, e in zip(frame.cht_c, expected["cht_c"])]
            + [m - e for m, e in zip(frame.egt_c, expected["egt_c"])]
            + [frame.oil_pressure_psi - expected["oil_pressure_psi"],
               frame.oil_temp_c - expected["oil_temp_c"],
               frame.fuel_flow_lph - expected["fuel_flow_lph"],
               frame.vibration_g - expected["vibration_g"],
               frame.batt_voltage - expected["batt_voltage"]],
            dtype=np.float64,
        )
        raw_residuals.append(raw)

    raw_residuals = np.stack(raw_residuals, axis=0)  # (T,14)
    return raw_residuals, onset_idx, dict(profile=profile, fault_type=fault_type, severity=severity, onset_s=onset_s)


def generate_all(
    scenarios: dict = None,
    full: bool = False,
    seed: int = 42,
    output_dir: Path = None,
    duration_override_min: float = None,
) -> Path:
    """
    Main generation routine.
    - First pass: generate all raw missions
    - Compute mean/std from healthy missions only
    - Second pass: normalize and window
    """
    if scenarios is None:
        scenarios = SCENARIOS_FULL if full else SCENARIOS

    output_dir = output_dir or TRAINING_DATA_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    physics = PhysicsExpectationModel()

    # ------------------------------------------------------------------
    # Collect raw missions
    # ------------------------------------------------------------------
    all_missions = []  # list of (raw_residuals (T,14), onset_idx, label_idx, meta)
    healthy_raw_list = []

    print(f"[generate] Using {'FULL' if full else 'FAST'} scenarios config")
    for scenario, cfg in scenarios.items():
        print(f"[generate] Scenario: {scenario} | cfg={cfg}")
        count = cfg.get("count", 10)
        duration_min = duration_override_min or cfg.get("duration_min", 10)
        duration_s = duration_min * 60

        for i in tqdm(range(count), desc=f"  {scenario}"):
            s = int(rng.integers(0, 1_000_000))
            # Determine fault label and params per scenario
            if scenario == "healthy":
                profile = rng.choice(cfg.get("profiles", ["cruise"]))
                alt = int(rng.choice(cfg.get("altitudes", [3000])))
                oat = int(rng.choice(cfg.get("ambient_temps", [15])))
                raw, onset_idx, meta = generate_one_mission(
                    profile=profile, duration_s=duration_s, fault_type="healthy", seed=s,
                    altitude_m=alt, oat_c=oat, physics=physics,
                )
                label_idx = FAULT_TO_IDX["healthy"]
                healthy_raw_list.append(raw)
                all_missions.append((raw, onset_idx, label_idx, meta))

            elif scenario == "cooling_degradation":
                severity = float(rng.choice(cfg.get("severities", [0.5])))
                profile = rng.choice(["cruise", "climb", "hot_weather", "high_altitude"])
                raw, onset_idx, meta = generate_one_mission(
                    profile=profile, duration_s=duration_s, fault_type="cooling_degradation",
                    severity=severity, seed=s, physics=physics,
                )
                all_missions.append((raw, onset_idx, FAULT_TO_IDX["cooling_degradation"], meta))

            elif scenario == "lubrication_fault":
                severity = float(rng.choice(cfg.get("severities", [0.4])))
                profile = rng.choice(["cruise", "loiter", "throttle_transitions"])
                raw, onset_idx, meta = generate_one_mission(
                    profile=profile, duration_s=duration_s, fault_type="lubrication_fault",
                    severity=severity, seed=s, physics=physics,
                )
                all_missions.append((raw, onset_idx, FAULT_TO_IDX["lubrication_fault"], meta))

            elif scenario == "misfire_single_cyl":
                severity = float(rng.choice(cfg.get("severities", [0.5])))
                cyl = int(rng.integers(1, 5))
                label_name = f"misfire_cyl{cyl}"
                profile = rng.choice(["cruise", "throttle_transitions", "loiter"])
                raw, onset_idx, meta = generate_one_mission(
                    profile=profile, duration_s=duration_s, fault_type=label_name,
                    severity=severity, seed=s, physics=physics, fault_kwargs=dict(cylinder=cyl),
                )
                all_missions.append((raw, onset_idx, FAULT_TO_IDX[label_name], meta))

            elif scenario == "injector_clog":
                severity = float(rng.choice(cfg.get("severities", [0.4])))
                cyl = int(rng.integers(1, 5))
                profile = rng.choice(["cruise", "throttle_transitions"])
                raw, onset_idx, meta = generate_one_mission(
                    profile=profile, duration_s=duration_s, fault_type="injector_clog",
                    severity=severity, seed=s, physics=physics, fault_kwargs=dict(cylinder=cyl),
                )
                # injector_clog single class regardless of cyl — but we keep cyl in meta
                all_missions.append((raw, onset_idx, FAULT_TO_IDX["injector_clog"], meta))

            elif scenario == "sensor_faults":
                sensor = str(rng.choice(cfg.get("sensors", ["oil_pressure"])))
                mode = str(rng.choice(cfg.get("types", ["drift"])))
                severity = float(rng.uniform(0.3, 0.8))
                profile = rng.choice(["cruise", "loiter"])
                raw, onset_idx, meta = generate_one_mission(
                    profile=profile, duration_s=duration_s, fault_type="sensor_fault",
                    severity=severity, seed=s, physics=physics,
                    fault_kwargs=dict(sensor=sensor, mode=mode),
                )
                all_missions.append((raw, onset_idx, FAULT_TO_IDX["sensor_fault"], meta))

            elif scenario == "ood_scenarios":
                cond = str(rng.choice(cfg.get("conditions", ["combined_faults"])))
                # Map OOD conditions to missions
                if cond == "extreme_altitude_8500m":
                    raw, onset_idx, meta = generate_one_mission(
                        profile="extreme_altitude", duration_s=duration_s, fault_type="healthy",
                        seed=s, physics=physics, altitude_m=8500, oat_c=-35,
                    )
                    # Label as healthy but will be used as OOD eval, not training? Mark as healthy for now but meta says OOD
                    meta["ood_condition"] = cond
                    all_missions.append((raw, onset_idx, FAULT_TO_IDX["healthy"], meta))
                elif cond == "combined_faults":
                    # Inject two faults at different times
                    profile = "cruise"
                    sim = EngineSimulator(seed=s, mission_profile=profile)
                    steps = int(duration_s / DT_S)
                    onset1 = int(steps * 0.3)
                    onset2 = int(steps * 0.6)
                    raw_list = []
                    params_nom = EngineParams()
                    gen_tmp = ResidualGenerator(physics)
                    for step in range(steps):
                        if step == onset1:
                            sim.inject_fault("cooling_degradation", 0.5)
                        if step == onset2:
                            sim.inject_fault("lubrication_fault", 0.3)
                        frame = sim.step()
                        expected = physics.predict_all(frame, params_nom)
                        raw = np.array(
                            [frame.rpm - expected["rpm"]]
                            + [m - e for m, e in zip(frame.cht_c, expected["cht_c"])]
                            + [m - e for m, e in zip(frame.egt_c, expected["egt_c"])]
                            + [frame.oil_pressure_psi - expected["oil_pressure_psi"],
                               frame.oil_temp_c - expected["oil_temp_c"],
                               frame.fuel_flow_lph - expected["fuel_flow_lph"],
                               frame.vibration_g - expected["vibration_g"],
                               frame.batt_voltage - expected["batt_voltage"]],
                            dtype=float,
                        )
                        raw_list.append(raw)
                    raw = np.stack(raw_list, axis=0)
                    meta = dict(ood_condition=cond, profile=profile)
                    all_missions.append((raw, onset1, FAULT_TO_IDX["healthy"], meta))
                else:
                    # Other OOD: treat as healthy with extreme env
                    raw, onset_idx, meta = generate_one_mission(
                        profile="cruise", duration_s=duration_s, fault_type="healthy",
                        seed=s, physics=physics, altitude_m=7000, oat_c=-30,
                    )
                    meta["ood_condition"] = cond
                    all_missions.append((raw, onset_idx, FAULT_TO_IDX["healthy"], meta))
            else:
                print(f"[warn] Unknown scenario {scenario}, skipping")

    # ------------------------------------------------------------------
    # Compute residual stats from healthy missions only
    # ------------------------------------------------------------------
    if healthy_raw_list:
        healthy_concat = np.concatenate(healthy_raw_list, axis=0)  # (N_healthy_frames,14)
        mean = healthy_concat.mean(axis=0)
        std = healthy_concat.std(axis=0) + 1e-6
        # Save
        np.savez(RESIDUAL_STATS_FILE, mean=mean, std=std)
        print(f"[generate] Saved residual stats to {RESIDUAL_STATS_FILE}")
        print(f"[generate] mean={mean.round(3)}")
        print(f"[generate] std={std.round(3)}")
    else:
        mean = np.zeros(14)
        std = np.ones(14)

    # ------------------------------------------------------------------
    # Normalize and create windows
    # ------------------------------------------------------------------
    X_all, y_all = [], []
    per_class_counts = {name: 0 for name in FAULT_CLASSES}
    ood_X, ood_meta = [], []

    for raw, onset_idx, label_idx, meta in all_missions:
        # Normalize
        norm = (raw - mean) / std  # (T,14)
        # Check if OOD mission -> save separately for eval but also generate windows for OOD detection training?
        is_ood = "ood_condition" in meta
        windows, labels = sliding_windows(norm, label_idx, onset_idx, WINDOW_SIZE, WINDOW_STRIDE)
        if is_ood:
            # For training we exclude pure OOD from classification; keep for OOD eval set
            ood_X.extend(windows)
            ood_meta.extend([meta] * len(windows))
            # Also add some healthy windows to main set? Already counted if healthy
            if label_idx == FAULT_TO_IDX["healthy"]:
                # Don't pollute main training with extreme OOD healthy? Include but mark
                # For now, include 50% to give model exposure
                if random.random() < 0.5:
                    X_all.extend(windows)
                    y_all.extend(labels)
                    for lbl in labels:
                        per_class_counts[FAULT_CLASSES[lbl]] += 1
        else:
            X_all.extend(windows)
            y_all.extend(labels)
            for lbl in labels:
                per_class_counts[FAULT_CLASSES[lbl]] += 1

    X_all = np.stack(X_all, axis=0) if X_all else np.zeros((0, 14, 30))
    y_all = np.array(y_all, dtype=np.int64)
    ood_X_arr = np.stack(ood_X, axis=0) if ood_X else np.zeros((0, 14, 30))

    # Shuffle
    perm = rng.permutation(len(X_all))
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Save main windows
    output_path = output_dir / "windows.npz"
    np.savez_compressed(output_path, X=X_all, y=y_all, mean=mean, std=std, per_class_counts=np.array(list(per_class_counts.items()), dtype=object))
    print(f"[generate] Saved windows to {output_path} : X={X_all.shape}, y={y_all.shape}")
    print(f"[generate] Per-class counts: {per_class_counts}")

    # Save OOD windows separately
    if len(ood_X_arr) > 0:
        ood_path = output_dir / "ood_windows.npz"
        np.savez_compressed(ood_path, X=ood_X_arr, meta=np.array(ood_meta, dtype=object))
        print(f"[generate] Saved OOD windows to {ood_path} : {ood_X_arr.shape}")

    # Also save residual stats for physics
    # Save class distribution plot data
    stats = dict(
        n_windows=len(X_all),
        n_ood=len(ood_X_arr),
        per_class_counts=per_class_counts,
        mean=mean.tolist(),
        std=std.tolist(),
    )
    import json
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate training data for SIH digital twin")
    parser.add_argument("--full", action="store_true", help="Use full scenario counts (60min missions)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test: 1 mission per scenario, 2 min each")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-min", type=float, default=None, help="Override mission duration in minutes")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        # Tiny config for testing
        scenarios = {
            "healthy": {"count": 2, "duration_min": 2, "profiles": ["cruise"], "altitudes": [3000], "ambient_temps": [15]},
            "cooling_degradation": {"count": 2, "severities": [0.5], "onset": "random"},
            "lubrication_fault": {"count": 1, "severities": [0.4], "onset": "random"},
            "misfire_single_cyl": {"count": 2, "severities": [0.5], "onset": "random"},
            "injector_clog": {"count": 1, "severities": [0.4], "onset": "random"},
            "sensor_faults": {"count": 1, "types": ["drift"], "sensors": ["oil_pressure"]},
            "ood_scenarios": {"count": 1, "conditions": ["combined_faults"]},
        }
        generate_all(scenarios=scenarios, seed=args.seed, output_dir=Path(args.output_dir) if args.output_dir else None, duration_override_min=args.duration_min)
    elif args.full:
        generate_all(full=True, seed=args.seed, output_dir=Path(args.output_dir) if args.output_dir else None, duration_override_min=args.duration_min)
    else:
        generate_all(full=args.full, seed=args.seed, output_dir=Path(args.output_dir) if args.output_dir else None, duration_override_min=args.duration_min)


if __name__ == "__main__":
    main()
