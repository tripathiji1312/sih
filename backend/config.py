"""
Central configuration — all constants, Rotax-912 parameters, and training hyperparams.
Single source of truth; imported everywhere.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_DATA_ROOT = DATA_ROOT / "training"
MODELS_ROOT = DATA_ROOT / "models"
RECORDINGS_ROOT = DATA_ROOT / "recordings"

# ---------------------------------------------------------------------------
# Rotax-912 ULS physical envelope (from published specs)
# ---------------------------------------------------------------------------
ROTAX_912 = {
    "displacement_cc": 1352,
    "displacement_l": 1.352,
    "max_rpm": 5800,
    "min_rpm": 1800,
    "idle_rpm": 1800,
    "cruise_rpm": 4800,
    "max_power_hp": 100,
    "cht_normal": (90.0, 135.0),  # °C
    "cht_limit": 150.0,
    "egt_normal": (600.0, 750.0),  # °C
    "egt_limit": 850.0,
    "oil_pressure_green": (30.0, 75.0),  # psi
    "oil_pressure_red_low": 12.0,
    "oil_pressure_red_high": 90.0,
    "oil_temp_normal": (50.0, 110.0),  # °C
    "oil_temp_limit": 130.0,
    "fuel_flow_cruise": (8.0, 28.0),  # L/hr
    "vibration_normal": (0.5, 2.5),  # g RMS
    "batt_voltage_normal": (12.5, 14.5),  # V
    "inj_timing_range": (20.0, 35.0),  # deg BTDC
    "altitude_max_m": 8500,
}

# ---------------------------------------------------------------------------
# Sampling / timing
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 10
DT_S = 1.0 / SAMPLE_RATE_HZ
WINDOW_SIZE = 30  # timesteps for ML input (3 seconds @10Hz)
WINDOW_STRIDE = 5  # stride for sliding window generation
N_RESIDUAL_CHANNELS = 14  # see ResidualVector definition
CIRCULAR_BUFFER_DURATION_S = 60.0

# ---------------------------------------------------------------------------
# Physics model constants (calibrated)
# ---------------------------------------------------------------------------
PHYSICS = {
    "tau_rpm": 0.8,  # RPM time constant (s)
    "t_combustion": 2100.0,  # peak combustion gas temp °C
    "k2_ambient_loss": 0.15,
    "k3_airspeed_cool": 0.008,
    "oil_viscosity_ref": 50.0,
}

# ---------------------------------------------------------------------------
# Fault classes — order matters; index = label
# ---------------------------------------------------------------------------
FAULT_CLASSES = [
    "healthy",              # 0
    "cooling_degradation",  # 1
    "lubrication_fault",    # 2
    "misfire_cyl1",         # 3
    "misfire_cyl2",         # 4
    "misfire_cyl3",         # 5
    "misfire_cyl4",         # 6
    "injector_clog",        # 7
    "sensor_fault",         # 8
]
N_CLASSES = len(FAULT_CLASSES)
FAULT_TO_IDX = {name: idx for idx, name in enumerate(FAULT_CLASSES)}
IDX_TO_FAULT = {idx: name for name, idx in FAULT_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Residual channel order — MUST be consistent across generate/train/infer
# ---------------------------------------------------------------------------
RESIDUAL_CHANNELS = [
    "rpm",
    "cht_1", "cht_2", "cht_3", "cht_4",
    "egt_1", "egt_2", "egt_3", "egt_4",
    "oil_p", "oil_t", "fuel_flow", "vibration", "batt_v",
]
assert len(RESIDUAL_CHANNELS) == N_RESIDUAL_CHANNELS

# Healthy residual stats (filled after data generation; used for z-score)
# Placeholder — computed dynamically in residual_generator.py and saved to
# data/models/residual_stats.npz during training data generation.
RESIDUAL_STATS_FILE = MODELS_ROOT / "residual_stats.npz"

# ---------------------------------------------------------------------------
# Training hyperparameters (SOTA defaults, override via CLI)
# ---------------------------------------------------------------------------
TRAINING = {
    "seed": 42,
    "batch_size": 64,
    "epochs": 60,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "annealing_epochs": 20,  # KL annealing for evidential loss
    "early_stopping_patience": 12,
    "scheduler": "cosine",  # cosine | plateau | none
    "optimizer": "adamw",  # adam | adamw
    "mixed_precision": True,
    "grad_clip": 1.0,
    "label_smoothing": 0.0,
    "mixup_alpha": 0.2,  # 0 = disabled
    "cutmix_alpha": 0.0,
    "aug_noise_std": 0.02,  # Gaussian noise augmentation
    "aug_scale_jitter": 0.05,
    "val_split": 0.15,
    "test_split": 0.15,
    "num_workers": 2,
    "use_class_weights": True,
    "focal_gamma": 0.0,  # 0 = disabled, 2.0 = focal loss
}

# OOD detector
OOD = {
    "threshold_percentile": 99.0,
    "shrinkage": 1e-4,  # covariance regularization
    "pca_components": None,  # None = no PCA; e.g. 64 = reduce dim before Mahalanobis
}

# Conformal prediction
CONFORMAL = {
    "alpha": 0.1,  # 90% coverage guarantee
}

# ---------------------------------------------------------------------------
# Data generation scenarios (mirrors archi.md Section 6.1)
# ---------------------------------------------------------------------------
SCENARIOS = {
    "healthy": {
        "count": 20,
        "duration_min": 10,  # reduced for fast iteration; use 60 for full
        "profiles": ["cruise", "climb", "descent", "loiter", "throttle_transitions"],
        "altitudes": [1000, 3000, 5000, 7000],
        "ambient_temps": [-20, 0, 15, 35, 45],
    },
    "cooling_degradation": {
        "count": 30,
        "severities": [0.3, 0.5, 0.7],
        "onset": "random_20-40min",
    },
    "lubrication_fault": {
        "count": 30,
        "severities": [0.2, 0.4, 0.6],
        "onset": "random_20-40min",
    },
    "misfire_single_cyl": {
        "count": 40,
        "severities": [0.3, 0.5, 0.8],
        "onset": "random_10-30min",
    },
    "injector_clog": {
        "count": 40,
        "severities": [0.2, 0.4, 0.6],
        "onset": "random_10-30min",
    },
    "sensor_faults": {
        "count": 30,
        "types": ["drift", "stuck", "noise", "spike"],
        "sensors": ["oil_pressure", "cht_1", "egt_2", "fuel_flow", "rpm"],
    },
    "ood_scenarios": {
        "count": 10,
        "conditions": [
            "extreme_altitude_8500m",
            "rapid_throttle_cycles",
            "combined_faults",
            "extreme_cold_-30C",
            "high_vibration_environment",
        ],
    },
}

# Full reference count for Kaggle (set FULL_DATA=True via env or CLI to use)
SCENARIOS_FULL = {
    "healthy": {
        "count": 20,
        "duration_min": 60,
        "profiles": ["cruise", "climb", "descent", "loiter", "throttle_transitions"],
        "altitudes": [1000, 3000, 5000, 7000],
        "ambient_temps": [-20, 0, 15, 35, 45],
    },
    "cooling_degradation": {"count": 30, "severities": [0.3, 0.5, 0.7], "onset": "random_20-40min"},
    "lubrication_fault": {"count": 30, "severities": [0.2, 0.4, 0.6], "onset": "random_20-40min"},
    "misfire_single_cyl": {"count": 40, "severities": [0.3, 0.5, 0.8], "onset": "random_10-30min"},
    "injector_clog": {"count": 40, "severities": [0.2, 0.4, 0.6], "onset": "random_10-30min"},
    "sensor_faults": {"count": 30, "types": ["drift", "stuck", "noise", "spike"], "sensors": ["oil_pressure", "cht_1", "egt_2", "fuel_flow", "rpm"]},
    "ood_scenarios": {"count": 10, "conditions": ["extreme_altitude_8500m", "rapid_throttle_cycles", "combined_faults", "extreme_cold_-30C", "high_vibration_environment"]},
}
