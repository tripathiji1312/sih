# SIH26054 — Complete SOTA Implementation Blueprint
## Digital Twin for MALE UAV Aero-Piston Engines: Physics-Anchored, Uncertainty-Aware, Edge-Ready

---

## Executive Architecture Summary

**The system is a five-layer, single-process digital twin: a fault-injectable Rotax-912-class sensor simulator feeds a physics expectation model whose residuals drive an evidential deep learning (EDL) uncertainty layer, all coordinated through a FastAPI/WebSocket backbone with a React dashboard — designed so every alert carries a cause, a confidence, and an honest acknowledgment of what the system doesn't know.**

This architecture synthesizes the current state-of-the-art across four research frontiers: (1) hybrid physics-data digital twins where semi-empirical models form the twin's core and are calibrated with operational data【turn1search12】, (2) evidential deep learning which directly predicts Dirichlet-distribution uncertainties and outperforms deep ensembles for fault-diagnosis OOD detection【turn0search5】【turn0search6】, (3) physics-informed conformal prediction providing distribution-free guaranteed uncertainty bounds【turn0search9】, and (4) lightweight edge-optimized transformer architectures purpose-built for resource-constrained real-time anomaly detection【turn1search5】【turn1search9】.

---

## 1. Final Technology Stack (Locked, No Alternatives)

| Component | Technology | Version | Why This Specific Choice |
|---|---|---|---|
| **Language** | Python | 3.11+ | Fastest ecosystem for scientific computing + web; async native |
| **Web framework** | FastAPI | 0.110+ | Native WebSocket support, async, auto OpenAPI docs, Pydantic validation【turn0search1】【turn0search4】 |
| **Frontend** | React + Vite | React 18, Vite 5 | Sub-second HMR during 24-hr build; WebSocket client built-in |
| **Charts** | Plotly.js (via react-plotly.js) | 2.35+ | Real-time streaming charts without full re-render |
| **Physics computation** | NumPy + SciPy + SymPy | latest | SymPy for deriving model equations symbolically before codegen |
| **State estimation** | FilterPy | 1.4.5+ | Kalman/UKF/particle filter implementations, well-documented【turn1search10】 |
| **ML framework** | PyTorch | 2.4+ | Evidential deep learning layers easier to implement than Keras; TorchScript for edge export |
| **ML serving** | ONNX Runtime | 1.18+ | Export trained models to ONNX for 3-5x faster CPU inference |
| **Time-series storage** | Parquet (via PyArrow) | latest | Columnar, compressed, fast range queries — no database server needed for 24-hr |
| **Dashboard state** | Zustand | 4.5+ | Lightweight state management for WebSocket data streaming |
| **Testing** | pytest + pytest-asyncio | latest | Async test support for FastAPI endpoints |
| **Packaging** | uv (pip replacement) | latest | 10-100x faster dependency resolution during hackathon |
| **Repo structure** | monorepo with `pnpm` workspaces | latest | Single `pnpm dev` starts both backend and frontend |

**What's deliberately NOT in the stack:** No MQTT broker, no InfluxDB, no Docker Compose, no Kubernetes, no message queue. For a 24-hour single-machine demo, every one of these adds an integration failure point without adding anything a judge evaluates. The Parquet files replace InfluxDB entirely for replay; FastAPI's built-in pub/sub via WebSocket replaces MQTT.

---

## 2. System Architecture — Five Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LAYER 5: OPERATOR INTERFACE                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐     │
│  │  React SPA   │  │  Health Score    │  │  2D SVG Engine     │     │
│  │  (Vite dev)  │  │  Panel (0-100)   │  │  Schematic         │     │
│  │  ← WebSocket │  │  + Confidence    │  │  (color-coded)     │     │
│  └──────────────┘  └──────────────────┘  └────────────────────┘     │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐     │
│  │  Alert Feed  │  │  Degradation     │  │  Watchdog Banner   │     │
│  │  (cause +    │  │  Trend + RUL     │  │  (data pipeline    │     │
│  │   confidence)│  │  + conf. band    │  │   health)          │     │
│  └──────────────┘  └──────────────────┘  └────────────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│                     LAYER 4: FUSION & DECISION ENGINE                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Evidence Aggregator                             │   │
│  │  Combines: physics residual + EDL prediction + sensor trust │   │
│  │  Outputs: Health Index + Confidence + FMEA attribution      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐   │
│  │  FMEA Causal Rules   │  │  Decision Arbiter                 │   │
│  │  Engine (pattern →   │  │  (engine fault vs sensor fault    │   │
│  │   cause → action)    │  │   vs novel condition → fallback)  │   │
│  └──────────────────────┘  └──────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     LAYER 3: INTELLIGENCE LAYER                       │
│  ┌────────────────────┐  ┌───────────────────────────────────┐     │
│  │  Evidential Deep   │  │  Degradation Tracker               │     │
│  │  Learning Model    │  │  (UKF on physics model params:     │     │
│  │  (Dirichlet output │  │   k1 cooling, η_c combustion,      │     │
│  │   → uncertainty)   │  │   friction coeff) → RUL projection │     │
│  └────────────────────┘  └───────────────────────────────────┘     │
│  ┌────────────────────┐  ┌───────────────────────────────────┐     │
│  │  OOD Detector      │  │  Sensor Trust Model                │     │
│  │  (Mahalanobis      │  │  (per-sensor agreement with        │     │
│  │   distance from    │  │   physics + correlated sensors)    │     │
│  │   training dist.)  │  │                                   │     │
│  └────────────────────┘  └───────────────────────────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│                     LAYER 2: PHYSICS & STATE ESTIMATION               │
│  ┌────────────────────┐  ┌───────────────────────────────────┐     │
│  │  Physics           │  │  Unscented Kalman Filter (UKF)     │     │
│  │  Expectation       │  │  Joint state + parameter estimation│     │
│  │  Model             │  │  Fuses: measured sensors ↔ physics │     │
│  │  (thermo/mech      │  │  Output: cleaned state + model     │     │
│  │   ODEs, Rotax-912  │  │         parameter estimates        │     │
│  │   calibrated)      │  │                                   │     │
│  └────────────────────┘  └───────────────────────────────────┘     │
│  ┌────────────────────┐  ┌───────────────────────────────────┐     │
│  │  Residual          │  │  Feature Extractor                  │     │
│  │  Generator         │  │  (rolling means, rates, spectral    │     │
│  │  (measured −       │  │   features from vibration, windowed │     │
│  │   expected)        │  │   stats)                            │     │
│  └────────────────────┘  └───────────────────────────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│                     LAYER 1: DATA INGESTION & VALIDATION              │
│  ┌────────────────────┐  ┌───────────────────────────────────┐     │
│  │  Sensor Simulator  │  │  Data Validator                    │     │
│  │  (fault-injectable,│  │  (range checks, rate limits,       │     │
│  │   Rotax-912-class) │  │   physical impossibility checks)   │     │
│  └────────────────────┘  └───────────────────────────────────┘     │
│  ┌────────────────────┐  ┌───────────────────────────────────┐     │
│  │  Watchdog          │  │  Circular Buffer                   │     │
│  │  (staleness timer  │  │  (last 60s of validated frames     │     │
│  │   per channel,     │  │   in memory for sliding-window     │     │
│  │   dropout counter) │  │   ML input)                        │     │
│  └────────────────────┘  └───────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow through the system (one 100ms tick):**
1. Simulator emits sensor frame → Validator checks physical plausibility → Watchdog stamps freshness
2. Frame appended to circular buffer → Feature extractor computes rolling statistics
3. Physics model (using UKF-estimated parameters from previous tick) predicts expected sensor values
4. Residual generator computes `residual[i] = measured[i] − expected[i]` for all 8 channels
5. EDL model takes 30-step window of residuals → outputs Dirichlet parameters → uncertainty
6. OOD detector checks Mahalanobis distance of current residual vector from training distribution
7. Sensor trust model checks per-sensor agreement with physics + correlated sensors
8. FMEA rules map residual *patterns* to fault causes
9. Evidence aggregator combines everything → Health Index + Confidence + Alert (if any)
10. WebSocket push to React dashboard → UI update in <50ms

---

## 3. Data Schema — Exact Definitions

### 3.1 Sensor Frame (Layer 1 → Layer 2)
```python
# Pydantic model — this IS the CAN frame equivalent
class SensorFrame(BaseModel):
    timestamp: float          # Unix epoch, ms precision
    mission_time_s: float     # seconds since mission start
    
    # Core engine parameters (Rotax-912 envelope)
    rpm: float                # 1800–5800 (redline 5800)
    cht_c: List[float]        # per-cylinder, 4 values, normal: 90–135°C
    egt_c: List[float]        # per-cylinder, 4 values, normal: 600–750°C
    oil_pressure_psi: float   # green arc: 30–75, red: <12 or >90
    oil_temp_c: float         # normal: 50–110°C
    fuel_flow_lph: float      # 8–28 L/hr cruise range
    vibration_g: float        # RMS, normal: 0.5–2.5g
    batt_voltage: float       # 12.5–14.5V (alternator charging)
    inj_timing_deg: float     # injection advance angle, 20–35° BTDC
    
    # Environmental context (from UAV avionics)
    altitude_m: float         # 0–8500m (MALE UAV ceiling)
    oat_c: float              # outside air temp
    airspeed_ms: float        # for cooling model
    
    # Simulator metadata (not in real system)
    fault_injected: Optional[str] = None  # for demo ground truth
```

### 3.2 Physics Model State (Layer 2 → Layer 3)
```python
class PhysicsState(BaseModel):
    # Estimated "true" engine state (from UKF)
    est_rpm: float
    est_cht_c: List[float]
    est_egt_c: List[float]
    est_oil_pressure_psi: float
    est_oil_temp_c: float
    
    # Estimated degradation parameters (slowly varying, tracked by UKF)
    # These ARE the digital twin's "memory" of engine wear
    cooling_efficiency_k1: float    # nominal 1.0, degrades to ~0.7
    combustion_efficiency: float    # nominal 1.0, degrades to ~0.85
    friction_coefficient: float     # nominal 1.0, increases with wear
    injector_health: List[float]    # per-cylinder, 1.0 → 0.6
    
    # Model confidence
    state_covariance: List[List[float]]  # UKF posterior covariance (diagonal sufficient)
```

### 3.3 Residual Vector (Layer 2 → Layer 3 ML input)
```python
class ResidualVector(BaseModel):
    timestamp: float
    rpm_residual: float           # measured - expected
    cht_residuals: List[float]    # per-cylinder
    egt_residuals: List[float]    # per-cylinder  
    oil_p_residual: float
    oil_t_residual: float
    fuel_flow_residual: float
    vibration_residual: float     # vs baseline spectral model
    
    # Normalized versions (z-score relative to healthy training data)
    normalized: List[float]       # 14-dim vector, this feeds the ML
```

### 3.4 ML Output with Uncertainty (Layer 3 → Layer 4)
```python
class DiagnosisOutput(BaseModel):
    timestamp: float
    
    # Evidential Deep Learning output
    fault_probability: float          # P(fault | data)
    epistemic_uncertainty: float       # "I don't know" uncertainty  
    aleatoric_uncertainty: float       # "Data is noisy" uncertainty
    dirichlet_params: List[float]     # raw evidence parameters
    
    # OOD detection
    mahalanobis_distance: float       # distance from training distribution
    is_ood: bool                      # True if outside 99% training envelope
    
    # Per-sensor trust scores
    sensor_trust: Dict[str, float]    # 0.0 (lying) to 1.0 (trusted)
    
    # Overall confidence in the diagnosis
    diagnosis_confidence: float       # combined metric, 0-1
```

### 3.5 Alert (Layer 4 → Layer 5)
```python
class Alert(BaseModel):
    timestamp: float
    severity: Literal["INFO", "CAUTION", "WARNING", "CRITICAL"]
    
    # FMEA attribution
    subsystem: Literal["cooling", "lubrication", "ignition", 
                       "combustion", "fuel_system", "sensors", "unknown"]
    fault_type: str                    # "cooling_degradation", "misfire", etc.
    causal_evidence: str               # human-readable: why we think this
    
    # Uncertainty tagging
    confidence: float                  # 0-1
    uncertainty_source: Literal["low", "model_disagreement", 
                                 "ood_input", "sensor_untrusted", "physics_fallback"]
    
    # Recommended action
    recommended_action: str            # "Reduce power to 75%, schedule inspection"
    
    # Fallback mode flag
    is_physics_fallback: bool          # True if ML abstained, physics-only mode
```

---

## 4. Core Component Implementations

### 4.1 Physics Expectation Model (Layer 2)

This is the "causal skeleton" — every equation maps to a physical mechanism, so it generalizes to unseen conditions.

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class EngineParams:
    """Slowly-varying degradation parameters, estimated by UKF"""
    k1_cooling: float = 1.0        # cooling efficiency multiplier
    eta_combustion: float = 1.0    # combustion efficiency
    mu_friction: float = 1.0       # friction coefficient
    inj_health: np.ndarray = None  # per-cylinder injector health [1.0]*4
    
    def __post_init__(self):
        if self.inj_health is None:
            self.inj_health = np.ones(4)

class PhysicsExpectationModel:
    """
    Simplified but causally-correct thermodynamic model.
    Calibrated to Rotax-912 ULS class:
    - 4-stroke, 4-cylinder, horizontally opposed
    - Air/oil cooled, 1352cc displacement
    - 100hp @ 5800 RPM (redline)
    - Fuel: AVGAS 100LL or MoGas
    """
    
    # Calibrated constants (adjust during pre-hackathon calibration)
    TAU_RPM = 0.8          # RPM time constant (s), load-dependent
    T_COMBUSTION = 2100.0   # peak combustion gas temp (°C), nominal
    K2_AMBIENT_LOSS = 0.15  # heat loss to ambient coefficient
    K3_AIRSPEED_COOL = 0.008  # airspeed cooling coefficient
    OIL_VISCOSITY_REF = 50.0  # reference oil temp for viscosity model
    
    def predict_rpm(self, rpm_current: float, rpm_commanded: float, 
                    load_fraction: float) -> float:
        """First-order lag: dRPM/dt = (RPM_cmd - RPM) / τ(load)"""
        tau = self.TAU_RPM * (1.0 + load_fraction * 0.5)  # heavier load → slower response
        dt = 0.1  # 10 Hz update
        return rpm_current + (rpm_commanded - rpm_current) * dt / tau
    
    def predict_cht(self, cht_current: float, rpm: float, 
                    ambient_temp_c: float, airspeed_ms: float,
                    params: EngineParams) -> float:
        """
        CHT dynamics: heat input from combustion, 
        losses to ambient and ram air cooling.
        """
        dt = 0.1
        # Combustion heat input scales with RPM²  and combustion efficiency
        t_effective = self.T_COMBUSTION * params.eta_combustion
        heat_input = (t_effective - cht_current) * (rpm / 5800.0)**2 * 0.02
        
        # Cooling losses (degraded by k1_cooling parameter)
        ambient_loss = (cht_current - ambient_temp_c) * self.K2_AMBIENT_LOSS * params.k1_cooling
        airspeed_loss = (cht_current - ambient_temp_c) * self.K3_AIRSPEED_COOL * airspeed_ms * params.k1_cooling
        
        dcht = heat_input - ambient_loss - airspeed_loss
        return cht_current + dcht * dt
    
    def predict_egt(self, rpm: float, mixture_ratio: float, 
                    inj_health_cyl: float, params: EngineParams) -> float:
        """
        EGT depends on: combustion efficiency, mixture, RPM.
        Degraded injector → lean condition → EGT changes.
        """
        base_egt = 650.0 + (rpm - 5000) * 0.05  # RPM effect
        mixture_effect = (mixture_ratio - 1.0) * 100.0  # rich→cooler, lean→hotter
        injector_effect = (1.0 - inj_health_cyl) * 150.0  # clogged injector → lean → EGT up
        efficiency_effect = (1.0 - params.eta_combustion) * 200.0
        return base_egt + mixture_effect + injector_effect + efficiency_effect
    
    def predict_oil_pressure(self, rpm: float, oil_temp_c: float) -> float:
        """
        Oil pressure: pump pressure (RPM-dependent) minus 
        viscosity loss (temperature-dependent).
        """
        # Pump pressure curve (empirical, linear in RPM)
        pump_pressure = 2.0 + rpm * 0.012  # ~25 psi at 2000, ~70 at 5800
        
        # Viscosity drop with temp → pressure drop
        temp_factor = max(0.3, 1.0 - (oil_temp_c - 50.0) * 0.004)
        
        return pump_pressure * temp_factor
    
    def predict_fuel_flow(self, rpm: float, manifold_pressure: float,
                          inj_health_avg: float) -> float:
        """Fuel flow: volumetric efficiency × displacement × RPM × mixture"""
        displacement_l = 1.352  # Rotax 912
        ve = 0.85 * inj_health_avg  # volumetric efficiency, affected by injectors
        cycles_per_sec = rpm / 60.0 / 2.0  # 4-stroke
        fuel_per_cycle_l = 3.0e-5  # ~30cc per cylinder-cycle at cruise
        return displacement_l * ve * cycles_per_sec * fuel_per_cycle_l * 3600  # L/hr
```

### 4.2 Unscented Kalman Filter — Joint State & Parameter Estimation

This is what makes it a *twin* rather than a monitor — the UKF continuously adapts the physics model's degradation parameters to match the real engine.

```python
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np

class TwinStateEstimator:
    """
    Joint estimation of engine state (fast-varying) and 
    degradation parameters (slow-varying) using UKF.
    
    State vector (14-dim):
    [rpm, cht1-4, egt1-4, oil_p, oil_t, k1_cooling, eta_comb, mu_friction]
    
    The degradation parameters are the digital twin's "memory" —
    they encode accumulated wear and are the basis for RUL projection.
    """
    
    def __init__(self):
        n_states = 14
        
        # Sigma point parameters (tuned for this problem)
        self.points = MerweScaledSigmaPoints(n=n_states, alpha=0.1, 
                                              beta=2.0, kappa=0.0)
        
        self.ukf = UnscentedKalmanFilter(
            dim_x=n_states, dim_z=11,  # 11 measurements (no k1, eta, mu directly)
            dt=0.1, hx=self._measurement_function, 
            fx=self._state_transition, points=self.points
        )
        
        # Initial state: nominal engine
        self.ukf.x = np.array([
            5000,       # rpm
            110, 110, 110, 110,  # cht per cylinder
            680, 680, 680, 680,  # egt per cylinder
            55.0,       # oil pressure
            80.0,       # oil temp
            1.0,        # k1_cooling (degradation param)
            1.0,        # eta_combustion
            1.0         # mu_friction
        ])
        
        # Process noise: degradation params vary slowly (small Q)
        self.ukf.Q = np.diag([
            100.0,     # rpm process noise
            1.0, 1.0, 1.0, 1.0,  # cht
            25.0, 25.0, 25.0, 25.0,  # egt
            4.0,       # oil pressure
            1.0,       # oil temp
            1e-6,      # k1_cooling (very slow)
            1e-6,      # eta_combustion (very slow)
            1e-6       # mu_friction (very slow)
        ])
        
        # Measurement noise (sensor accuracy)
        self.ukf.R = np.diag([
            20.0,       # rpm ±5
            4.0, 4.0, 4.0, 4.0,   # cht ±2°C
            100.0, 100.0, 100.0, 100.0,  # egt ±10°C
            4.0,        # oil pressure ±2 psi
            1.0         # oil temp ±1°C
        ])
    
    def _state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition: apply physics model for one timestep.
        Degradation parameters are random-walk (slowly varying).
        """
        # ... (uses PhysicsExpectationModel from above)
        # Extract states, run physics, reconstruct
        # Degradation params: x[i] += small noise
        return x_next
    
    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """Measurement: directly observe 11 of 14 states (not degradation params)"""
        return x[:11]
    
    def update(self, measurement: np.ndarray):
        """Call every sensor frame — updates state + degradation estimates"""
        self.ukf.predict()
        self.ukf.update(measurement)
        return self.ukf.x, self.ukf.P  # state + covariance
    
    def get_degradation_params(self) -> dict:
        """Extract current degradation parameter estimates"""
        return {
            "k1_cooling": self.ukf.x[11],
            "eta_combustion": self.ukf.x[12],
            "mu_friction": self.ukf.x[13],
            "param_uncertainty": np.sqrt(np.diag(self.ukf.P)[11:])
        }
```

### 4.3 Evidential Deep Learning Model (Layer 3)

This is the SOTA upgrade over deep ensembles — EDL directly outputs a Dirichlet distribution over class probabilities, where the concentration parameters encode evidence, and low evidence flags uncertainty without needing an ensemble【turn0search5】【turn0search6】.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialFaultClassifier(nn.Module):
    """
    Evidential Deep Learning for fault diagnosis.
    
    Instead of softmax (which always outputs confident probabilities),
    this outputs Dirichlet parameters (evidence) for each class.
    High total evidence → confident prediction.
    Low total evidence → model says "I don't know."
    
    This is the 2024-2025 SOTA for uncertainty-aware fault diagnosis
    in non-stationary industrial environments【turn0search14】.
    
    Architecture: 1D-CNN feature extractor + evidential head
    Input: (batch, 14_channels, 30_timesteps) — residual windows
    Output: Dirichlet parameters for K fault classes + healthy
    """
    
    def __init__(self, n_channels=14, n_timesteps=30, n_classes=7):
        super().__init__()
        
        # Lightweight 1D CNN (edge-deployable)
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        
        # Evidential head: outputs K Dirichlet concentration parameters
        self.evidence_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
            nn.Softplus()  # Evidence must be non-negative
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns dict with:
        - evidence: Dirichlet concentration params (K-dim)
        - alpha: evidence + 1 (Dirichlet params)
        - probs: expected class probabilities (K-dim)
        - epistemic_uncertainty: K / sum(alpha) — high when evidence low
        - aleatoric_uncertainty: from Dirichlet entropy
        """
        feat = self.features(x).squeeze(-1)
        evidence = self.evidence_head(feat)
        
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength
        
        # Expected probabilities (mean of Dirichlet)
        probs = alpha / S
        
        # Epistemic uncertainty: how much total evidence do we have?
        # S → ∞: certain; S → K (all evidence=0): maximally uncertain
        epistemic = n_classes := alpha.shape[-1]
        epistemic = alpha.shape[-1] / S.squeeze(-1)  # K/S
        
        # Aleatoric uncertainty: irreducible noise (from Dirichlet entropy)
        # This captures "data is noisy" not "I don't know"
        digamma = torch.digamma(alpha)
        aleatoric = -(alpha * (digamma - torch.digamma(S))).sum(-1)
        
        return {
            "evidence": evidence,
            "alpha": alpha,
            "probs": probs,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
        }

# Loss function for evidential training (from Sensoy et al. 2018, updated)
def evidential_loss(output: dict, target: torch.Tensor, 
                    epoch: int, annealing_epochs: int = 10) -> torch.Tensor:
    """
    Combined loss: MSE on Dirichlet params + KL divergence regularization.
    Anneals the KL term over epochs to prevent premature uncertainty collapse.
    """
    alpha = output["alpha"]
    
    # One-hot target
    y = F.one_hot(target, num_classes=alpha.shape[-1]).float()
    
    # Error term: penalize wrong evidence
    S = alpha.sum(-1, keepdim=True)
    err = (y - alpha / S).pow(2).sum(-1).mean()
    
    # KL regularization: prevent overconfidence
    # Anneal: start with strong KL, reduce over epochs
    annealing_coeff = min(1.0, epoch / annealing_epochs)
    kl = kl_dirichlet_uniform(alpha).mean() * annealing_coeff
    
    return err + kl

def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(D(alpha) || D(1)) — divergence from uniform Dirichlet"""
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(-1, keepdim=True)
    S_beta = beta.sum(-1, keepdim=True)
    
    t1 = torch.lgamma(S_alpha) - torch.lgamma(S_beta)
    t2 = (torch.lgamma(alpha) - torch.lgamma(beta)).sum(-1, keepdim=True)
    t3 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(-1, keepdim=True)
    
    return t1 - t2 + t3
```

### 4.4 Sensor Trust Model (Layer 3)

```python
class SensorTrustEvaluator:
    """
    Answers: "Is the sensor lying, or is the engine actually failing?"
    
    Logic: Each sensor has physical relationships with other sensors.
    If one sensor diverges but all correlated sensors agree with physics,
    → sensor fault. If multiple correlated sensors diverge in a 
    physically consistent direction → engine fault.
    """
    
    # Correlation groups: sensors that should move together
    CORRELATION_GROUPS = {
        "thermal": ["cht_1", "cht_2", "cht_3", "cht_4", "oil_temp", "egt_1"],
        "lubrication": ["oil_pressure", "oil_temp", "rpm"],  # oil_p follows RPM
        "combustion": ["egt_1", "egt_2", "egt_3", "egt_4", "fuel_flow"],
        "mechanical": ["rpm", "vibration", "oil_pressure"],
    }
    
    def evaluate_trust(self, residuals: dict, physics_state: dict) -> dict:
        """
        Per-sensor trust score in [0, 1].
        1.0 = sensor agrees with physics + correlated sensors
        0.0 = sensor contradicts everything (likely sensor fault)
        """
        trust_scores = {}
        
        for sensor in residuals:
            sensor_residual = residuals[sensor]
            
            # Find correlated sensors
            correlated = self._get_correlated_sensors(sensor)
            correlated_residuals = [residuals[s] for s in correlated if s in residuals]
            
            # Does the sensor agree with physics? (residual small)
            physics_agreement = 1.0 - min(1.0, abs(sensor_residual) / self._get_threshold(sensor))
            
            # Do correlated sensors show similar divergence?
            if correlated_residuals:
                corr_agreement = 1.0 - min(1.0, abs(sensor_residual - np.mean(correlated_residuals)) 
                                           / self._get_threshold(sensor))
            else:
                corr_agreement = 1.0
            
            # Combined trust: both must agree
            trust = physics_agreement * corr_agreement
            trust_scores[sensor] = trust
        
        return trust_scores
    
    def arbitrate(self, trust_scores: dict, residuals: dict) -> str:
        """
        Decision: engine fault vs sensor fault vs uncertain.
        Returns one of: "engine_fault", "sensor_fault", "uncertain"
        """
        n_untrusted = sum(1 for t in trust_scores.values() if t < 0.3)
        
        if n_untrusted == 0:
            return "normal" if all(abs(r) < 0.5 for r in residuals.values()) else "engine_fault"
        
        if n_untrusted == 1:
            # One sensor diverging, others agree with physics → sensor fault
            return "sensor_fault"
        
        if n_untrusted >= 2:
            # Multiple sensors diverging — check if physically consistent
            if self._physically_consistent(residuals):
                return "engine_fault"
            else:
                return "sensor_fault"  # inconsistent divergence = sensor issue
        
        return "uncertain"
```

### 4.5 FMEA Causal Rules Engine (Layer 4)

```python
class FMEAAttributionEngine:
    """
    Maps residual *patterns* to fault causes using FMEA logic.
    Not ML — this is domain knowledge encoded as rules.
    Every alert gets a human-readable causal explanation.
    """
    
    RULES = [
        {
            "name": "cooling_degradation",
            "subsystem": "cooling",
            "condition": lambda r: (
                np.mean([abs(r["cht_1"]), abs(r["cht_2"]), 
                        abs(r["cht_3"]), abs(r["cht_4"])]) > 0.7 and
                abs(r["oil_pressure"]) < 0.3 and  # oil pressure normal
                abs(r["rpm"]) < 0.3               # RPM normal
            ),
            "explanation": "CHT residuals elevated across all cylinders while oil "
                          "pressure and RPM remain nominal — indicates cooling "
                          "system efficiency loss, not combustion or mechanical issue.",
            "action": "Reduce power to 75%, increase airspeed for cooling. "
                     "Schedule cooling system inspection (baffles, radiator, "
                     "coolant) within next 10 flight hours."
        },
        {
            "name": "lubrication_fault",
            "subsystem": "lubrication", 
            "condition": lambda r: (
                r["oil_pressure"] < -0.6 and  # pressure low
                r["oil_temp"] > 0.5 and        # temp high (friction)
                abs(r["rpm"]) < 0.3
            ),
            "explanation": "Oil pressure below expected with concurrent oil "
                          "temperature elevation — indicates lubrication system "
                          "degradation (pump wear, oil leak, or bearing friction increase).",
            "action": "CRITICAL: Reduce power immediately. Monitor oil pressure "
                     "closely. If pressure continues to drop, prepare for "
                     "precautionary landing. Do not continue mission."
        },
        {
            "name": "misfire_cylinder_{i}",
            "subsystem": "ignition",
            "condition": lambda r, i: (
                r[f"egt_{i}"] > 0.8 and           # EGT spike on one cyl
                r[f"egt_{i}"] - np.mean([r[f"egt_{j}"] for j in range(1,5) if j != i]) > 0.5
            ),
            "explanation": "EGT residual spike on cylinder {i} while other cylinders "
                          "remain nominal — indicates misfire (spark plug, ignition "
                          "lead, or injector issue on this cylinder).",
            "action": "Run mag check if possible. Cylinder {i} misfire confirmed. "
                     "Reduce power to minimize vibration. Schedule cylinder {i} "
                     "inspection (plugs, leads, injector) before next flight."
        },
        {
            "name": "sensor_fault_{sensor}",
            "subsystem": "sensors",
            "condition": "dynamic",  # handled by SensorTrustEvaluator
            "explanation": "Sensor {sensor} output diverges from physics model "
                          "prediction and from all correlated sensors — sensor "
                          "or wiring fault, not engine fault.",
            "action": "Verify sensor wiring and connection. If sensor confirmed "
                     "faulty, replace before relying on this parameter for "
                     "engine health monitoring."
        },
    ]
    
    def attribute(self, residuals: dict, sensor_arbitration: str) -> Optional[Alert]:
        """Try to attribute fault to a specific cause; return None if pattern unknown"""
        if sensor_arbitration == "sensor_fault":
            # Identify which sensor
            untrusted = [s for s, t in self.trust_scores.items() if t < 0.3]
            if untrusted:
                return self._make_sensor_alert(untrusted[0])
            return None
        
        # Try each rule
        for rule in self.RULES:
            if rule["condition"] == "dynamic":
                continue
            if rule["condition"](residuals):
                return self._make_alert(rule)
        
        return None  # Unknown pattern → trigger uncertainty fallback
```

### 4.6 Watchdog (Layer 1)

```python
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ChannelWatchdog:
    """Per-sensor-channel staleness and dropout tracking"""
    channel_name: str
    max_staleness_s: float = 2.0      # flag if no update in 2s
    last_update_time: float = 0.0
    expected_rate_hz: float = 10.0
    frame_count: int = 0
    dropout_count: int = 0
    malformed_count: int = 0
    
    # Rolling window of inter-frame intervals for rate monitoring
    _intervals: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def on_frame(self, timestamp: float, valid: bool = True):
        if not valid:
            self.malformed_count += 1
            self.dropout_count += 1
            return
        
        if self.last_update_time > 0:
            interval = timestamp - self.last_update_time
            self._intervals.append(interval)
        
        self.last_update_time = timestamp
        self.frame_count += 1
    
    def check_health(self, current_time: float) -> dict:
        staleness = current_time - self.last_update_time
        is_stale = staleness > self.max_staleness_s
        
        # Rate check
        if len(self._intervals) >= 10:
            avg_interval = np.mean(list(self._intervals))
            measured_rate = 1.0 / avg_interval
            rate_ok = measured_rate > self.expected_rate_hz * 0.5  # 50% tolerance
        else:
            measured_rate = 0.0
            rate_ok = True  # not enough data yet
        
        return {
            "channel": self.channel_name,
            "staleness_s": round(staleness, 2),
            "is_stale": is_stale,
            "measured_rate_hz": round(measured_rate, 1),
            "rate_ok": rate_ok,
            "dropouts": self.dropout_count,
            "malformed": self.malformed_count,
            "status": "OK" if (not is_stale and rate_ok) else "DEGRADED" if not is_stale else "STALE"
        }

class SystemWatchdog:
    """Aggregate watchdog across all channels — feeds dashboard banner"""
    
    def __init__(self):
        self.channels = {
            "rpm": ChannelWatchdog("rpm"),
            "cht": ChannelWatchdog("cht"),
            "egt": ChannelWatchdog("egt"),
            "oil": ChannelWatchdog("oil"),
            "fuel": ChannelWatchdog("fuel"),
            "vibration": ChannelWatchdog("vibration"),
        }
    
    def system_health(self) -> dict:
        current = time.time()
        channel_health = {name: wd.check_health(current) 
                         for name, wd in self.channels.items()}
        
        n_stale = sum(1 for h in channel_health.values() if h["is_stale"])
        n_degraded = sum(1 for h in channel_health.values() 
                        if h["status"] == "DEGRADED")
        
        if n_stale >= 3:
            overall = "CRITICAL_DATA_LOSS"
        elif n_stale >= 1 or n_degraded >= 3:
            overall = "DATA_DEGRADED"
        else:
            overall = "HEALTHY"
        
        return {
            "overall_status": overall,
            "channels": channel_health,
            "recommendation": {
                "HEALTHY": "Data pipeline nominal",
                "DATA_DEGRADED": "Some channels degraded — ML confidence reduced",
                "CRITICAL_DATA_LOSS": "Multiple channels stale — falling back to physics-only mode"
            }[overall]
        }
```

---

## 5. Complete Project Folder Structure

```
sih26054-digital-twin/
├── README.md
├── pyproject.toml              # uv/pip dependencies
├── package.json                # pnpm workspace root
├── pnpm-workspace.yaml
│
├── backend/                    # Python backend (single process)
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # All constants, Rotax-912 parameters
│   │
│   ├── simulator/              # Layer 1: Data source
│   │   ├── __init__.py
│   │   ├── engine_simulator.py     # Physics-based sensor data generator
│   │   ├── fault_injector.py       # Misfire, cooling, lubrication, injector, sensor faults
│   │   └── mission_profiles.py     # High-altitude, endurance, hot-weather, throttle transitions
│   │
│   ├── ingestion/              # Layer 1: Validation
│   │   ├── __init__.py
│   │   ├── validator.py            # Physical plausibility checks
│   │   ├── watchdog.py             # ChannelWatchdog + SystemWatchdog (above)
│   │   └── circular_buffer.py      # 60s rolling window for ML input
│   │
│   ├── physics/                # Layer 2: Physics model
│   │   ├── __init__.py
│   │   ├── expectation_model.py    # PhysicsExpectationModel (above)
│   │   ├── state_estimator.py      # TwinStateEstimator / UKF (above)
│   │   ├── residual_generator.py   # Measured - Expected computation
│   │   └── calibration.py          # Pre-hackathon: calibrate against sim
│   │
│   ├── ml/                     # Layer 3: Intelligence
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── evidential_model.py    # EvidentialFaultClassifier (above)
│   │   │   ├── ood_detector.py        # Mahalanobis distance OOD
│   │   │   └── sensor_trust.py        # SensorTrustEvaluator (above)
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── generate_training_data.py  # Run simulator with fault injection
│   │   │   ├── train_evidential.py         # Train EDL model
│   │   │   └── calibrate_ood.py            # Fit Mahalanobis on healthy data
│   │   └── inference.py             # Load ONNX model, run inference
│   │
│   ├── fusion/                 # Layer 4: Decision
│   │   ├── __init__.py
│   │   ├── fMEA_rules.py            # FMEAAttributionEngine (above)
│   │   ├── evidence_aggregator.py   # Combine physics + ML + sensor trust
│   │   ├── decision_arbiter.py      # Engine fault vs sensor fault vs fallback
│   │   └── health_index.py          # Aggregate to 0-100 score with confidence
│   │
│   ├── api/                    # WebSocket + REST endpoints
│   │   ├── __init__.py
│   │   ├── websocket.py             # Real-time streaming to frontend
│   │   ├── routes.py                # REST endpoints (replay, missions)
│   │   └── schemas.py               # Pydantic models (SensorFrame, Alert, etc.)
│   │
│   └── replay/                 # Mission replay system
│       ├── __init__.py
│       ├── recorder.py              # Save sessions to Parquet
│       └── player.py                # Replay with speed control
│
├── frontend/                   # React SPA
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── stores/
│   │   │   ├── useTelemetryStore.ts    # Zustand: WebSocket data
│   │   │   ├── useAlertStore.ts
│   │   │   └── useWatchdogStore.ts
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts         # WS connection with reconnect
│   │   │   └── useAnimationFrame.ts
│   │   ├── components/
│   │   │   ├── HealthScore.tsx         # Big gauge: overall health
│   │   │   ├── ConfidenceTag.tsx       # Confidence indicator per alert
│   │   │   ├── EngineSchematic.tsx     # 2D SVG color-coded engine
│   │   │   ├── AlertFeed.tsx           # Scrollable alert list
│   │   │   ├── DegradationChart.tsx    # Trend + RUL projection
│   │   │   ├── WatchdogBanner.tsx      # Data pipeline status
│   │   │   ├── PhysicsOverlay.tsx      # Measured vs Expected overlay
│   │   │   └── MissionControls.tsx     # Fault injection for demo
│   │   └── utils/
│   │       ├── websocket.ts
│   │       └── formatters.ts
│   └── index.html
│
├── data/                       # Generated during training, gitignored
│   ├── training/
│   │   ├── healthy/            # Normal operation sequences
│   │   ├── faults/
│   │   │   ├── cooling/        # Various severities
│   │   │   ├── lubrication/
│   │   │   ├── misfire/
│   │   │   ├── injector/
│   │   │   └── sensor/
│   │   └── ood/                # Out-of-distribution scenarios
│   ├── models/
│   │   ├── evidential_model.onnx
│   │   └── ood_stats.npz       # Mahalanobis mean/covariance
│   └── recordings/             # Demo session recordings (Parquet)
│
├── scripts/                    # Development utilities
│   ├── calibrate_physics.py    # Day-0: physics calibration script
│   ├── generate_all_data.py    # Generate training datasets
│   ├── train_models.py         # Train EDL + calibrate OOD
│   └── validate_models.py      # Sanity checks
│
├── tests/
│   ├── test_physics.py
│   ├── test_evidential.py
│   ├── test_watchdog.py
│   └── test_fmea_rules.py
│
├── docs/
│   ├── architecture.md         # This document
│   ├── deployment_roadmap.md   # GCS → test rig → fleet
│   ├── what_we_didnt_build.md  # Scope discipline documentation
│   └── demo_script.md          # 5-minute pitch script
│
└── .gitignore
```

---

## 6. Model Training Pipeline (Pre-Hackathon + Early Hackathon)

### 6.1 Data Generation Strategy

```python
# scripts/generate_all_data.py
"""
Generates all training data from the physics simulator with fault injection.
NO external datasets needed — everything is physics-consistent.
"""

SCENARIOS = {
    # Healthy operation: 20 missions × 60 min × 10 Hz = 720,000 frames
    "healthy": {
        "count": 20,
        "duration_min": 60,
        "profiles": ["cruise", "climb", "descent", "loiter", "throttle_transitions"],
        "altitudes": [1000, 3000, 5000, 7000],  # meters
        "ambient_temps": [-20, 0, 15, 35, 45],   # °C — includes hot weather
    },
    
    # Fault scenarios: 10 missions per fault type per severity
    "cooling_degradation": {
        "count": 30,  # 3 severities × 10 missions
        "severities": [0.3, 0.5, 0.7],  # k1_cooling reduction
        "onset": "random_20-40min",  # fault starts mid-mission
    },
    
    "lubrication_fault": {
        "count": 30,
        "severities": [0.2, 0.4, 0.6],  # oil pressure reduction
        "onset": "random_20-40min",
    },
    
    "misfire_single_cyl": {
        "count": 40,  # 4 cylinders × 10 missions
        "severities": [0.3, 0.5, 0.8],  # combustion efficiency loss on one cyl
        "onset": "random_10-30min",
    },
    
    "injector_clog": {
        "count": 40,  # 4 cylinders × 10 missions
        "severities": [0.2, 0.4, 0.6],  # flow restriction
        "onset": "random_10-30min",
    },
    
    "sensor_faults": {
        "count": 30,  # Each sensor type × various failure modes
        "types": ["drift", "stuck", "noise", "spike"],
        "sensors": ["oil_pressure", "cht_1", "egt_2", "fuel_flow", "rpm"],
    },
    
    # OOD scenarios: conditions NOT in training, for uncertainty evaluation
    "ood_scenarios": {
        "count": 10,
        "conditions": [
            "extreme_altitude_8500m",      # Above typical training range
            "rapid_throttle_cycles",        # Aggressive transitions
            "combined_faults",              # Two simultaneous faults
            "extreme_cold_-30C",            # Cold weather start
            "high_vibration_environment",   # Turbulence simulation
        ]
    }
}
```

### 6.2 Training Script

```python
# scripts/train_models.py
"""
Trains the evidential model on generated data.
Total time: ~30-45 minutes on CPU, ~10 min on GPU.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def prepare_training_data():
    """
    Load all scenarios, compute residuals against healthy physics model,
    create sliding windows for ML input.
    """
    X_windows = []  # (N, 14, 30) — 14 residual channels, 30 timestep window
    y_labels = []   # (N,) — 0=healthy, 1=cooling, 2=lubrication, 3=misfire, etc.
    y_confidence = []  # (N,) — 1.0 for clear faults, 0.5 for ambiguous
    
    # ... load each scenario, run physics model, compute residuals, 
    # create sliding windows, assign labels
    
    return np.array(X_windows), np.array(y_labels)

def train_evidential_model(X, y, epochs=50, batch_size=64):
    """
    Train with evidential loss (annealed KL).
    The annealing is critical: start with weak KL so model learns,
    then strengthen to calibrate uncertainty.
    """
    model = EvidentialFaultClassifier(n_channels=14, n_timesteps=30, n_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            output = model(batch_x)
            loss = evidential_loss(output, batch_y, epoch, annealing_epochs=20)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation: check calibration
        if epoch % 10 == 0:
            val_metrics = evaluate_calibration(model, val_loader)
            print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}, "
                  f"val_accuracy={val_metrics['accuracy']:.3f}, "
                  f"val_ece={val_metrics['ece']:.3f}")  # Expected Calibration Error
    
    return model

def calibrate_ood_detector(X_healthy):
    """
    Fit Mahalanobis distance statistics on healthy residual data.
    At runtime: if Mahalanobis distance > threshold (99th percentile),
    flag as OOD → low confidence.
    """
    # Flatten windows to feature vectors
    X_flat = X_healthy.reshape(len(X_healthy), -1)
    
    mean = X_flat.mean(axis=0)
    cov = np.cov(X_flat.T) + 1e-6 * np.eye(X_flat.shape[1])  # regularization
    cov_inv = np.linalg.inv(cov)
    
    # Compute distances on healthy data to set threshold
    distances = [np.sqrt((x - mean) @ cov_inv @ (x - mean)) for x in X_flat]
    threshold_99 = np.percentile(distances, 99)
    
    return {"mean": mean, "cov_inv": cov_inv, "threshold": threshold_99}
```

---

## 7. FastAPI Backend — Main Application

```python
# backend/main.py
"""
Single-process FastAPI application.
Runs: simulator → ingestion → physics → ML → fusion → WebSocket stream.
All in one asyncio event loop — no external services needed.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time
import json
from .simulator import EngineSimulator, FaultInjector
from .ingestion import DataValidator, SystemWatchdog, CircularBuffer
from .physics import PhysicsExpectationModel, TwinStateEstimator, ResidualGenerator
from .ml import EvidentialModelInference, OODDetector, SensorTrustEvaluator
from .fusion import EvidenceAggregator, FMEAAttributionEngine, DecisionArbiter
from .api.schemas import SensorFrame, DiagnosisOutput, Alert, WatchdogStatus

app = FastAPI(title="Digital Twin — MALE UAV Aero-Piston Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_websockets=True,
)

# --- Component initialization (singleton) ---
simulator = EngineSimulator()
fault_injector = FaultInjector(simulator)
validator = DataValidator()
watchdog = SystemWatchdog()
buffer = CircularBuffer(duration_s=60.0)
physics_model = PhysicsExpectationModel()
state_estimator = TwinStateEstimator()
residual_gen = ResidualGenerator(physics_model)
ml_model = EvidentialModelInference()  # loads ONNX
ood_detector = OODDetector()  # loads calibration stats
sensor_trust = SensorTrustEvaluator()
fmea_engine = FMEAAttributionEngine()
decision_arbiter = DecisionArbiter()
evidence_aggregator = EvidenceAggregator()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# --- Main processing loop ---
async def processing_loop():
    """
    Runs at 10 Hz — processes one sensor frame through the entire pipeline.
    This is the heartbeat of the digital twin.
    """
    while True:
        tick_start = time.time()
        
        # 1. Get sensor frame from simulator (or CAN bus in production)
        frame = simulator.get_frame()
        
        # 2. Validate + watchdog
        is_valid = validator.validate(frame)
        watchdog.on_frame(frame)
        watchdog_status = watchdog.system_health()
        
        # 3. Add to circular buffer
        buffer.append(frame)
        
        # 4. Physics prediction using UKF-estimated parameters
        degradation_params = state_estimator.get_degradation_params()
        expected = physics_model.predict_all(frame, degradation_params)
        
        # 5. UKF state update (fuse measurement with prediction)
        state_estimate, covariance = state_estimator.update(frame.to_vector())
        
        # 6. Compute residuals
        residuals = residual_gen.compute(frame, expected)
        
        # 7. ML inference (only if we have enough window data)
        if buffer.is_full():
            window = buffer.get_ml_window()  # (14, 30) tensor
            ml_output = ml_model.infer(window)
            ood_result = ood_detector.check(residuals.to_vector())
            
            # 8. Sensor trust evaluation
            trust_scores = sensor_trust.evaluate_trust(
                residuals.to_dict(), state_estimate
            )
            arbitration = decision_arbiter.arbitrate(trust_scores, residuals)
            
            # 9. FMEA attribution
            alert = fmea_engine.attribute(residuals, arbitration)
            
            # 10. Evidence aggregation → Health Index
            health = evidence_aggregator.compute(
                residuals, ml_output, trust_scores, 
                ood_result, arbitration, watchdog_status
            )
            
            # 11. Broadcast to all connected clients
            await manager.broadcast({
                "type": "telemetry",
                "timestamp": frame.timestamp,
                "measured": frame.dict(),
                "expected": expected.dict(),
                "residuals": residuals.to_dict(),
                "ml_output": ml_output.dict(),
                "sensor_trust": trust_scores,
                "arbitration": arbitration,
                "alert": alert.dict() if alert else None,
                "health": health.dict(),
                "watchdog": watchdog_status,
                "degradation_params": degradation_params,
            })
        
        # 10 Hz tick
        elapsed = time.time() - tick_start
        await asyncio.sleep(max(0, 0.1 - elapsed))

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(processing_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Listen for client commands (fault injection for demo)
            data = await websocket.receive_json()
            if data.get("command") == "inject_fault":
                fault_injector.inject(data["fault_type"], data.get("severity", 0.5))
            elif data.get("command") == "clear_fault":
                fault_injector.clear()
            elif data.get("command") == "set_mission":
                simulator.set_mission_profile(data["profile"])
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# REST endpoints for replay and mission management
@app.get("/api/missions")
async def list_missions():
    return simulator.available_missions()

@app.post("/api/replay/{session_id}")
async def replay_session(session_id: str, speed: float = 1.0):
    """Replay a recorded session at given speed"""
    # Implementation uses Parquet reader + same processing pipeline
    pass
```

---

## 8. React Dashboard — Key Components

```typescript
// frontend/src/components/EngineSchematic.tsx
/**
 * 2D SVG engine schematic, color-coded by subsystem health.
 * This is what makes it READ as a digital twin, not a dashboard.
 * 
 * Layout: 4-cylinder horizontally-opposed (like Rotax 912)
 * Each cylinder colored by its CHT/EGT health
 * Oil system, cooling, fuel system all visible
 */

import { useTelemetryStore } from '../stores/useTelemetryStore';

const SUBSYSTEM_COLORS = {
  healthy: '#22c55e',      // green
  caution: '#eab308',      // yellow  
  warning: '#f97316',      // orange
  critical: '#ef4444',     // red
  unknown: '#6b7280',      // gray (sensor untrusted)
  physics_fallback: '#3b82f6',  // blue (ML abstained)
};

function EngineSchematic() {
  const { health, sensorTrust, alerts } = useTelemetryStore();
  
  const getCylinderColor = (cylNum: number) => {
    const cylHealth = health.subsystems[`cylinder_${cylNum}`];
    const sensorTrusted = sensorTrust[`cht_${cylNum}`] > 0.5;
    
    if (!sensorTrusted) return SUBSYSTEM_COLORS.unknown;
    if (cylHealth < 0.3) return SUBSYSTEM_COLORS.critical;
    if (cylHealth < 0.5) return SUBSYSTEM_COLORS.warning;
    if (cylHealth < 0.7) return SUBSYSTEM_COLORS.caution;
    return SUBSYSTEM_COLORS.healthy;
  };
  
  // SVG: 4 cylinders (2 left, 2 right), oil sump, cooling fins, injectors
  return (
    <svg viewBox="0 0 400 300" className="engine-schematic">
      {/* Cylinder 1 (top-left) */}
      <rect x={60} y={60} width={80} height={60} rx={8} 
            fill={getCylinderColor(1)} opacity={0.8} />
      <text x={100} y={95} textAnchor="middle" fill="white" fontSize="14">
        C1
      </text>
      
      {/* Cylinder 2 (bottom-left) */}
      <rect x={60} y={180} width={80} height={60} rx={8} 
            fill={getCylinderColor(2)} opacity={0.8} />
      
      {/* Cylinders 3, 4 (right side) */}
      {/* ... similar */}
      
      {/* Crankcase / oil system */}
      <ellipse cx={200} cy={150} rx={60} ry={80} 
               fill={SUBSYSTEM_COLORS[health.subsystems.lubrication > 0.5 ? 'healthy' : 'warning']} />
      
      {/* Labels and indicators */}
    </svg>
  );
}

// frontend/src/components/WatchdogBanner.tsx
function WatchdogBanner() {
  const { watchdog } = useTelemetryStore();
  
  const bannerStyle = {
    HEALTHY: { bg: '#22c55e', text: 'Data Pipeline: NOMINAL' },
    DATA_DEGRADED: { bg: '#eab308', text: 'Data Pipeline: DEGRADED — Some channels slow' },
    CRITICAL_DATA_LOSS: { bg: '#ef4444', text: 'DATA LOSS — Physics-only fallback active' },
  }[watchdog.overall_status];
  
  return (
    <div className={`watchdog-banner ${watchdog.overall_status.toLowerCase()}`}
         style={{ backgroundColor: bannerStyle.bg }}>
      <span>{bannerStyle.text}</span>
      {/* Per-channel staleness indicators */}
      <div className="channel-status">
        {Object.entries(watchdog.channels).map(([name, ch]) => (
          <span key={name} className={`channel ${ch.status.toLowerCase()}`}>
            {name}: {ch.staleness_s.toFixed(1)}s
          </span>
        ))}
      </div>
    </div>
  );
}
```

---

## 9. 24-Hour Team Execution Plan

### Team Division (6 members — SIH standard)

| Member | Role | Owns |
|---|---|---|
| **A** | Physics/Systems Lead | Physics model, UKF, calibration |
| **B** | ML Engineer | EDL model, OOD detector, training pipeline |
| **C** | Backend/Integration | FastAPI, WebSocket, processing loop |
| **D** | Frontend Lead | React dashboard, WebSocket client, all visualizations |
| **E** | Simulator/Fault Expert | Engine simulator, fault injector, FMEA rules |
| **F** | Documentation/Demo Lead | Slides, demo script, architecture diagrams, testing |

### Hour-by-Hour Plan

| Hours | A (Physics) | B (ML) | C (Backend) | D (Frontend) | E (Simulator) | F (Docs) |
|---|---|---|---|---|---|---|
| **0–2** | Verify physics model equations; test against simulator healthy output | Load pre-trained EDL model; verify ONNX inference works | Set up FastAPI skeleton; WebSocket endpoint working | Set up React/Vite; WebSocket client connects; display raw telemetry | Verify simulator outputs valid Rotax-912 ranges; test all fault injection modes | Start architecture diagram; take screenshots of initial setup |
| **2–4** | Calibrate physics model against simulator (tune constants until residuals < 5% during healthy) | Run OOD calibration on healthy data; verify Mahalanobis threshold | Wire simulator → validator → circular buffer → physics → residual pipeline | Build HealthScore gauge component; real-time updating | Create mission profiles (cruise, climb, high-alt, hot-weather) | Document architecture decisions; screenshot pipeline data flow |
| **4–6** | Implement UKF state estimator; test convergence on healthy data | Test EDL on synthetic fault data; verify uncertainty increases for OOD | Wire ML inference into pipeline; handle window buffering | Build EngineSchematic SVG component; color-code by subsystem health | Fine-tune fault severities to produce visible but realistic signatures | Write FMEA rules documentation with sensor signature table |
| **6–8** | Integrate UKF degradation parameter tracking; verify k1_cooling responds to cooling fault | Integrate OOD detector; test that novel conditions trigger low confidence | Wire sensor trust + FMEA attribution + decision arbiter into pipeline | Build AlertFeed with cause + confidence + recommended action display | Create combined-fault scenarios (misfire + cooling) for robustness testing | Screenshot working dashboard with live telemetry |
| **8–10** | Test physics model fallback mode (when ML abstains) | Test full ML pipeline end-to-end; verify uncertainty propagation | Implement Watchdog integration; test staleness detection by killing feed | Build WatchdogBanner; PhysicsOverlay (measured vs expected); DegradationChart | Create sensor-fault scenarios (stuck, drift, noise) | Record video of fault detection working |
| **10–12** | **CHECKPOINT: Full pipeline demo-able** | Fine-tune model if accuracy low | **CHECKPOINT: WebSocket streaming all data** | **CHECKPOINT: Full dashboard with live data** | **CHECKPOINT: All fault types injectable** | Draft demo script; prepare presentation skeleton |
| **12–14** | Test edge cases: extreme altitude, rapid throttle | Test model on combined faults | Implement replay endpoint (load Parquet, re-run pipeline) | Build MissionControls panel (fault injection buttons for demo) | Test sensor-fault vs engine-fault arbitration logic | Write "What we didn't build and why" slide |
| **14–16** | Optimize physics model performance (target < 5ms per tick) | Export final model to ONNX; verify inference time < 10ms | Performance optimization; ensure 10 Hz tick maintains | Polish UI: confidence tags, color coding, alert animations | Prepare demo script scenarios in order | Prepare technical Q&A answers |
| **16–18** | **Full integration test: run all 5 demo scenarios** | Verify uncertainty outputs are sensible for each scenario | Test WebSocket with multiple clients (dashboard + judge view) | Final UI polish; responsive layout; loading states | Practice fault injection timing for smooth demo | Finalize slides; record backup demo video |
| **18–20** | Fix any physics model bugs found in integration | Fix any ML issues | Fix any backend bugs | Fix any frontend bugs | Fix any simulator timing issues | Complete documentation; architecture diagram final |
| **20–22** | Write unit tests for physics model | Write unit tests for EDL + OOD | Write integration tests | Build offline fallback (recorded data if live demo fails) | Test backup demo video playback | **REHEARSE 5-minute pitch (×3)** |
| **22–24** | Final code review; remove debug code | Final model validation | Deploy check: `pnpm dev` works from clean clone | Final visual check | **FREEZE CODE** | Final pitch rehearsal; Q&A prep |

---

## 10. Day-0 Pre-Hackathon Checklist (Complete BEFORE Event)

### Code Repository
- [ ] Repo created with above folder structure
- [ ] `pyproject.toml` with all dependencies pinned
- [ ] `pnpm-workspace.yaml` configured
- [ ] README with setup instructions
- [ ] `.gitignore` (data/, models/, node_modules/, __pycache__/)

### Physics Model
- [ ] All equations implemented in `expectation_model.py`
- [ ] Constants calibrated against Rotax-912 published specs
- [ ] Unit tests passing: `pytest backend/physics/`
- [ ] Healthy simulation produces residuals < 5% for all parameters

### Simulator
- [ ] Produces valid Rotax-912 ranges for all sensors
- [ ] All 5 fault types injectable (cooling, lubrication, misfire, injector, sensor)
- [ ] Mission profiles: cruise, climb, high-altitude, hot-weather, throttle-transitions
- [ ] 10 Hz stable output for 60+ minutes without drift

### ML Model
- [ ] EDL model architecture implemented
- [ ] Training data generated (all scenarios from Section 6.1)
- [ ] Model trained and saved as ONNX
- [ ] OOD detector calibrated (Mahalanobis stats saved)
- [ ] Inference time < 10ms verified on CPU
- [ ] Uncertainty outputs sensible: low for known faults, high for OOD

### Backend
- [ ] FastAPI app runs with `uvicorn backend.main:app --reload`
- [ ] WebSocket endpoint accepts connections
- [ ] Processing loop runs at 10 Hz without drift
- [ ] All Pydantic schemas defined and tested

### Frontend
- [ ] `pnpm dev` starts without errors
- [ ] WebSocket connects to backend
- [ ] Basic telemetry display works
- [ ] All component stubs created

### Documentation
- [ ] Architecture diagram (this document's Section 2) as PNG/SVG
- [ ] FMEA fault signature table (Section 4.5 rules) as slide
- [ ] Tech stack justification table (Section 1)
- [ ] "What we didn't build" slide drafted
- [ ] 5-minute demo script written (Section 11)

### Team Prep
- [ ] All members can run the full stack locally
- [ ] Each member knows their hour-by-hour tasks
- [ ] Demo roles assigned (who talks, who injects faults, who handles Q&A)
- [ ] Backup plan: recorded demo video if live demo fails

---

## 11. 5-Minute Demo Script (Final)

**Minute 0–1: The Setup**
> "This is a digital twin of a Rotax-912-class aero piston engine — the same engine class used in MALE UAVs like India's RUSTOM-II. On the left, you see live engine telemetry. On the right, our physics model's prediction of what those values *should* be. When the lines diverge, that's a residual — and residuals are where all fault detection happens. Notice they're currently near zero — the twin is synchronized with the engine."

**Minute 1–2: Fault Detection with Causality**
> "Now I'm injecting a cooling system degradation. Watch the CHT residuals climb across all four cylinders — while oil pressure and RPM stay nominal. Our FMEA rules engine recognizes this *pattern*: CHT up, oil stable, RPM stable → cooling system, not lubrication, not mechanical. The alert names the subsystem, explains the causal evidence in plain language, and recommends an action. This isn't a black box — it's explainable by design."

**Minute 2–3: Uncertainty — The System Knows What It Doesn't Know**
> "Now I'm injecting a condition the ML model was never trained on — a combined fault scenario. Watch what happens: the model's uncertainty spikes, and instead of confidently guessing wrong, it explicitly flags 'Low confidence — novel condition' and falls back to physics-model-only monitoring. This is the difference between a system that claims accuracy and a system that claims *trustworthiness*. Our system tells you when to stop trusting it."

**Minute 3–4: Sensor Fault vs Engine Fault**
> "Here's the scenario that fools most monitoring systems: oil pressure reads zero. A threshold system would trigger a mission abort. But our sensor cross-validation notices that RPM, CHT, and vibration all agree perfectly with physics — if the oil system had actually failed, we'd see correlated changes in temperature and vibration too. One sensor diverging while everything else agrees? That's a sensor fault. The system correctly recommends 'verify wiring' instead of aborting the mission."

**Minute 4–5: Data Dropout — Graceful Degradation**
> "Finally, watch what happens when the data feed dies. [Kill the simulator.] The watchdog banner immediately flags 'STALE — 4 seconds since last update.' The system doesn't freeze, doesn't show old data as current — it explicitly enters physics-only fallback mode and says so. When data returns, it seamlessly resumes full ML-augmented monitoring. This is what 'mission reliability' actually means — the system behaves correctly when things go wrong."

**Closing (30 seconds):**
> "Our digital twin is physics-anchored, uncertainty-aware, and failure-tolerant. Every alert carries a cause, a confidence, and an honest acknowledgment of what the system doesn't know. This is the architecture for a system that earns trust in a defense-grade Ground Control Station — not just a dashboard that looks good when everything works."

---

## 12. Deliverables Checklist (Final Submission)

- [ ] **Functional prototype** — `git clone` + `pnpm install` + `pnpm dev` = working demo
- [ ] **Architecture design** — This document, formatted as PDF
- [ ] **Engine simulation model** — `backend/simulator/` + `backend/physics/`
- [ ] **AI/ML anomaly detection module** — `backend/ml/` with trained ONNX model
- [ ] **Visualization dashboard** — `frontend/` with all components
- [ ] **Demo video** — 5-minute recorded demo (backup for live demo)
- [ ] **Technical documentation**:
  - [ ] Architecture rationale (why each choice)
  - [ ] FMEA fault signature table
  - [ ] Model training methodology
  - [ ] Uncertainty quantification approach
  - [ ] What we deliberately didn't build and why
- [ ] **Deployment roadmap** — GCS integration → engine test rig → fleet-level monitoring
- [ ] **Presentation deck** — 10-12 slides, demo-heavy, minimal text
- [ ] **Code with comments** — Every major component documented
- [ ] **Test suite passing** — `pytest` + frontend build clean

---

This architecture is buildable in 24 hours by a 6-person team with the pre-hackathon preparation, demonstrates every capability in the problem statement, and positions the uncertainty-aware physics-anchored approach as the core innovation — which is exactly where the field is moving in 2025-2026 research【turn0search5】【turn1search12】【turn1search11】.