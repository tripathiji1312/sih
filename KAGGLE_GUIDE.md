# Kaggle Training Guide — SIH26054 Digital Twin (SOTA EDL Pipeline + W&B)

This repo is **GPU-free locally**: you clone it on Kaggle and train there. No external datasets needed — all training data is physics-simulated via `EngineSimulator`. **W&B pushes models automatically** so you don't need to manually download zips (though you still can).

## 0. What Kaggle will produce

After training you get in `data/models/` (also auto-pushed to W&B as artifact `sih-edl-<run_id>:latest`):
```
evidential_model.pt          # PyTorch checkpoint (for backend inference)
evidential_model.onnx        # ONNX for fast CPU inference (auto-loaded by FastAPI)
evidential_model_best.pt     # best epoch checkpoint
ood_stats.npz                # Mahalanobis mean/cov_inv/threshold
residual_stats.npz           # healthy residual mean/std (z-score)
conformal_stats.json         # conformal prediction threshold
training_history.json        # loss/acc/ece per epoch
val_metrics.json / test_metrics.json
```
Locally: `wandb artifact get <entity>/sih26054-digital-twin/sih-edl-<id>:latest --root data/models/` or download `sih_models.zip` from Kaggle Output.

---

## 1. Kaggle Notebook Setup (do this once per session)

1. Create new Kaggle Notebook at kaggle.com/code → **New Notebook**
2. Settings (right panel):
   - **Accelerator: GPU T4 x2** (or P100). T4 is enough; training is ~10 min full, ~2 min quick.
   - **Internet: ON** (required for `pip install` + `git clone` + `wandb`)
   - **Environment: Latest** (Python 3.11+)
   - **No datasets need to be added** via "Add Data" — leave empty. If you added one by mistake, remove it.
3. **Do NOT attach any Kaggle Dataset** for this project. Data is generated on-the-fly.

> If you want to skip data generation (to save 2-3 min), you *can* attach a self-uploaded dataset containing `windows.npz` as `/kaggle/input/sih-windows/windows.npz` and point `--data` to it — see Cell 3 alternative below.

### 1b. W&B Setup (2 minutes, highly recommended)

1. Create W&B account at **wandb.ai** → **User Settings → API Keys → Copy**
2. In Kaggle notebook: **Add-ons → Secrets → Add Secret**
   - Name: `WANDB_API_KEY`
   - Value: paste your key
   - **Check "Attach to this notebook"** (important — otherwise secret not injected)
3. Optionally also add `GITHUB_TOKEN` if you want Kaggle to push models back to GitHub directly.
4. In notebook Cell 1b you will load it via `kaggle_secrets` — see below. If you skip this, training still works with `--wandb-mode offline` (local logs) or `--no-wandb` (no logging) — but online push is the perfect setup.

---

## 2. Cells to run (copy-paste in order)

You can also just open and run the ready-made notebook in this repo: `kaggle/kaggle_training.ipynb` → **File → Import Notebook** on Kaggle, or copy cells below. The notebook already has W&B integrated.

### Cell 1 — Clone repo + checkout correct branch

```python
import os, sys, pathlib
# Replace with your actual repo URL
REPO_URL = "https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git"  # e.g. https://github.com/tripathiji1312/sih.git
BRANCH = "master"  # or main

# Kaggle working dir is /kaggle/working
!rm -rf /kaggle/working/sih
!git clone {REPO_URL} /kaggle/working/sih
%cd /kaggle/working/sih
!git checkout {BRANCH}
!ls -la
!cat archi.md | head -20
```

If your repo is private: add a Kaggle Secret `GITHUB_TOKEN` and clone via `https://{token}@github.com/...`.

### Cell 1b — W&B setup (loads WANDB_API_KEY from Kaggle Secrets)

```python
import os
!pip install -q wandb
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    print("WANDB_API_KEY loaded:", "***" + os.environ["WANDB_API_KEY"][-4:] if os.environ["WANDB_API_KEY"] else "EMPTY")
except Exception as e:
    print("Kaggle Secrets not available:", e)
    print("If WANDB_API_KEY set manually in env, it will be used. Else wandb runs offline/disabled.")

os.environ["WANDB_PROJECT"] = "sih26054-digital-twin"
print("W&B project:", os.environ.get("WANDB_PROJECT"))
print("WANDB_API_KEY set:", bool(os.environ.get("WANDB_API_KEY")))
```

If you see `EMPTY` or error: go back to **Add-ons → Secrets** and ensure `WANDB_API_KEY` is added **and** the toggle for this notebook is ON, then re-run.

### Cell 2 — Install dependencies (no full torch reinstall if already present)

```python
import sys
%cd /kaggle/working/sih

# Kaggle already has torch, numpy, sklearn. Only install missing.
!pip install -q filterpy onnx onnxruntime pyarrow tqdm wandb
!pip install -q -e .  # uses pyproject.toml (now includes wandb)

# Verify imports & GPU
import torch, wandb
print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("wandb", wandb.__version__, "WANDB_API_KEY set:", bool(os.environ.get("WANDB_API_KEY")))

# Check simulator works
from backend.simulator.engine_simulator import EngineSimulator
sim = EngineSimulator(seed=0)
frame = sim.step()
print("Simulator OK — rpm:", frame.rpm, "cht:", frame.cht_c)
```

**Expected output**: `torch 2.x cuda_available: True`, `wandb 0.18+`, `Simulator OK`.

### Cell 3 — Generate training data (physics-simulated)

This is the **only** data step. No external CSVs.

```python
%cd /kaggle/working/sih

# Quick smoke test (1-2 min, ~1k windows) — use for first verification
!python scripts/generate_all_data.py --quick

# ---- OR for full SOTA run (8-12 min, ~25k-40k windows): ----
# !python scripts/generate_all_data.py --full
# ---- OR fast default (4-6 min, ~12k windows, 10 min missions): ----
# !python scripts/generate_all_data.py

# Inspect output
!ls -lh data/training/
!ls -lh data/models/residual_stats.npz
!cat data/training/generation_stats.json
import numpy as np, json
d = np.load("data/training/windows.npz")
print("X", d["X"].shape, "y", d["y"].shape, "bincount", np.bincount(d["y"]))
```

**What it does** (`backend/ml/training/generate_training_data.py`):
- Runs `EngineSimulator` for each scenario in `backend/config.py:SCENARIOS` (or `SCENARIOS_FULL` if `--full`)
- Injects faults at random onset (cooling, lubrication, misfire x4, injector, sensor drift/stuck/noise/spike)
- Computes raw residuals `measured - expected` via `PhysicsExpectationModel` (nominal params)
- Computes `mean/std` from **healthy only** → saves `data/models/residual_stats.npz` → z-score normalizes all
- Sliding window `WINDOW_SIZE=30, STRIDE=5` → `X=(N,14,30), y=(N,)` → `data/training/windows.npz`
- Also saves `data/training/ood_windows.npz` for OOD eval

**If you attached a pre-generated `windows.npz` as Kaggle Dataset**, skip generation and do:
```python
!mkdir -p data/training
!cp /kaggle/input/sih-windows/windows.npz data/training/windows.npz
!cp /kaggle/input/sih-windows/residual_stats.npz data/models/residual_stats.npz
```

### Cell 4 — Train Evidential Model (SOTA) with W&B push

```python
%cd /kaggle/working/sih

# Quick verification (3 epochs, ~30s) WITH W&B online push:
!python -m backend.ml.training.train_evidential --quick --num-workers 0 --wandb --wandb-project sih26054-digital-twin --wandb-tags quick sota kaggle

# ---- If WANDB_API_KEY not set, use offline (logs locally, sync later via `wandb sync`): ----
# !python -m backend.ml.training.train_evidential --quick --num-workers 0 --wandb --wandb-mode offline

# ---- Full SOTA (60 epochs, early stopping, AMP on GPU) with W&B: ----
# !python -m backend.ml.training.train_evidential --epochs 60 --batch-size 64 --num-workers 0 --wandb --wandb-project sih26054-digital-twin --wandb-tags full sota

# ---- Without W&B (no cloud) ----
# !python -m backend.ml.training.train_evidential --quick --num-workers 0 --no-wandb

# With custom data path if using attached dataset:
# !python -m backend.ml.training.train_evidential --data /kaggle/input/sih-windows/windows.npz --epochs 60 --wandb
```

**SOTA features** (see `backend/ml/training/train_evidential.py`):
- `EvidentialFaultClassifier` 1D-CNN + SEBlock + Dirichlet head (`backend/ml/models/evidential_model.py`)
- Loss: Sensoy 2018 evidential loss with **KL annealing** (`annealing_epochs=20`), optional **focal + label smoothing + MixUp**
- **Mixed precision (AMP)** auto-enabled on GPU → 2-3x faster, set `--no-amp` to disable
- Optim: **AdamW + CosineAnnealingWarmRestarts** (or ReduceLROnPlateau), grad clip 1.0, weight_decay 1e-4
- Aug: Gaussian noise `0.02` + scale jitter `0.05` on training windows
- Early stopping on `val_loss + 0.5*ECE - 0.2*acc`, saves `evidential_model_best.pt` + `evidential_model.pt`
- Exports **ONNX + TorchScript** at end, **conformal calibration** saves `conformal_stats.json`
- Logs `training_history.json`, `val_metrics.json`, `test_metrics.json` (accuracy, F1 macro/weighted, ECE, confusion matrix, uncertainty AUROC)
- **W&B**: logs `train_loss/val_loss/val_accuracy/val_ece/val_f1/lr` per epoch, `best_val_*` in run summary, confusion matrix heatmap, and uploads artifact `sih-edl-<run_id>:latest` containing all models + stats. View at `wandb.ai/<entity>/<project>/runs/<id>`.

**Expected full run**: 60 epochs → best val `acc ~0.88-0.96`, `ECE ~0.03-0.08` (depends on seed/data size). Early stopping often at epoch 30-45. W&B run URL printed at end.

**W&B flags:**
- `--wandb` enable, `--no-wandb` disable, `--wandb-mode online|offline|disabled`, `--wandb-project`, `--wandb-entity`, `--wandb-name`, `--wandb-tags`, `--wandb-watch gradients|all|none`, `--wandb-log-model artifact|none`

### Cell 5 — Calibrate OOD Detector (Mahalanobis)

> This is auto-run at end of `train_evidential`, but you can re-run standalone:

```python
%cd /kaggle/working/sih
!python -m backend.ml.training.calibrate_ood --threshold 99.0

# Check threshold
import numpy as np
stats = np.load("data/models/ood_stats.npz")
print("threshold", float(stats["threshold"]), "mean shape", stats["mean"].shape)
```

Fits Mahalanobis on **healthy windows only**, threshold at 99th percentile. Later `OODDetector.check()` is used in `backend/main.py` processing loop.

### Cell 6 — Validate & inspect

```python
%cd /kaggle/working/sih
!python scripts/validate_models.py

# Quick manual inference test
import numpy as np
from backend.ml.inference import EvidentialModelInference
infer = EvidentialModelInference()
d = np.load("data/training/windows.npz")
X, y = d["X"], d["y"]
for i in [0, len(y)//2, -1]:
    out = infer.infer(X[i])
    print(f"true={y[i]} pred={out['predicted_label']} epistemic={out['epistemic_uncertainty']:.3f} conf={out['confidence']:.3f}")

# Plot training history if matplotlib available
import json, pathlib
try:
    import matplotlib.pyplot as plt
    hist = json.loads(pathlib.Path("data/models/training_history.json").read_text())
    plt.plot([h["train_loss"] for h in hist], label="train_loss")
    plt.plot([h["val_loss"] for h in hist], label="val_loss")
    plt.plot([h["val_acc"] for h in hist], label="val_acc")
    plt.legend(); plt.show()
except Exception as e:
    print("Plot skipped:", e)
```

Expected: `validate` prints per-class predictions, OOD healthy mean < fault mean, AUROC >0.85.

### Cell 7 — W&B artifacts verification + Save zip

```python
%cd /kaggle/working/sih
!ls -lh data/models/
!cat data/models/val_metrics.json
!cat data/models/test_metrics.json

# W&B local files
!ls -lh wandb/ 2>&1 | head -20
!find wandb -name "*.wandb" 2>&1 | head -5
print("If --wandb-mode offline, sync later via: wandb sync wandb/latest-run/")

!zip -r /kaggle/working/sih_models.zip data/models/
print("Zip at /kaggle/working/sih_models.zip — Download via Kaggle Output panel on right")
!ls -lh /kaggle/working/sih_models.zip
```

On Kaggle, the **Output** tab (right sidebar) will show `sih_models.zip` → **Download**. Unzip locally into your cloned repo's `data/models/` then `git add data/models/*.onnx data/models/*.npz` (the `.pt` is gitignored — add if you want, or keep only ONNX).

**W&B download (preferred):** Go to `wandb.ai → your project → Runs → <run> → Artifacts → sih-edl-<id>:latest → Download`, or locally:
```bash
pip install wandb && wandb login
wandb artifact get <entity>/sih26054-digital-twin/sih-edl-<run_id>:latest --root data/models/
ls -lh data/models/
```

Optional: push back to GitHub from Kaggle (if you set `GITHUB_TOKEN` secret):
```python
!git config --global user.email "kaggle@kaggle.com"
!git config --global user.name "kaggle"
!git add data/models/evidential_model.onnx data/models/ood_stats.npz data/models/residual_stats.npz data/models/training_history.json
!git commit -m "kaggle: add trained SOTA models"
!git push https://$GITHUB_TOKEN@github.com/<USER>/<REPO>.git HEAD:master
```

---

## 3. Alternative: Use Kaggle Datasets instead of `git clone`

If you prefer not to clone, you can upload this repo as a Kaggle Dataset:

1. On Kaggle → **Datasets → New Dataset** → upload repo zip → name `sih-repo`
2. In notebook → **Add Data → Your Datasets → sih-repo**
3. Then cells use ` /kaggle/input/sih-repo/` instead of `/kaggle/working/sih`:
   ```python
   !cp -r /kaggle/input/sih-repo/* /kaggle/working/sih && cd /kaggle/working/sih && pip install ...
   ```
   Cloning is simpler and keeps you on latest `master`.

---

## 4. Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'torch'` | Kaggle image already has it — if missing, `!pip install torch --index-url https://download.pytorch.org/whl/cu121` then restart kernel |
| `CUDA out of memory` | Reduce `--batch-size 32` or `--epochs 40` |
| `No training data found` | Re-run Cell 3; check `data/training/windows.npz` exists; path is `TRAINING_DATA_ROOT` from `backend/config.py` |
| `FileNotFoundError: ood_stats.npz` | Run Cell 5 |
| `ONNX export failed` | Non-fatal — `pip install onnx` then re-run train; inference falls back to `.pt` |
| `Simulator OK` fails | `pip install filterpy`; check `backend/simulator/__init__.py` |
| `wandb: API key not found` | Add Secret `WANDB_API_KEY` in Kaggle Add-ons → Secrets and attach to notebook, then re-run Cell 1b; or use `--wandb-mode offline` |
| `wandb: 403 Forbidden` | Wrong entity: set `--wandb-entity <your-username>` or leave default; check wandb.ai project exists |
| `wandb: disabled` | You passed `--no-wandb` or `--wandb-mode disabled`; remove to enable |
| Internet disabled | Enable **Internet ON** in notebook settings (required for pip + git + wandb) |
| Training too slow (>30 min) | Use `--quick` for smoke test, or `--epochs 30` + default data (not `--full`) |

---

## 5. Local verification after downloading (W&B or zip)

```bash
# Option A: from W&B
pip install wandb && wandb login
wandb artifact get <entity>/sih26054-digital-twin/sih-edl-<run_id>:latest --root data/models/
# Option B: from zip
unzip ~/Downloads/sih_models.zip -d .

ls -lh data/models/
python scripts/validate_models.py
# Should print OK for all models and AUROC >0.85

# Run backend with trained models
uvicorn backend.main:app --reload  # loads ONNX automatically
# In another terminal
python -m backend.physics.calibration  # should still pass
```

---

## 6. File map (training pipeline only)

```
backend/config.py                         # SCENARIOS, hyperparams, paths
backend/simulator/engine_simulator.py     # physics simulator (data source)
backend/physics/expectation_model.py      # nominal physics for residuals
backend/physics/residual_generator.py     # raw -> normalized + stats save
backend/ml/models/evidential_model.py     # EDL 1D-CNN + loss
backend/ml/models/ood_detector.py         # Mahalanobis
backend/ml/models/sensor_trust.py         # not trained, rule-based
backend/ml/training/generate_training_data.py  # -> windows.npz
backend/ml/training/datasets.py           # splits, loaders, MixUp
backend/ml/training/metrics.py            # acc, F1, ECE, AUROC
backend/ml/training/train_evidential.py   # main training loop (SOTA) + W&B
backend/ml/training/wandb_utils.py        # W&B helpers (init, log, artifact)
backend/ml/training/calibrate_ood.py      # fits OOD detector
backend/ml/training/calibrate_conformal.py# conformal thresholds
scripts/generate_all_data.py              # wrapper for Kaggle cell
scripts/train_models.py                   # train + calibrate wrapper (supports --wandb passthrough)
scripts/validate_models.py                # sanity check
kaggle/kaggle_training.ipynb              # ready-to-import Kaggle notebook (W&B integrated)
```

No other files are needed for training. Frontend/backend WebSocket code is ignored on Kaggle.

## 7. W&B Artifact Details

**What is pushed:** At end of `train_evidential --wandb`, `backend/ml/training/wandb_utils.py:log_artifact()` uploads a single artifact `sih-edl-<run_id>` containing:
- `evidential_model.pt` / `evidential_model_best.pt` / `evidential_model.onnx` / `evidential_model_scripted.pt`
- `ood_stats.npz` + `residual_stats.npz` + `conformal_stats.json`
- `training_history.json` + `val_metrics.json` + `test_metrics.json`

**Aliases:** `latest` and `best` — use `:latest` to pull last run.

**Dashboard:** `wandb.ai/<entity>/sih26054-digital-twin` shows per-epoch `val_accuracy/val_ece/val_f1/train_loss`, best summary, and `test_confusion_heatmap`.

**Offline mode:** If `WANDB_API_KEY` missing, `--wandb-mode offline` writes to `wandb/latest-run/` locally. Later `wandb sync wandb/latest-run/` pushes when key is available. No code change needed.
