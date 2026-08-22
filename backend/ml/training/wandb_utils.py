"""
W&B helper — safe init/log/artifact handling.
Works offline/disabled if WANDB_API_KEY missing or wandb not installed.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

_HAS_WANDB = None
_wandb = None

def _try_import():
    global _HAS_WANDB, _wandb
    if _HAS_WANDB is not None:
        return _HAS_WANDB
    try:
        import wandb as _w  # type: ignore
        _wandb = _w
        _HAS_WANDB = True
    except ImportError:
        _HAS_WANDB = False
    return _HAS_WANDB

def is_wandb_available() -> bool:
    return _try_import()

def init_wandb(
    enabled: bool = False,
    project: str = "sih26054-digital-twin",
    entity: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[list] = None,
    mode: str = "online",
    config: Optional[dict] = None,
    dir: Optional[str] = None,
):
    """
    Returns wandb run or None. Handles offline/disabled fallback.
    Kaggle: set secret WANDB_API_KEY → wandb login via env var auto.
    If WANDB_API_KEY not set and mode=online, falls back to disabled (no crash).
    """
    if not enabled:
        return None
    if not is_wandb_available():
        print("[wandb] not installed — run `pip install wandb` or disable with --no-wandb")
        return None

    # Auto-detect missing key -> offline
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key and mode == "online":
        print("[wandb] WANDB_API_KEY not set — switching to mode=offline (local logging only, no cloud push)")
        mode = "offline"
        # Still init offline to get local logs; user can later `wandb sync`
    if mode == "disabled":
        print("[wandb] mode=disabled — no logging")
        return None

    # Kaggle: ensure dir is writable
    if dir is None:
        dir = os.environ.get("WANDB_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else None)

    try:
        run = _wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            mode=mode,
            config=config,
            dir=dir,
            save_code=True,
        )
        print(f"[wandb] init ok — project={project} run={run.name} id={run.id} mode={mode} url={run.url}")
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}) — continuing without wandb")
        return None

def log_metrics(step: Optional[int] = None, **metrics):
    if not is_wandb_available():
        return
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log(metrics, step=step)
    except Exception as e:
        print(f"[wandb] log failed: {e}")

def log_artifact(run, output_dir: Path, name: str = "sih-digital-twin-model", type: str = "model", aliases: list = None):
    """
    Uploads all key artifacts as a single W&B Artifact.
    Called at end of training.
    """
    if run is None or not is_wandb_available():
        return None
    try:
        import wandb
        aliases = aliases or ["latest"]
        artifact = wandb.Artifact(name=name, type=type, description="EDL classifier + OOD + residual stats + metrics")
        files_added = 0
        for pattern in [
            "evidential_model.pt",
            "evidential_model_best.pt",
            "evidential_model.onnx",
            "evidential_model_scripted.pt",
            "ood_stats.npz",
            "residual_stats.npz",
            "conformal_stats.json",
            "training_history.json",
            "val_metrics.json",
            "test_metrics.json",
            "generation_stats.json",
        ]:
            # Check both output_dir and parent data/training for generation_stats
            candidates = [output_dir / pattern, output_dir.parent / "training" / pattern, Path("data/models") / pattern, Path("data/training") / pattern]
            for p in candidates:
                if p.exists() and p.is_file():
                    artifact.add_file(str(p))
                    print(f"[wandb] artifact add: {p}")
                    files_added += 1
                    break
        # Also add any json in output_dir
        for f in output_dir.glob("*.json"):
            if f.name not in [a.name for a in []]:  # dedup already added
                try:
                    artifact.add_file(str(f))
                    files_added += 1
                except Exception:
                    pass

        if files_added == 0:
            print("[wandb] no artifact files found to upload")
            return None

        run.log_artifact(artifact, aliases=aliases)
        print(f"[wandb] artifact logged: {name} ({files_added} files) aliases={aliases}")
        # Also use wandb.save for direct file logging (legacy)
        for f in ["training_history.json", "val_metrics.json", "test_metrics.json"]:
            p = output_dir / f
            if p.exists():
                try:
                    wandb.save(str(p), base_path=str(output_dir))
                except Exception:
                    pass
        return artifact
    except Exception as e:
        print(f"[wandb] artifact log failed: {e}")
        return None

def finish(run, exit_code: int = 0):
    if run is None or not is_wandb_available():
        return
    try:
        import wandb
        wandb.finish(exit_code=exit_code)
        print("[wandb] finished")
    except Exception as e:
        print(f"[wandb] finish failed: {e}")

def watch_model(run, model, log: str = "gradients", log_freq: int = 100):
    if run is None or not is_wandb_available():
        return
    try:
        import wandb
        wandb.watch(model, log=log, log_freq=log_freq)
        print(f"[wandb] watch model log={log}")
    except Exception as e:
        print(f"[wandb] watch failed: {e}")
