"""
SOTA training pipeline for EvidentialFaultClassifier.
Features:
- Mixed precision (AMP), gradient clipping, AdamW + CosineAnnealingWarmRestarts
- KL annealing, label smoothing, MixUp, class-weighted sampling/weights
- Early stopping on val ECE + accuracy, checkpoint best
- Conformal calibration, ONNX export, TorchScript
- Detailed logging + plots for Kaggle
- W&B integration: metrics, artifacts, model registry
"""
from __future__ import annotations
import argparse
import json
import time
import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.config import (
    N_CLASSES, FAULT_CLASSES, TRAINING, MODELS_ROOT, DATA_ROOT, TRAINING_DATA_ROOT, RESIDUAL_STATS_FILE,
)
from backend.ml.models.evidential_model import EvidentialFaultClassifier, evidential_loss, FocalEvidentialLoss
from backend.ml.training.datasets import load_all_windows, stratified_splits, make_loaders, compute_class_weights, mixup_data
from backend.ml.training.metrics import evaluate_model, expected_calibration_error, compute_metrics

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def parse_args():
    p = argparse.ArgumentParser(description="Train EvidentialFaultClassifier")
    p.add_argument("--data", type=str, default=str(TRAINING_DATA_ROOT), help="Path to windows.npz dir or file")
    p.add_argument("--epochs", type=int, default=TRAINING["epochs"])
    p.add_argument("--batch-size", type=int, default=TRAINING["batch_size"])
    p.add_argument("--lr", type=float, default=TRAINING["lr"])
    p.add_argument("--weight-decay", type=float, default=TRAINING["weight_decay"])
    p.add_argument("--annealing-epochs", type=int, default=TRAINING["annealing_epochs"])
    p.add_argument("--patience", type=int, default=TRAINING["early_stopping_patience"])
    p.add_argument("--seed", type=int, default=TRAINING["seed"])
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--mixup", type=float, default=TRAINING["mixup_alpha"])
    p.add_argument("--label-smoothing", type=float, default=TRAINING["label_smoothing"])
    p.add_argument("--focal-gamma", type=float, default=TRAINING["focal_gamma"])
    p.add_argument("--scheduler", type=str, default=TRAINING["scheduler"], choices=["cosine", "plateau", "none"])
    p.add_argument("--optimizer", type=str, default=TRAINING["optimizer"], choices=["adam", "adamw"])
    p.add_argument("--val-split", type=float, default=TRAINING["val_split"])
    p.add_argument("--test-split", type=float, default=TRAINING["test_split"])
    p.add_argument("--quick", action="store_true", help="Quick smoke test (few epochs, no val)")
    p.add_argument("--output-dir", type=str, default=str(MODELS_ROOT))
    p.add_argument("--export-onnx", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (0 for Kaggle / low RAM)")
    # W&B — COMPULSORY (training will fail if WANDB_API_KEY not set and mode=online)
    p.add_argument("--wandb", action="store_true", help=argparse.SUPPRESS)  # deprecated, kept for compat
    p.add_argument("--no-wandb", action="store_true", help=argparse.SUPPRESS)  # deprecated, not allowed now
    p.add_argument("--wandb-project", type=str, default="sih26054-digital-twin", help="W&B project name (compulsory)")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    p.add_argument("--wandb-name", type=str, default=None, help="W&B run name (default auto)")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="W&B tags, e.g. --wandb-tags sota kaggle")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline"], help="W&B mode: online (requires WANDB_API_KEY) is compulsory; offline only for debugging without push")
    p.add_argument("--wandb-watch", type=str, default="gradients", choices=["none", "gradients", "all"], help="W&B watch log level")
    p.add_argument("--wandb-log-model", type=str, default="artifact", choices=["artifact"], help="Upload model as W&B artifact (compulsory)")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, epoch, annealing_epochs, scaler=None, mixup_alpha=0.0, label_smoothing=0.0, focal_gamma=0.0, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # MixUp
        if mixup_alpha > 0 and np.random.rand() < 0.5:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(x.size(0), device=device)
            mixed_x = lam * x + (1 - lam) * x[idx]
            y_a, y_b = y, y[idx]
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None and device.type == "cuda")):
                out_a = model(mixed_x)
                loss_a = evidential_loss(out_a, y_a, epoch, annealing_epochs, label_smoothing=label_smoothing)
                loss_b = evidential_loss(out_a, y_b, epoch, annealing_epochs, label_smoothing=label_smoothing)
                loss = lam * loss_a + (1 - lam) * loss_b
        else:
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None and device.type == "cuda")):
                out = model(x)
                if focal_gamma > 0:
                    from backend.ml.models.evidential_model import FocalEvidentialLoss
                    crit = FocalEvidentialLoss(gamma=focal_gamma, annealing_epochs=annealing_epochs)
                    loss = crit(out, y, epoch)
                else:
                    loss = evidential_loss(out, y, epoch, annealing_epochs, label_smoothing=label_smoothing)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, epoch, annealing_epochs):
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs, all_labels, all_unc = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = evidential_loss(out, y, epoch, annealing_epochs)
        total_loss += loss.item()
        all_probs.append(out["probs"].cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_unc.append(out["epistemic_uncertainty"].cpu().numpy())
        n += 1
    avg_loss = total_loss / max(n, 1)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, N_CLASSES))
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,))
    unc = np.concatenate(all_unc, axis=0) if all_unc else np.zeros((0,))
    metrics = compute_metrics(probs, labels, unc) if len(labels) > 0 else {"accuracy": 0, "ece": 1}
    metrics["loss"] = avg_loss
    return metrics


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 3
        args.batch_size = 32
        args.patience = 100
        args.val_split = 0.2
        args.test_split = 0.1

    # Handle deprecated flags
    if getattr(args, "no_wandb", False):
        raise SystemExit("[W&B] --no-wandb not allowed — W&B is compulsory (remove the flag)")
    # --wandb is now implicit, ignored

    # W&B is COMPULSORY — fail fast if not configured
    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise SystemExit(
            "[W&B] WANDB_API_KEY not set but --wandb-mode=online (compulsory). "
            "Fix: on Kaggle, add Secrets → WANDB_API_KEY (wandb.ai → Settings → API Keys) and attach to notebook, "
            "or run Cell 1b to load it. For local debugging without push use --wandb-mode offline."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device} | CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[train] W&B COMPULSORY — project={args.wandb_project} mode={args.wandb_mode} (set WANDB_API_KEY to push)")

    # W&B init (before data loading so config is logged) — always
    wandb_run = None
    try:
        from backend.ml.training.wandb_utils import init_wandb, watch_model as wandb_watch
        wandb_config = {
            **vars(args),
            "n_classes": N_CLASSES,
            "fault_classes": FAULT_CLASSES,
            "training_defaults": TRAINING,
            "device": str(device),
        }
        tags = args.wandb_tags or ["sota", "edl", "compulsory"]
        run_name = args.wandb_name or ("sih-quick" if args.quick else f"sih-edl-{args.epochs}ep")
        wandb_run = init_wandb(
            enabled=True,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            mode=args.wandb_mode,
            config=wandb_config,
        )
        if wandb_run is None:
            raise SystemExit("[W&B] init failed — training requires W&B (compulsory). Check WANDB_API_KEY and internet.")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"[W&B] init failed (compulsory): {e}") from e

    # Load data
    data_path = Path(args.data)
    if data_path.is_file():
        data = np.load(data_path)
        X = data["X"]
        y = data["y"]
        if X.ndim == 3 and X.shape[1] == 30 and X.shape[2] == 14:
            X = X.transpose(0, 2, 1)
    elif (data_path / "windows.npz").exists():
        X, y, meta = load_all_windows(data_path)
    else:
        X, y, meta = load_all_windows(Path(args.data))

    print(f"[train] Loaded X={X.shape} y={y.shape} | classes={np.bincount(y, minlength=N_CLASSES)}")
    # Splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_splits(X, y, val_split=args.val_split, test_split=args.test_split, seed=args.seed)
    print(f"[train] Splits: train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    if len(y_val) < 10 and len(y_train) > 20:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=args.seed)
        print(f"[train] Adjusted val split: train={len(y_train)} val={len(y_val)}")

    use_weighted = TRAINING.get("use_class_weights", True)
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_weighted_sampler=False,
        augment_train=True, noise_std=TRAINING["aug_noise_std"], scale_jitter=TRAINING["aug_scale_jitter"],
    )
    print(f"[train] Loaders: train_batches={len(train_loader)} val_batches={len(val_loader)}")

    # Model
    model = EvidentialFaultClassifier(n_channels=14, n_timesteps=30, n_classes=N_CLASSES, dropout=0.2).to(device)
    print(f"[train] Model params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")

    # W&B watch
    if wandb_run is not None and args.wandb_watch != "none":
        try:
            from backend.ml.training.wandb_utils import watch_model
            watch_model(wandb_run, model, log=args.wandb_watch)
        except Exception:
            pass

    # Optimizer + scheduler
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        plateau = False
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
        plateau = True
    else:
        scheduler = None
        plateau = False

    # AMP scaler
    use_amp = (not args.no_amp) and (device.type == "cuda") and TRAINING.get("mixed_precision", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[train] Mixed precision ENABLED (AMP)")

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_ece = float("inf")
    patience_counter = 0
    history = []
    best_state = None

    start_time = time.time()
    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch, args.annealing_epochs,
                scaler=scaler, mixup_alpha=args.mixup, label_smoothing=args.label_smoothing,
                focal_gamma=args.focal_gamma, grad_clip=TRAINING.get("grad_clip", 1.0),
            )
            val_metrics = validate(model, val_loader, device, epoch, args.annealing_epochs)

            if scheduler is not None:
                if plateau:
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_metrics["loss"], val_acc=val_metrics["accuracy"], val_ece=val_metrics["ece"], val_f1=val_metrics["f1_macro"], lr=current_lr))

            print(f"Epoch {epoch+1:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
                  f"val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['f1_macro']:.3f} val_ece={val_metrics['ece']:.3f} | lr={current_lr:.2e} | {(time.time()-epoch_start):.1f}s")

            # W&B per-epoch logging
            if wandb_run is not None:
                try:
                    from backend.ml.training.wandb_utils import log_metrics
                    log_metrics(
                        step=epoch,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_metrics["loss"],
                        val_accuracy=val_metrics["accuracy"],
                        val_f1_macro=val_metrics["f1_macro"],
                        val_f1_weighted=val_metrics.get("f1_weighted", 0),
                        val_ece=val_metrics["ece"],
                        val_uncertainty_auroc=val_metrics.get("uncertainty_auroc", 0),
                        lr=current_lr,
                        best_val_loss=best_val_loss,
                    )
                except Exception as e:
                    print(f"[wandb] log failed: {e}")

            # Checkpoint logic — composite: accuracy - ECE (want high acc, low ECE)
            score = val_metrics["accuracy"] - val_metrics["ece"]
            best_score = best_val_acc - best_ece
            is_better = score > best_score + 1e-4
            if is_better:
                best_val_loss = val_metrics["loss"]
                best_val_acc = val_metrics["accuracy"]
                best_ece = val_metrics["ece"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                torch.save({"epoch": epoch, "model_state": best_state, "val_metrics": val_metrics, "args": vars(args)}, output_dir / "evidential_model_best.pt")
                print(f"  -> New best! Saved to evidential_model_best.pt (val_loss={best_val_loss:.4f}, acc={best_val_acc:.3f}, ece={best_ece:.3f})")
                if wandb_run is not None:
                    try:
                        import wandb
                        wandb.run.summary["best_val_loss"] = best_val_loss
                        wandb.run.summary["best_val_accuracy"] = best_val_acc
                        wandb.run.summary["best_val_ece"] = best_ece
                    except Exception:
                        pass
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"[train] Early stopping at epoch {epoch+1} (patience {args.patience})")
                break

            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_metrics": val_metrics}, output_dir / "evidential_model_last.pt")

    except KeyboardInterrupt:
        print("[train] Interrupted — saving current state")

    elapsed = time.time() - start_time
    print(f"[train] Training done in {elapsed/60:.1f} min | Best val_loss={best_val_loss:.4f} acc={best_val_acc:.3f} ece={best_ece:.3f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "val_metrics": dict(loss=best_val_loss, acc=best_val_acc, ece=best_ece)}, output_dir / "evidential_model.pt")
    torch.save(model.state_dict(), output_dir / "evidential_model_state.pt")
    print(f"[train] Saved final model to {output_dir / 'evidential_model.pt'}")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Test evaluation
    test_metrics = None
    val_detailed = None
    if test_loader is not None and len(y_test) > 0:
        test_metrics = evaluate_model(model, test_loader, device)
        print(f"[train] TEST: acc={test_metrics['accuracy']:.3f} f1_macro={test_metrics['f1_macro']:.3f} ece={test_metrics['ece']:.3f}")
        print(f"[train] Confusion matrix:\n{np.array(test_metrics['confusion_matrix'])}")
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        val_detailed = evaluate_model(model, val_loader, device)
        with open(output_dir / "val_metrics.json", "w") as f:
            json.dump(val_detailed, f, indent=2)

        # W&B final test/val logging
        if wandb_run is not None:
            try:
                from backend.ml.training.wandb_utils import log_metrics
                log_metrics(
                    test_accuracy=test_metrics["accuracy"],
                    test_f1_macro=test_metrics["f1_macro"],
                    test_f1_weighted=test_metrics["f1_weighted"],
                    test_ece=test_metrics["ece"],
                    test_uncertainty_auroc=test_metrics.get("uncertainty_auroc", 0),
                    val_detail_accuracy=val_detailed["accuracy"] if val_detailed else 0,
                )
                # Log confusion matrix as W&B table
                try:
                    import wandb
                    cm = np.array(test_metrics["confusion_matrix"])
                    table = wandb.Table(
                        data=[[FAULT_CLASSES[i]] + [int(x) for x in row] for i, row in enumerate(cm.tolist())],
                        columns=["true_class"] + [f"pred_{c}" for c in FAULT_CLASSES],
                    )
                    wandb.log({"confusion_matrix": table})
                except Exception as e:
                    print(f"[wandb] confusion matrix log failed: {e}")
            except Exception as e:
                print(f"[wandb] final metrics log failed: {e}")

    # Export ONNX
    if args.export_onnx:
        try:
            model.eval()
            dummy = torch.randn(1, 14, 30, device=device)

            class _OnnxWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x)["evidence"]

            wrapper = _OnnxWrapper(model)
            wrapper.eval()
            onnx_path = output_dir / "evidential_model.onnx"
            torch.onnx.export(
                wrapper, dummy, str(onnx_path),
                input_names=["input"], output_names=["evidence"],
                dynamic_axes={"input": {0: "batch"}, "evidence": {0: "batch"}},
                opset_version=17,
            )
            print(f"[train] Exported ONNX to {onnx_path} ({onnx_path.stat().st_size/1024:.1f} KB)")
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            dummy_cpu = torch.randn(1, 14, 30)
            ort_out = sess.run(None, {"input": dummy_cpu.numpy()})
            model_cpu = model.cpu()
            torch_out = model_cpu(dummy_cpu)["evidence"].detach().numpy()
            model.to(device)
            diff = np.abs(ort_out[0] - torch_out).max()
            print(f"[train] ONNX verification max diff: {diff:.2e}")
        except Exception as e:
            print(f"[train] ONNX export failed: {e}")

    # Conformal calibration
    try:
        from backend.ml.training.calibrate_conformal import calibrate_conformal
        calibrate_conformal(model, val_loader, device, output_dir)
    except Exception as e:
        print(f"[train] Conformal calibration skipped: {e}")

    # W&B artifact upload (model + stats) — compulsory
    if True:  # always, wandb_run is guaranteed compulsory
        try:
            from backend.ml.training.wandb_utils import log_artifact
            # Ensure OOD stats exist — if not, try to generate quickly from healthy subset
            ood_path = output_dir / "ood_stats.npz"
            if not ood_path.exists():
                try:
                    from backend.ml.models.ood_detector import OODDetector
                    det = OODDetector(stats_path=ood_path)
                    healthy_mask = (y == 0)  # healthy idx
                    # Use training healthy windows
                    X_healthy = X[healthy_mask]
                    if len(X_healthy) > 20:
                        det.fit(X_healthy[: min(2000, len(X_healthy))])
                        print("[train] Auto-generated ood_stats.npz for artifact")
                except Exception as e:
                    print(f"[train] auto OOD for artifact failed: {e}")
            log_artifact(wandb_run, output_dir, name=f"sih-edl-{wandb_run.id}", type="model", aliases=["latest", "best"] )
            print(f"[wandb] View run at: {wandb_run.url}")
        except Exception as e:
            print(f"[wandb] artifact upload failed: {e}")

    # W&B finish — compulsory
    try:
        from backend.ml.training.wandb_utils import finish
        finish(wandb_run)
    except Exception:
        pass
    if wandb_run is None:
        raise SystemExit("[W&B] run missing — compulsory W&B failed")

    print("[train] Done. Artifacts in", output_dir)


if __name__ == "__main__":
    main()
