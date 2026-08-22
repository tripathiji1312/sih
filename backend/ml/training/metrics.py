"""
Metrics: accuracy, per-class F1, ECE, uncertainty calibration, OOD AUROC.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE: bin predicted confidences, compare accuracy vs confidence.
    probs: (N, K) softmax/dirichlet mean
    labels: (N,)
    """
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += np.abs(bin_conf - bin_acc) * mask.mean()
    return float(ece)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, uncertainties: np.ndarray = None) -> dict:
    preds = probs.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    ece = expected_calibration_error(probs, labels)
    # Per-class F1
    per_class = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)

    out = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "ece": float(ece),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": per_class[2].tolist(),
    }
    if uncertainties is not None:
        # Uncertainty vs correctness correlation: should be higher for wrong preds
        correct = (preds == labels)
        if len(np.unique(correct)) > 1:
            try:
                # AUROC of uncertainty as predictor of error
                auroc = roc_auc_score((~correct).astype(int), uncertainties)
                out["uncertainty_auroc"] = float(auroc)
            except Exception:
                out["uncertainty_auroc"] = 0.5
        # Mean uncertainty for correct vs wrong
        out["uncertainty_correct_mean"] = float(uncertainties[correct].mean()) if correct.any() else 0.0
        out["uncertainty_wrong_mean"] = float(uncertainties[~correct].mean()) if (~correct).any() else 0.0

    return out


def evaluate_model(model, loader, device="cpu") -> dict:
    model.eval()
    all_probs, all_labels, all_uncert = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = out["probs"].cpu().numpy()
            unc = out["epistemic_uncertainty"].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
            all_uncert.append(unc)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    unc = np.concatenate(all_uncert, axis=0)
    return compute_metrics(probs, labels, unc)


def ood_auroc(healthy_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC where higher score = more OOD."""
    y_true = np.concatenate([np.zeros(len(healthy_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([healthy_scores, ood_scores])
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5
