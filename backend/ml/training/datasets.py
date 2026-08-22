"""
Dataset utilities — loading, splitting, augmentation, DataLoaders.
SOTA: stratified splits, weighted sampling, MixUp, Gaussian noise aug.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Optional

from backend.config import N_CLASSES, TRAINING, FAULT_CLASSES


class ResidualWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, noise_std: float = 0.02, scale_jitter: float = 0.05):
        """
        X: (N, 14, 30) float32
        y: (N,) int64
        """
        self.X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y
        self.augment = augment
        self.noise_std = noise_std
        self.scale_jitter = scale_jitter

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            # Gaussian noise
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            # Scale jitter (per-channel)
            if self.scale_jitter > 0:
                scale = 1.0 + torch.randn(x.shape[0], 1) * self.scale_jitter
                x = x * scale
        return x, y


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    X = data["X"]  # (N,14,30)
    y = data["y"]
    # Ensure channel-first
    if X.ndim == 3 and X.shape[1] == 30 and X.shape[2] == 14:
        X = X.transpose(0, 2, 1)
    return X, y


def load_all_windows(data_root: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads data/training/windows.npz if exists, else tries per-scenario files.
    Returns X, y, meta.
    """
    window_file = data_root / "windows.npz"
    if window_file.exists():
        data = np.load(window_file, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        meta = dict(data.get("meta", {})) if "meta" in data else {}
        if X.ndim == 3 and X.shape[1] == 30 and X.shape[2] == 14:
            X = X.transpose(0, 2, 1)
        return X, y, meta

    # Fallback: collect per-class .npz
    Xs, ys = [], []
    for f in data_root.glob("*.npz"):
        d = np.load(f)
        if "X" in d and "y" in d:
            Xi = d["X"]
            yi = d["y"]
            if Xi.ndim == 3 and Xi.shape[1] == 30 and Xi.shape[2] == 14:
                Xi = Xi.transpose(0, 2, 1)
            Xs.append(Xi)
            ys.append(yi)
    if not Xs:
        raise FileNotFoundError(f"No training data found in {data_root}. Run generate_training_data first.")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, {}


def stratified_splits(X: np.ndarray, y: np.ndarray, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42):
    """Returns (X_train,y_train),(X_val,y_val),(X_test,y_test) with stratification."""
    # First split off test
    if test_split > 0:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_split, stratify=y, random_state=seed)
    else:
        X_temp, y_temp = X, y
        X_test, y_test = X[:0], y[:0]

    # Then split val from remaining
    if val_split > 0:
        # Adjust val fraction relative to remaining
        val_frac = val_split / (1 - test_split) if test_split < 1 else val_split
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=seed)
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = X_temp[:0], y_temp[:0]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.mean() * n_classes / n_classes  # keep scale ~1
    # Normalize to mean 1
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(
    X_train, y_train, X_val, y_val, X_test=None, y_test=None,
    batch_size: int = 64, num_workers: int = 2, use_weighted_sampler: bool = False,
    augment_train: bool = True, noise_std: float = 0.02, scale_jitter: float = 0.05,
):
    train_ds = ResidualWindowDataset(X_train, y_train, augment=augment_train, noise_std=noise_std, scale_jitter=scale_jitter)
    val_ds = ResidualWindowDataset(X_val, y_val, augment=False)
    test_ds = ResidualWindowDataset(X_test, y_test, augment=False) if X_test is not None and len(X_test) > 0 else None

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_weights = compute_class_weights(y_train)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None

    return train_loader, val_loader, test_loader


def mixup_data(x, y, alpha: float = 0.2, device="cpu"):
    """Returns mixed inputs, pairs of targets, and lambda. If alpha=0, no mixup."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
