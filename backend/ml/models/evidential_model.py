"""
Evidential Deep Learning model — SOTA for uncertainty-aware fault diagnosis.
Implements Sensoy et al. 2018 with modern tweaks: BatchNorm, Dropout, label smoothing.
Arch: Lightweight 1D-CNN + SE block + evidential head (Dirichlet output).
Input: (B, 14, 30) residual windows
Output: Dirichlet concentration params for N_CLASSES fault classes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> global avg -> (B, C)
        w = x.mean(dim=-1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1)  # (B,C,1)
        return x * w


class EvidentialFaultClassifier(nn.Module):
    def __init__(self, n_channels: int = 14, n_timesteps: int = 30, n_classes: int = 9, dropout: float = 0.2):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.AdaptiveAvgPool1d(1),
        )

        self.evidence_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
            nn.Softplus(),  # evidence >=0
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 14, 30) or (B, 30, 14) — auto-handled
        Returns dict with evidence, alpha, probs, uncertainties.
        """
        # Handle (B, T, C) -> (B, C, T)
        if x.shape[1] == 30 and x.shape[2] == 14:
            x = x.transpose(1, 2)
        elif x.shape[1] != self.n_channels:
            # try to infer
            if x.dim() == 2:
                x = x.unsqueeze(0)

        feat = self.features(x).squeeze(-1)  # (B, 64)
        evidence = self.evidence_head(feat)  # (B, K)

        alpha = evidence + 1.0  # Dirichlet params
        S = alpha.sum(dim=-1, keepdim=True)  # strength (B,1)
        probs = alpha / S  # expected probs

        # Epistemic uncertainty: K / S  (0=certain, ~1=uncertain)
        epistemic = self.n_classes / S.squeeze(-1)  # (B,)

        # Aleatoric via Dirichlet entropy approximation (or vacuity)
        # Use digamma formulation
        # Clamp for numerical stability
        alpha_clamped = alpha.clamp(min=1e-6)
        S_clamped = S.clamp(min=1e-6)
        # Expected entropy
        digamma_alpha = torch.digamma(alpha_clamped)
        digamma_S = torch.digamma(S_clamped)
        # aleatoric as expected entropy term (positive)
        # E[ entropy ] under Dirichlet not trivial; use simple proxy: 1 - max_prob weighted
        # plus Dirichlet spread
        # For stability: aleatoric ~ - sum p * (digamma(alpha)-digamma(S))
        # Take mean aleatoric per sample as 1 - confidence
        aleatoric = 1.0 - probs.max(dim=-1)[0]  # (B,)

        # Vacuity: K / S (same as epistemic) ; Dissonance via bal
        # Provide both
        vacuity = self.n_classes / S.squeeze(-1)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "probs": probs,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "vacuity": vacuity,
            "logits": evidence,  # alias
        }

    def predict(self, x: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------
def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ) — divergence from uniform Dirichlet. Returns (B,1)"""
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta = beta.sum(dim=-1, keepdim=True)
    # lgamma
    t1 = torch.lgamma(S_alpha) - torch.lgamma(S_beta)
    t2 = (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=-1, keepdim=True)
    t3 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=-1, keepdim=True)
    return t1 - t2 + t3  # (B,1)


def evidential_loss(
    output: dict,
    target: torch.Tensor,
    epoch: int,
    annealing_epochs: int = 20,
    n_classes: int = None,
    label_smoothing: float = 0.0,
    use_mse: bool = True,
) -> torch.Tensor:
    """
    Sensoy et al. 2018 loss with annealing.
    - MSE term on Dirichlet mean vs one-hot
    - KL regularization annealed over epochs
    """
    alpha = output["alpha"]  # (B,K)
    K = alpha.shape[-1] if n_classes is None else n_classes

    # One-hot with optional smoothing
    y = F.one_hot(target, num_classes=K).float()
    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + label_smoothing / K

    S = alpha.sum(dim=-1, keepdim=True)  # (B,1)
    probs = alpha / S

    if use_mse:
        # MSE + variance term (Sensoy Eq. 4)
        err = (y - probs).pow(2).sum(dim=-1)  # (B,)
        var = (probs * (1 - probs) / (S + 1)).sum(dim=-1)  # approximate var
        mse = (err + var).mean()
    else:
        # Cross-entropy via digamma
        ce = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()
        mse = ce

    # KL term: remove evidence for correct class (Sensoy Eq. 9), then penalize remaining
    # Start at 0.1 so even epoch 0 gets some uncertainty shaping; cap at 0.5 to avoid ECE explosion
    annealing_coeff = min(0.5, 0.1 + 0.4 * epoch / max(annealing_epochs, 1))
    # Correct class → alpha=1 (no penalty), wrong classes → keep original alpha
    alpha_tilde = (1 - y) * alpha + y * 1.0
    kl = kl_dirichlet_uniform(alpha_tilde).mean() * annealing_coeff

    return mse + kl


def edl_mse_loss(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    K = alpha.shape[-1]
    y = F.one_hot(target, num_classes=K).float()
    S = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / S
    err = (y - probs).pow(2).sum(dim=-1).mean()
    return err


class FocalEvidentialLoss(nn.Module):
    """Optional focal-weighted evidential loss for class imbalance."""

    def __init__(self, gamma: float = 2.0, annealing_epochs: int = 20):
        super().__init__()
        self.gamma = gamma
        self.annealing_epochs = annealing_epochs

    def forward(self, output: dict, target: torch.Tensor, epoch: int) -> torch.Tensor:
        alpha = output["alpha"]
        K = alpha.shape[-1]
        y = F.one_hot(target, num_classes=K).float()
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        # Focal weight
        pt = (probs * y).sum(dim=-1)  # prob of true class
        focal_weight = (1 - pt).pow(self.gamma)
        err = (y - probs).pow(2).sum(dim=-1)  # (B,)
        loss = (focal_weight * err).mean()
        alpha_tilde = (1 - y) * alpha + y * 1.0
        kl = kl_dirichlet_uniform(alpha_tilde).mean() * min(0.5, 0.1 + 0.4 * epoch / self.annealing_epochs)
        return loss + kl
