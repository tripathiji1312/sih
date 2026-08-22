"""
Inference wrapper — loads ONNX or PyTorch checkpoint and runs single-window inference.
Falls back gracefully if no model exists (returns uniform uncertainty).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from backend.config import MODELS_ROOT, N_CLASSES, FAULT_CLASSES


class EvidentialModelInference:
    def __init__(self, model_path: Path = None, use_onnx: bool = True):
        self.model_path_torch = model_path or (MODELS_ROOT / "evidential_model.pt")
        self.model_path_onnx = MODELS_ROOT / "evidential_model.onnx"
        self.use_onnx = use_onnx and HAS_ONNX
        self.session = None
        self.torch_model = None
        self.loaded = False
        self._try_load()

    def _try_load(self):
        # Prefer ONNX
        if self.use_onnx and self.model_path_onnx.exists():
            try:
                self.session = ort.InferenceSession(str(self.model_path_onnx), providers=["CPUExecutionProvider"])
                self.loaded = True
                return
            except Exception as e:
                print(f"[inference] ONNX load failed: {e}")

        if HAS_TORCH and self.model_path_torch.exists():
            try:
                from backend.ml.models.evidential_model import EvidentialFaultClassifier
                ckpt = torch.load(str(self.model_path_torch), map_location="cpu")
                # Handle both full ckpt dict and state_dict
                state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
                # Infer n_classes from state
                n_classes = state.get("evidence_head.3.weight", None)
                if n_classes is not None:
                    n_classes = n_classes.shape[0]
                else:
                    n_classes = N_CLASSES
                model = EvidentialFaultClassifier(n_classes=n_classes)
                model.load_state_dict(state, strict=False)
                model.eval()
                self.torch_model = model
                self.loaded = True
            except Exception as e:
                print(f"[inference] Torch load failed: {e}")

    def infer(self, window: np.ndarray) -> dict:
        """
        window: (14, 30) or (30, 14) or (1,14,30)
        Returns dict with fault_probability, uncertainties, probs, predicted class.
        """
        if not self.loaded:
            # uniform fallback
            K = N_CLASSES
            return {
                "fault_probability": 0.5,
                "epistemic_uncertainty": 1.0,
                "aleatoric_uncertainty": 0.5,
                "probs": [1.0 / K] * K,
                "predicted_class": 0,
                "predicted_label": FAULT_CLASSES[0],
                "dirichlet_params": [1.0] * K,
                "is_ood": False,
                "confidence": 0.0,
            }

        # Normalize shape to (1,14,30)
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape == (30, 14):
                arr = arr.T
            arr = arr[np.newaxis, ...]  # (1,14,30)
        elif arr.ndim == 3:
            if arr.shape[1] == 30 and arr.shape[2] == 14:
                arr = arr.transpose(0, 2, 1)

        if self.session is not None:
            # ONNX
            input_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {input_name: arr})
            # out[0] is evidence or alpha? We exported evidence; handle both
            evidence = out[0]  # (1,K)
            alpha = evidence + 1.0
            probs = alpha / alpha.sum(axis=-1, keepdims=True)
            epistemic = N_CLASSES / alpha.sum(axis=-1)
        else:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(arr)
                out = self.torch_model(t)
                alpha = out["alpha"].cpu().numpy()
                probs = out["probs"].cpu().numpy()
                epistemic = out["epistemic_uncertainty"].cpu().numpy()

        probs_list = probs[0].tolist()
        pred_idx = int(np.argmax(probs[0]))
        epistemic_val = float(epistemic[0] if hasattr(epistemic, "__len__") else epistemic)
        # fault probability = 1 - P(healthy)
        fault_prob = float(1.0 - probs[0][0])
        confidence = float(probs[0][pred_idx] * (1 - epistemic_val))

        return {
            "fault_probability": fault_prob,
            "epistemic_uncertainty": epistemic_val,
            "aleatoric_uncertainty": float(1.0 - max(probs_list)),
            "probs": probs_list,
            "predicted_class": pred_idx,
            "predicted_label": FAULT_CLASSES[pred_idx] if pred_idx < len(FAULT_CLASSES) else str(pred_idx),
            "dirichlet_params": alpha[0].tolist(),
            "confidence": float(np.clip(confidence, 0, 1)),
        }
