"""
Circular buffer — holds last 60s of validated frames in memory for sliding-window ML input.
Also handles conversion to residual windows for training.
"""
from collections import deque
from typing import List, Optional
import numpy as np

from backend.config import CIRCULAR_BUFFER_DURATION_S, SAMPLE_RATE_HZ, WINDOW_SIZE
from backend.simulator.engine_simulator import SensorFrame


class CircularBuffer:
    def __init__(self, duration_s: float = CIRCULAR_BUFFER_DURATION_S, sample_rate_hz: float = SAMPLE_RATE_HZ):
        self.duration_s = duration_s
        self.sample_rate_hz = sample_rate_hz
        self.maxlen = int(duration_s * sample_rate_hz)
        self.frames: deque[SensorFrame] = deque(maxlen=self.maxlen)
        # For ML we store residuals separately if needed
        self.residuals: deque[np.ndarray] = deque(maxlen=self.maxlen)

    def append(self, frame: SensorFrame):
        self.frames.append(frame)

    def append_residual(self, residual_vec: np.ndarray):
        self.residuals.append(residual_vec)

    def is_full(self) -> bool:
        return len(self.frames) >= WINDOW_SIZE

    def __len__(self):
        return len(self.frames)

    def get_ml_window(self) -> np.ndarray:
        """
        Returns (14, 30) array for model input — last WINDOW_SIZE residuals if available,
        otherwise last frames converted naïvely. Prefer residuals.
        """
        if len(self.residuals) >= WINDOW_SIZE:
            window = np.stack(list(self.residuals)[-WINDOW_SIZE:], axis=1)  # (14, 30)
            return window
        # fallback: stack frame vectors (not ideal, but keeps pipeline alive)
        if len(self.frames) >= WINDOW_SIZE:
            vecs = [f.to_vector() for f in list(self.frames)[-WINDOW_SIZE:]]
            # expand to 14 dims by padding
            arr = np.stack(vecs, axis=1)  # (11, 30)
            # pad to 14
            padded = np.zeros((14, WINDOW_SIZE))
            padded[: arr.shape[0], :] = arr
            return padded
        raise ValueError("Buffer not full")

    def get_recent_frames(self, n: int = 10) -> List[SensorFrame]:
        return list(self.frames)[-n:]

    def clear(self):
        self.frames.clear()
        self.residuals.clear()
