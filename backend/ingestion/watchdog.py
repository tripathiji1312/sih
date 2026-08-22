"""
Watchdog — per-channel staleness + dropout tracking (archi.md 4.6)
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from backend.simulator.engine_simulator import SensorFrame


@dataclass
class ChannelWatchdog:
    channel_name: str
    max_staleness_s: float = 2.0
    last_update_time: float = 0.0
    expected_rate_hz: float = 10.0
    frame_count: int = 0
    dropout_count: int = 0
    malformed_count: int = 0
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
        staleness = current_time - self.last_update_time if self.last_update_time else 999.0
        is_stale = staleness > self.max_staleness_s
        if len(self._intervals) >= 10:
            avg_interval = float(np.mean(list(self._intervals)))
            measured_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0
            rate_ok = measured_rate > self.expected_rate_hz * 0.5
        else:
            measured_rate = 0.0
            rate_ok = True
        return {
            "channel": self.channel_name,
            "staleness_s": round(staleness, 2),
            "is_stale": bool(is_stale),
            "measured_rate_hz": round(measured_rate, 1),
            "rate_ok": bool(rate_ok),
            "dropouts": self.dropout_count,
            "malformed": self.malformed_count,
            "status": "OK" if (not is_stale and rate_ok) else "DEGRADED" if not is_stale else "STALE",
        }


class SystemWatchdog:
    def __init__(self):
        self.channels = {
            "rpm": ChannelWatchdog("rpm"),
            "cht": ChannelWatchdog("cht"),
            "egt": ChannelWatchdog("egt"),
            "oil": ChannelWatchdog("oil"),
            "fuel": ChannelWatchdog("fuel"),
            "vibration": ChannelWatchdog("vibration"),
        }

    def on_frame(self, frame: SensorFrame, valid: bool = True):
        ts = frame.timestamp
        for wd in self.channels.values():
            wd.on_frame(ts, valid=valid)

    def system_health(self) -> dict:
        current = time.time()
        channel_health = {name: wd.check_health(current) for name, wd in self.channels.items()}
        n_stale = sum(1 for h in channel_health.values() if h["is_stale"])
        n_degraded = sum(1 for h in channel_health.values() if h["status"] == "DEGRADED")
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
                "CRITICAL_DATA_LOSS": "Multiple channels stale — falling back to physics-only mode",
            }[overall],
        }
