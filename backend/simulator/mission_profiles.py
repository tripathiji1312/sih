"""
Mission profiles for the Rotax-912 simulator.
Each profile is a function: t (seconds) -> {rpm_command, altitude_m, oat_c, airspeed_ms, load_fraction}
"""
import math
import numpy as np


def _cruise(t, **kwargs):
    base_rpm = 4800 + 80 * math.sin(t * 0.02)  # gentle variation
    return dict(
        rpm_command=base_rpm,
        altitude_m=kwargs.get("altitude_m", 3000),
        oat_c=kwargs.get("oat_c", 15),
        airspeed_ms=55 + 5 * math.sin(t * 0.01),
        load_fraction=0.65,
        mixture_ratio=1.0,
    )


def _climb(t, **kwargs):
    progress = min(t / 600.0, 1.0)  # 10 min climb
    rpm = 5200 + 100 * progress
    alt = 1000 + 4000 * progress
    oat = 15 - 0.0065 * alt  # ISA lapse
    return dict(
        rpm_command=rpm,
        altitude_m=alt,
        oat_c=oat,
        airspeed_ms=45,
        load_fraction=0.85,
        mixture_ratio=1.0,
    )


def _descent(t, **kwargs):
    progress = min(t / 600.0, 1.0)
    rpm = 5200 - 1400 * progress
    alt = 5000 - 4000 * progress
    oat = 15 - 0.0065 * alt
    return dict(
        rpm_command=rpm,
        altitude_m=max(alt, 500),
        oat_c=oat,
        airspeed_ms=50,
        load_fraction=0.4,
        mixture_ratio=1.0,
    )


def _loiter(t, **kwargs):
    # Figure-8 loiter with throttle steps
    rpm = 4000 + 300 * math.sin(t * 0.05)
    return dict(
        rpm_command=rpm,
        altitude_m=kwargs.get("altitude_m", 3000),
        oat_c=kwargs.get("oat_c", 15),
        airspeed_ms=40,
        load_fraction=0.55,
        mixture_ratio=1.0,
    )


def _throttle_transitions(t, **kwargs):
    # Aggressive throttle cycles every 60s
    phase = (t % 60) / 60
    if phase < 0.25:
        rpm = 5200
    elif phase < 0.5:
        rpm = 3500
    elif phase < 0.75:
        rpm = 4800
    else:
        rpm = 4000
    rpm += np.random.randn() * 20
    return dict(
        rpm_command=float(np.clip(rpm, 1800, 5800)),
        altitude_m=kwargs.get("altitude_m", 3000),
        oat_c=kwargs.get("oat_c", 15),
        airspeed_ms=50,
        load_fraction=0.6 + 0.2 * math.sin(t * 0.1),
        mixture_ratio=1.0,
    )


def _hot_weather(t, **kwargs):
    return dict(
        rpm_command=5000,
        altitude_m=1000,
        oat_c=45,
        airspeed_ms=30,  # low airspeed = less cooling
        load_fraction=0.75,
        mixture_ratio=1.0,
    )


def _high_altitude(t, **kwargs):
    return dict(
        rpm_command=5200,
        altitude_m=7000,
        oat_c=-25,
        airspeed_ms=55,
        load_fraction=0.7,
        mixture_ratio=0.95,  # lean at altitude
    )


def _extreme_altitude(t, **kwargs):
    return dict(
        rpm_command=5400,
        altitude_m=8500,
        oat_c=-35,
        airspeed_ms=50,
        load_fraction=0.75,
        mixture_ratio=0.9,
    )


MISSION_PROFILES = {
    "cruise": _cruise,
    "climb": _climb,
    "descent": _descent,
    "loiter": _loiter,
    "throttle_transitions": _throttle_transitions,
    "hot_weather": _hot_weather,
    "high_altitude": _high_altitude,
    "extreme_altitude": _extreme_altitude,
}


def get_mission_profile(name: str):
    if name not in MISSION_PROFILES:
        raise ValueError(f"Unknown mission profile '{name}'. Available: {list(MISSION_PROFILES)}")
    return MISSION_PROFILES[name]
