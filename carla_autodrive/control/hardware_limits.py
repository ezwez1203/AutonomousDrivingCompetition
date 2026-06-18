"""Hardware output limits for final vehicle command clamping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HardwareLimitConfig:
    smps_voltage_v: float = 12.0
    allow_voltage_boost: bool = False
    pwm_min: int = 0
    pwm_max: int = 255
    current_limit_a: float | None = None
    driver_protection_enabled: bool = True
    enabled: bool = True
    max_throttle_cmd: float = 0.45
    max_reverse_cmd: float = 0.45
    max_brake_cmd: float = 0.75
    max_accel_delta_per_sec: float = 3.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "HardwareLimitConfig":
        cfg = cfg or {}
        power = cfg.get("power") or {}
        motor = cfg.get("motor_driver") or {}
        clamp = cfg.get("control_clamp") or {}
        return cls(
            smps_voltage_v=float(power.get("smps_voltage_v", 12.0)),
            allow_voltage_boost=bool(power.get("allow_voltage_boost", False)),
            pwm_min=int(motor.get("pwm_min", 0)),
            pwm_max=int(motor.get("pwm_max", 255)),
            current_limit_a=_optional_float(motor.get("current_limit_a")),
            driver_protection_enabled=bool(motor.get("driver_protection_enabled", True)),
            enabled=bool(clamp.get("enabled", True)),
            max_throttle_cmd=float(clamp.get("max_throttle_cmd", 0.45)),
            max_reverse_cmd=float(clamp.get("max_reverse_cmd", 0.45)),
            max_brake_cmd=float(clamp.get("max_brake_cmd", 0.75)),
            max_accel_delta_per_sec=float(clamp.get("max_accel_delta_per_sec", 3.0)),
        )


class HardwareLimiter:
    """Final safety clamp for drive/brake commands before VehicleControl."""

    def __init__(self, cfg: HardwareLimitConfig | None = None):
        self.cfg = cfg or HardwareLimitConfig()
        self._last_throttle = 0.0

    def clamp(self, *, throttle: float, brake: float, reverse: bool, dt: float) -> tuple[float, float]:
        if not self.cfg.enabled:
            return float(throttle), float(brake)

        max_drive = self.cfg.max_reverse_cmd if reverse else self.cfg.max_throttle_cmd
        limited_throttle = _clamp(float(throttle), 0.0, max_drive)
        limited_brake = _clamp(float(brake), 0.0, self.cfg.max_brake_cmd)

        max_delta = max(0.0, self.cfg.max_accel_delta_per_sec) * max(0.0, dt)
        if max_delta > 0.0 and limited_throttle > self._last_throttle + max_delta:
            limited_throttle = self._last_throttle + max_delta
        self._last_throttle = limited_throttle
        return limited_throttle, limited_brake

    def reset(self) -> None:
        self._last_throttle = 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
