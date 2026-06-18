"""Small PID controller for vehicle speed."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PIDConfig:
    kp: float = 0.25
    ki: float = 0.02
    kd: float = 0.0
    integral_limit: float = 8.0
    output_min: float = -1.0
    output_max: float = 1.0


class PIDController:
    def __init__(self, cfg: PIDConfig | None = None):
        self.cfg = cfg or PIDConfig()
        self.integral = 0.0
        self.prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def step(self, error: float, dt: float) -> float:
        dt = max(float(dt), 1e-3)
        self.integral += error * dt
        self.integral = max(-self.cfg.integral_limit, min(self.cfg.integral_limit, self.integral))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        output = self.cfg.kp * error + self.cfg.ki * self.integral + self.cfg.kd * derivative
        return max(self.cfg.output_min, min(self.cfg.output_max, output))
