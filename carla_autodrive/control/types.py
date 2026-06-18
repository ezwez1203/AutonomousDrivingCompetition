"""Control command/status types for Phase 4."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ControlCommand:
    """Normalized vehicle command."""

    throttle: float
    steer: float
    brake: float
    current_speed_mps: float
    target_speed_mps: float
    desired_speed_mps: float
    target_distance_m: float
    cross_track_error_m: float
    heading_error_rad: float
    reason: str = "lane_follow"
    reverse: bool = False

    def summary(self) -> str:
        gear = "R" if self.reverse else "D"
        return (
            f"cmd gear={gear} throttle={self.throttle:.2f} steer={self.steer:+.2f} brake={self.brake:.2f} "
            f"speed={self.current_speed_mps:.2f}m/s target={self.target_speed_mps:.2f}m/s "
            f"desired={self.desired_speed_mps:.2f}m/s "
            f"cte={self.cross_track_error_m:+.2f}m heading={self.heading_error_rad:+.2f} "
            f"{self.reason}"
        )
