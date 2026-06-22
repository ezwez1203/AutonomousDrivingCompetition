"""Short-range ultrasonic-style obstacle sensor wrapper."""
from __future__ import annotations

from dataclasses import dataclass

from .base import SensorBase


@dataclass(slots=True)
class UltrasonicReading:
    frame: int
    distance_m: float
    actor_id: int | None
    actor_type: str | None


class FrontUltrasonic(SensorBase):
    """Approximate a front ultrasonic sensor with CARLA's obstacle detector."""

    def __init__(self, cfg: dict, name: str = "ultrasonic_front"):
        super().__init__(name, cfg)

    def _configure_blueprint(self, bp) -> None:
        for attr in ("distance", "hit_radius", "only_dynamics", "debug_linetrace", "sensor_tick"):
            if attr in self.cfg:
                bp.set_attribute(attr, str(self.cfg[attr]))

    def _on_data(self, event) -> None:
        other = getattr(event, "other_actor", None)
        reading = UltrasonicReading(
            frame=int(event.frame),
            distance_m=float(event.distance),
            actor_id=None if other is None else int(other.id),
            actor_type=None if other is None else str(other.type_id),
        )
        self._store(event.frame, reading)

    def latest_reading(self) -> UltrasonicReading | None:
        _frame, reading = self.get_latest()
        return reading

    def summary(self) -> str:
        frame, reading = self.get_latest()
        if reading is None:
            return f"{self.name}: clear/no echo"
        actor = "unknown" if reading.actor_type is None else reading.actor_type
        return f"{self.name}: frame={frame} distance={reading.distance_m:.2f}m actor={actor}"
