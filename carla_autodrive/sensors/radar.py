"""Radar sensor wrapper."""
from __future__ import annotations

import math

import numpy as np

from .base import SensorBase


class Radar(SensorBase):
    """sensor.other.radar -> numpy array [depth, velocity, azimuth, altitude, x, y, z]."""

    def __init__(self, cfg: dict, name: str = "radar"):
        super().__init__(name, cfg)

    def _configure_blueprint(self, bp) -> None:
        for attr in ("horizontal_fov", "vertical_fov", "range", "points_per_second"):
            if attr in self.cfg:
                bp.set_attribute(attr, str(self.cfg[attr]))
        if "sensor_tick" in self.cfg:
            bp.set_attribute("sensor_tick", str(self.cfg["sensor_tick"]))

    def _on_data(self, measurement) -> None:
        rows = []
        for detection in measurement:
            depth = float(detection.depth)
            azimuth = float(detection.azimuth)
            altitude = float(detection.altitude)
            x = depth * math.cos(altitude) * math.cos(azimuth)
            y = depth * math.cos(altitude) * math.sin(azimuth)
            z = depth * math.sin(altitude)
            rows.append((
                depth,
                float(detection.velocity),
                azimuth,
                altitude,
                x,
                y,
                z,
            ))

        payload = np.asarray(rows, dtype=np.float32).reshape((-1, 7))
        self._store(measurement.frame, payload)

    def summary(self) -> str:
        """Return a compact one-line summary for console logs."""
        frame, detections = self.get_latest()
        if detections is None or len(detections) == 0:
            return f"{self.name}: (waiting for data)"
        depth = detections[:, 0]
        velocity = detections[:, 1]
        return (f"{self.name}: frame={frame} detections={len(detections):>4} "
                f"closest={depth.min():.2f}m mean_v={velocity.mean():+.2f}m/s")
