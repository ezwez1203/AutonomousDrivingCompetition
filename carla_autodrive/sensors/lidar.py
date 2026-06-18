"""LiDAR wrapper."""
from __future__ import annotations

import numpy as np

from .base import SensorBase


class Lidar(SensorBase):
    """Store sensor.lidar.ray_cast data as an (N, 4) numpy array: [x, y, z, intensity]."""

    def __init__(self, cfg: dict):
        super().__init__("lidar", cfg)

    def _configure_blueprint(self, bp) -> None:
        for attr in ("channels", "range", "points_per_second",
                     "rotation_frequency", "upper_fov", "lower_fov"):
            if attr in self.cfg:
                bp.set_attribute(attr, str(self.cfg[attr]))

    def _on_data(self, measurement) -> None:
        points = np.frombuffer(measurement.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))  # x, y, z, intensity
        self._store(measurement.frame, points)

    def summary(self) -> str:
        """one-line summary for console output."""
        frame, pts = self.get_latest()
        if pts is None or len(pts) == 0:
            return "LiDAR: (waiting for data)"
        dist = np.linalg.norm(pts[:, :3], axis=1)
        return (f"LiDAR: frame={frame} points={len(pts):>6} "
                f"closest={dist.min():.2f}m farthest={dist.max():.2f}m")
