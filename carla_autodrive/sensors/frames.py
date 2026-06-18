"""Standard Phase 2 sensor frame passed to perception modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class PerceptionInput:
    """Synchronized best-effort snapshot in the vehicle coordinate frame.

    Arrays use CARLA vehicle coordinates: x forward, y right, z up.
    radar_points columns are:
        x, y, z, depth, velocity, azimuth, altitude, sensor_index
    """

    sim_frame: int
    timestamp: float
    vehicle_transform: dict
    speed_mps: float
    camera_bgra: np.ndarray | None = None
    lidar_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )
    radar_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 8), dtype=np.float32)
    )
    radar_by_name: dict[str, np.ndarray] = field(default_factory=dict)
    sensor_frames: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        camera = "none" if self.camera_bgra is None else str(tuple(self.camera_bgra.shape))
        return (
            f"frame={self.sim_frame} speed={self.speed_mps * 3.6:5.1f}km/h "
            f"camera={camera} lidar={len(self.lidar_points)} "
            f"radar={len(self.radar_points)}"
        )
