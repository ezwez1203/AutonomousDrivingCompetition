"""LiDAR/Radar obstacle detection for Phase 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Obstacle, ObstacleObservation


@dataclass(slots=True)
class ObstacleDetectorConfig:
    x_min: float = -1.0
    x_max: float = 8.0
    y_abs_max: float = 3.0
    z_min: float = -0.4
    z_max: float = 1.6
    ground_z_max: float = -0.15
    grid_size_m: float = 0.35
    min_lidar_points: int = 8
    radar_x_min: float = -1.5
    radar_x_max: float = 2.0
    radar_y_abs_max: float = 1.6


class ObstacleDetector:
    """Simple ROI + grid clustering obstacle detector.

    LiDAR clustering uses connected occupied grid cells instead of DBSCAN to
    avoid adding scikit-learn as a hard dependency. Radar detections are kept
    as individual near-field obstacle candidates.
    """

    def __init__(self, cfg: ObstacleDetectorConfig | None = None):
        self.cfg = cfg or ObstacleDetectorConfig()

    def detect(self, lidar_points: np.ndarray, radar_points: np.ndarray) -> ObstacleObservation:
        obstacles: list[Obstacle] = []
        lidar_used = 0
        radar_used = 0

        if lidar_points is not None and len(lidar_points) > 0:
            lidar_obstacles, lidar_used = self._detect_lidar(lidar_points)
            obstacles.extend(lidar_obstacles)

        if radar_points is not None and len(radar_points) > 0:
            radar_obstacles, radar_used = self._detect_radar(radar_points)
            obstacles.extend(radar_obstacles)

        obstacles.sort(key=lambda obstacle: obstacle.distance)
        return ObstacleObservation(
            obstacles=obstacles,
            lidar_points_used=lidar_used,
            radar_points_used=radar_used,
        )

    def _detect_lidar(self, points: np.ndarray) -> tuple[list[Obstacle], int]:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return [], 0

        xyz = pts[:, :3]
        mask = (
            (xyz[:, 0] >= self.cfg.x_min)
            & (xyz[:, 0] <= self.cfg.x_max)
            & (np.abs(xyz[:, 1]) <= self.cfg.y_abs_max)
            & (xyz[:, 2] >= self.cfg.z_min)
            & (xyz[:, 2] <= self.cfg.z_max)
            & (xyz[:, 2] > self.cfg.ground_z_max)
        )
        roi = xyz[mask]
        if len(roi) < self.cfg.min_lidar_points:
            return [], int(len(roi))

        grid = np.floor(roi[:, :2] / self.cfg.grid_size_m).astype(np.int32)
        cell_to_indices: dict[tuple[int, int], list[int]] = {}
        for idx, cell in enumerate(grid):
            key = (int(cell[0]), int(cell[1]))
            cell_to_indices.setdefault(key, []).append(idx)

        visited: set[tuple[int, int]] = set()
        obstacles: list[Obstacle] = []
        for start in cell_to_indices:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            cluster_indices: list[int] = []
            while stack:
                cell = stack.pop()
                cluster_indices.extend(cell_to_indices[cell])
                cx, cy = cell
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        neighbor = (nx, ny)
                        if neighbor in cell_to_indices and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)

            if len(cluster_indices) < self.cfg.min_lidar_points:
                continue
            cluster = roi[np.asarray(cluster_indices, dtype=np.int32)]
            min_xyz = cluster.min(axis=0)
            max_xyz = cluster.max(axis=0)
            center = cluster.mean(axis=0)
            distance = float(np.linalg.norm(center[:2]))
            obstacles.append(
                Obstacle(
                    source="lidar",
                    x=float(center[0]),
                    y=float(center[1]),
                    z=float(center[2]),
                    distance=distance,
                    width=float(max_xyz[1] - min_xyz[1]),
                    length=float(max_xyz[0] - min_xyz[0]),
                    height=float(max_xyz[2] - min_xyz[2]),
                    points=int(len(cluster)),
                )
            )
        return obstacles, int(len(roi))

    def _detect_radar(self, points: np.ndarray) -> tuple[list[Obstacle], int]:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 8:
            return [], 0

        mask = (
            (pts[:, 0] >= self.cfg.radar_x_min)
            & (pts[:, 0] <= self.cfg.radar_x_max)
            & (np.abs(pts[:, 1]) <= self.cfg.radar_y_abs_max)
        )
        roi = pts[mask]
        obstacles: list[Obstacle] = []
        for row in roi:
            distance = float(np.linalg.norm(row[:2]))
            obstacles.append(
                Obstacle(
                    source=f"radar_{int(row[7])}",
                    x=float(row[0]),
                    y=float(row[1]),
                    z=float(row[2]),
                    distance=distance,
                    points=1,
                    velocity_mps=float(row[4]),
                )
            )
        return obstacles, int(len(roi))
