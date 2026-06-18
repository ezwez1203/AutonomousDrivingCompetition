"""Typed outputs for Phase 3 perception."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LaneObservation:
    """Camera-derived lane/track alignment estimate."""

    detected: bool
    center_error_px: float = 0.0
    center_error_norm: float = 0.0
    heading_error_rad: float = 0.0
    lane_center_px: float | None = None
    image_center_px: float | None = None
    confidence: float = 0.0
    mask_pixels: int = 0
    message: str = ""


@dataclass(slots=True)
class ParkingLineObservation:
    """Camera-derived parking line alignment estimate."""

    detected: bool
    center_error_px: float = 0.0
    center_error_norm: float = 0.0
    confidence: float = 0.0
    mask_pixels: int = 0
    message: str = ""


@dataclass(slots=True)
class Obstacle:
    """Obstacle candidate in vehicle coordinates."""

    source: str
    x: float
    y: float
    z: float
    distance: float
    width: float = 0.0
    length: float = 0.0
    height: float = 0.0
    points: int = 0
    velocity_mps: float | None = None


@dataclass(slots=True)
class ObstacleObservation:
    """LiDAR/Radar obstacle candidates."""

    obstacles: list[Obstacle] = field(default_factory=list)
    lidar_points_used: int = 0
    radar_points_used: int = 0

    @property
    def nearest(self) -> Obstacle | None:
        if not self.obstacles:
            return None
        return min(self.obstacles, key=lambda obstacle: obstacle.distance)


@dataclass(slots=True)
class PerceptionOutput:
    """Combined Phase 3 output passed to control/FSM."""

    sim_frame: int
    timestamp: float
    speed_mps: float
    lane: LaneObservation
    parking: ParkingLineObservation
    obstacles: ObstacleObservation

    def summary(self) -> str:
        lane = (
            f"lane err={self.lane.center_error_norm:+.3f} "
            f"heading={self.lane.heading_error_rad:+.3f} conf={self.lane.confidence:.2f}"
            if self.lane.detected
            else f"lane undetected ({self.lane.message})"
        )
        parking = (
            f"parking err={self.parking.center_error_norm:+.3f} conf={self.parking.confidence:.2f}"
            if self.parking.detected
            else f"parking undetected ({self.parking.message})"
        )
        nearest = self.obstacles.nearest
        obstacle = (
            f"nearest {nearest.source} x={nearest.x:.2f} y={nearest.y:.2f} d={nearest.distance:.2f}m"
            if nearest is not None
            else "no obstacle"
        )
        return f"frame={self.sim_frame} {lane} | {parking} | {obstacle}"
