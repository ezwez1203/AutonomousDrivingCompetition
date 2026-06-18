"""Phase 3 perception pipeline."""
from __future__ import annotations

from carla_autodrive.sensors.frames import PerceptionInput

from .lane import LaneDetector, LaneDetectorConfig, ParkingLineDetector, ParkingLineDetectorConfig
from .obstacles import ObstacleDetector, ObstacleDetectorConfig
from .types import PerceptionOutput


class PerceptionPipeline:
    """Run all Phase 3 detectors on a Phase 2 sensor snapshot."""

    def __init__(
        self,
        lane_cfg: LaneDetectorConfig | None = None,
        parking_cfg: ParkingLineDetectorConfig | None = None,
        obstacle_cfg: ObstacleDetectorConfig | None = None,
    ):
        self.lane_detector = LaneDetector(lane_cfg)
        self.parking_line_detector = ParkingLineDetector(parking_cfg)
        self.obstacle_detector = ObstacleDetector(obstacle_cfg)

    def process(self, snapshot: PerceptionInput) -> PerceptionOutput:
        lane = self.lane_detector.detect(snapshot.camera_bgra)
        parking = self.parking_line_detector.detect(snapshot.camera_bgra)
        obstacles = self.obstacle_detector.detect(
            snapshot.lidar_points,
            snapshot.radar_points,
        )
        return PerceptionOutput(
            sim_frame=snapshot.sim_frame,
            timestamp=snapshot.timestamp,
            speed_mps=snapshot.speed_mps,
            lane=lane,
            parking=parking,
            obstacles=obstacles,
        )
