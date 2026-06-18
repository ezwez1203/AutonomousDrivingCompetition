"""Pure Pursuit waypoint follower for CARLA maps."""
from __future__ import annotations

import math
from dataclasses import dataclass

import carla


@dataclass(slots=True)
class PurePursuitConfig:
    wheel_base_m: float = 2.875
    lookahead_base_m: float = 3.0
    lookahead_gain: float = 0.25
    min_lookahead_m: float = 2.0
    max_lookahead_m: float = 8.0
    max_steer_angle_rad: float = math.radians(35.0)


@dataclass(slots=True)
class PursuitTarget:
    waypoint: carla.Waypoint
    distance_m: float
    local_x_m: float
    local_y_m: float
    steer: float
    cross_track_error_m: float
    heading_error_rad: float


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    return angle


class PurePursuitController:
    """Select a lookahead waypoint and convert it to normalized steering."""

    def __init__(self, cfg: PurePursuitConfig | None = None):
        self.cfg = cfg or PurePursuitConfig()

    def lookahead_distance(self, speed_mps: float) -> float:
        dist = self.cfg.lookahead_base_m + self.cfg.lookahead_gain * max(0.0, speed_mps)
        return max(self.cfg.min_lookahead_m, min(self.cfg.max_lookahead_m, dist))

    def target(self, vehicle_transform: carla.Transform, waypoint: carla.Waypoint, speed_mps: float) -> PursuitTarget:
        lookahead = self.lookahead_distance(speed_mps)
        target_wp = self._advance_waypoint(waypoint, lookahead)
        local_x, local_y = self._to_vehicle_local(vehicle_transform, target_wp.transform.location)
        distance = max(1e-3, math.hypot(local_x, local_y))
        curvature = 2.0 * local_y / max(distance * distance, 1e-6)
        steer_angle = math.atan(self.cfg.wheel_base_m * curvature)
        steer = max(-1.0, min(1.0, steer_angle / self.cfg.max_steer_angle_rad))
        yaw = math.radians(vehicle_transform.rotation.yaw)
        wp_yaw = math.radians(target_wp.transform.rotation.yaw)
        heading_error = wrap_pi(wp_yaw - yaw)
        return PursuitTarget(
            waypoint=target_wp,
            distance_m=distance,
            local_x_m=local_x,
            local_y_m=local_y,
            steer=steer,
            cross_track_error_m=local_y,
            heading_error_rad=heading_error,
        )

    def _advance_waypoint(self, waypoint: carla.Waypoint, distance_m: float) -> carla.Waypoint:
        candidates = waypoint.next(distance_m)
        if not candidates:
            return waypoint
        # Keep same-lane continuity when CARLA returns several branch options.
        same_lane = [
            wp for wp in candidates
            if wp.road_id == waypoint.road_id and wp.lane_id == waypoint.lane_id
        ]
        return same_lane[0] if same_lane else candidates[0]

    @staticmethod
    def _to_vehicle_local(transform: carla.Transform, location: carla.Location) -> tuple[float, float]:
        dx = location.x - transform.location.x
        dy = location.y - transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        right_x = -math.sin(yaw)
        right_y = math.cos(yaw)
        local_x = dx * forward_x + dy * forward_y
        local_y = dx * right_x + dy * right_y
        return local_x, local_y
