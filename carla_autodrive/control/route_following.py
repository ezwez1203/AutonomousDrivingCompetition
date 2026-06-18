"""Closed-route Pure Pursuit follower for custom TrackSpec routes."""
from __future__ import annotations

import math
from dataclasses import dataclass

import carla
import numpy as np

from carla_autodrive.maps import TrackSpec

from .pure_pursuit import PurePursuitConfig, wrap_pi


@dataclass(slots=True)
class RoutePoint:
    x: float
    y: float
    heading: float
    s: float
    reverse: bool = False
    speed_limit_mps: float | None = None


@dataclass(slots=True)
class RouteTarget:
    point: RoutePoint
    distance_m: float
    local_x_m: float
    local_y_m: float
    steer: float
    cross_track_error_m: float
    heading_error_rad: float
    speed_limit_mps: float | None = None
    reason: str | None = None
    reverse: bool = False


def build_track_lane_route(
    spec: TrackSpec | None = None,
    *,
    lane: int = 1,
    spacing_mm: float = 100.0,
    carla_coordinates: bool = True,
) -> list[RoutePoint]:
    """Sample a closed route from config/track.yaml lane center.

    CARLA's generated OpenDriveMap mirrors the OpenDRIVE y axis into world
    coordinates, so by default y and heading are sign-flipped to match actor
    locations returned by the simulator.
    """
    spec = spec or TrackSpec()
    total_mm = spec.total_length() / spec.scale * 1000.0
    if total_mm <= 0:
        raise ValueError("track route length is zero")
    count = max(4, int(math.ceil(total_mm / spacing_mm)))
    route: list[RoutePoint] = []
    for idx in range(count):
        s_mm = idx * total_mm / count
        pose = spec.lane_center_pose(s_mm, lane)
        if carla_coordinates:
            route.append(RoutePoint(pose.x, -pose.y, -pose.heading, pose.s))
        else:
            route.append(RoutePoint(pose.x, pose.y, pose.heading, pose.s))
    return route


class RoutePurePursuitController:
    """Pure Pursuit controller on a fixed closed route.

    The nearest route point search is constrained to a window around the previous
    index.  This avoids snapping to a nearby but topologically different part of
    a closed custom track.
    """

    def __init__(
        self,
        route: list[RoutePoint],
        cfg: PurePursuitConfig | None = None,
        *,
        closed_route: bool = True,
        search_back: int = 15,
        search_ahead: int = 100,
        curve_speed_enabled: bool = True,
        curve_speed_cap_mps: float = 3.0,
        curve_speed_min_mps: float = 1.2,
        curve_speed_max_lat_acc_mps2: float = 0.45,
        curve_speed_lookahead_m: float = 8.0,
        steer_speed_gain_mps: float = 2.2,
        steer_speed_threshold: float = 0.15,
        stop_at_end: bool = False,
        end_slowdown_distance_m: float = 5.0,
        finish_distance_m: float = 0.8,
        reverse_lookahead_m: float = 0.6,
        reverse_steer_scale: float = 0.55,
        reverse_end_min_speed_mps: float = 0.12,
    ):
        if len(route) < 4:
            raise ValueError("route needs at least 4 points")
        self.route = route
        self.cfg = cfg or PurePursuitConfig()
        self.closed_route = bool(closed_route)
        self.search_back = int(search_back)
        self.search_ahead = int(search_ahead)
        self.curve_speed_enabled = curve_speed_enabled
        self.curve_speed_cap_mps = float(curve_speed_cap_mps)
        self.curve_speed_min_mps = float(curve_speed_min_mps)
        self.curve_speed_max_lat_acc_mps2 = float(curve_speed_max_lat_acc_mps2)
        self.curve_speed_lookahead_m = float(curve_speed_lookahead_m)
        self.steer_speed_gain_mps = float(steer_speed_gain_mps)
        self.steer_speed_threshold = float(steer_speed_threshold)
        self.stop_at_end = bool(stop_at_end)
        self.end_slowdown_distance_m = float(end_slowdown_distance_m)
        self.finish_distance_m = float(finish_distance_m)
        self.reverse_lookahead_m = float(reverse_lookahead_m)
        self.reverse_steer_scale = float(reverse_steer_scale)
        self.reverse_end_min_speed_mps = float(reverse_end_min_speed_mps)
        self.index: int | None = None
        self._xy = np.asarray([(p.x, p.y) for p in route], dtype=np.float64)
        self._segment_lengths = self._compute_segment_lengths()
        self._segment_heading_deltas = self._compute_segment_heading_deltas()
        reverse_indices = [idx for idx, point in enumerate(route) if point.reverse]
        self._first_reverse_idx = reverse_indices[0] if reverse_indices else None
        self._reverse_started = False

    @property
    def current_index(self) -> int | None:
        """Nearest route index selected during the latest target calculation."""
        return self.index

    @property
    def current_point(self) -> RoutePoint | None:
        if self.index is None:
            return None
        return self.route[self.index]

    @property
    def current_s_m(self) -> float | None:
        point = self.current_point
        return None if point is None else float(point.s)

    def lookahead_distance(self, speed_mps: float) -> float:
        dist = self.cfg.lookahead_base_m + self.cfg.lookahead_gain * max(0.0, speed_mps)
        return max(self.cfg.min_lookahead_m, min(self.cfg.max_lookahead_m, dist))

    def target(self, vehicle_transform: carla.Transform, speed_mps: float) -> RouteTarget:
        nearest_idx = self._nearest_index(vehicle_transform.location)
        if self._first_reverse_idx is not None and not self._reverse_started:
            staging_idx = max(0, self._first_reverse_idx - 1)
            staging = self.route[staging_idx]
            staging_distance = math.hypot(
                staging.x - vehicle_transform.location.x,
                staging.y - vehicle_transform.location.y,
            )
            if (staging_distance <= self.finish_distance_m or nearest_idx >= staging_idx) and speed_mps <= 0.25:
                self._reverse_started = True
                self.index = self._first_reverse_idx
                nearest_idx = self._first_reverse_idx
            elif nearest_idx >= staging_idx:
                self.index = staging_idx
                return self._make_target(
                    vehicle_transform,
                    nearest_idx=staging_idx,
                    target_idx=staging_idx,
                    speed_limit=0.0,
                    reason="gear_shift_to_reverse",
                    reverse=False,
                )

        if self._first_reverse_idx is not None and not self._reverse_started:
            target_idx = min(self._advance_index(nearest_idx, self.lookahead_distance(speed_mps)), self._first_reverse_idx - 1)
        elif self._reverse_started and nearest_idx < self._first_reverse_idx:
            nearest_idx = self._first_reverse_idx
            target_idx = self._advance_index(nearest_idx, self._reverse_lookahead_distance(speed_mps))
        elif self._reverse_started:
            target_idx = self._advance_index(nearest_idx, self._reverse_lookahead_distance(speed_mps))
        else:
            target_idx = self._advance_index(nearest_idx, self.lookahead_distance(speed_mps))
        self.index = nearest_idx
        return self._make_target(vehicle_transform, nearest_idx=nearest_idx, target_idx=target_idx)

    def _make_target(
        self,
        vehicle_transform: carla.Transform,
        *,
        nearest_idx: int,
        target_idx: int,
        speed_limit: float | None = None,
        reason: str | None = None,
        reverse: bool | None = None,
    ) -> RouteTarget:
        point = self.route[target_idx]
        nearest_point = self.route[nearest_idx]
        local_x, local_y = self._to_vehicle_local(vehicle_transform, point)
        _, nearest_local_y = self._to_vehicle_local(vehicle_transform, nearest_point)
        distance = max(1e-3, math.hypot(local_x, local_y))
        target_reverse = bool(point.reverse if reverse is None else reverse)
        pursuit_local_y = -local_y if target_reverse else local_y
        curvature = 2.0 * pursuit_local_y / max(distance * distance, 1e-6)
        steer_angle = math.atan(self.cfg.wheel_base_m * curvature)
        steer = max(-1.0, min(1.0, steer_angle / self.cfg.max_steer_angle_rad))
        if target_reverse:
            steer = max(-1.0, min(1.0, steer * self.reverse_steer_scale))
        yaw = math.radians(vehicle_transform.rotation.yaw)
        heading_error = wrap_pi(nearest_point.heading - yaw)
        route_speed_limit = self._speed_limit(nearest_idx, abs(steer))
        if speed_limit is None:
            speed_limit = route_speed_limit
        elif route_speed_limit is not None:
            speed_limit = min(speed_limit, route_speed_limit)
        if point.speed_limit_mps is not None:
            speed_limit = float(point.speed_limit_mps) if speed_limit is None else min(speed_limit, float(point.speed_limit_mps))
        allow_end_stop = self._first_reverse_idx is None or self._reverse_started
        if not self.closed_route and self.stop_at_end and allow_end_stop:
            end_speed_limit, end_reason = self._end_speed_limit(vehicle_transform.location, reverse=target_reverse)
            if end_speed_limit is not None:
                speed_limit = end_speed_limit if speed_limit is None else min(speed_limit, end_speed_limit)
                reason = reason or end_reason
        return RouteTarget(
            point=point,
            distance_m=distance,
            local_x_m=local_x,
            local_y_m=local_y,
            steer=steer,
            cross_track_error_m=nearest_local_y,
            heading_error_rad=heading_error,
            speed_limit_mps=speed_limit,
            reason=reason,
            reverse=target_reverse,
        )

    def is_finished(self, location: carla.Location, speed_mps: float = 0.0) -> bool:
        if self.closed_route or not self.stop_at_end:
            return False
        if self._first_reverse_idx is not None and not self._reverse_started:
            return False
        if self._first_reverse_idx is not None and (self.index is None or self.index < len(self.route) - 4):
            return False
        final = self.route[-1]
        distance = math.hypot(final.x - location.x, final.y - location.y)
        return distance <= self.finish_distance_m and speed_mps <= 0.25

    def _compute_segment_lengths(self) -> np.ndarray:
        if self.closed_route:
            nxt = np.roll(self._xy, -1, axis=0)
            return np.linalg.norm(nxt - self._xy, axis=1)
        lengths = np.zeros(len(self.route), dtype=np.float64)
        lengths[:-1] = np.linalg.norm(self._xy[1:] - self._xy[:-1], axis=1)
        return lengths

    def _compute_segment_heading_deltas(self) -> np.ndarray:
        headings = np.asarray([p.heading for p in self.route], dtype=np.float64)
        if self.closed_route:
            nxt = np.roll(headings, -1)
            return np.asarray([abs(wrap_pi(float(b - a))) for a, b in zip(headings, nxt)], dtype=np.float64)
        deltas = np.zeros(len(self.route), dtype=np.float64)
        deltas[:-1] = [abs(wrap_pi(float(b - a))) for a, b in zip(headings[:-1], headings[1:])]
        return deltas

    def _upcoming_curvature(self, start_idx: int) -> float:
        remaining = max(0.0, self.curve_speed_lookahead_m)
        if remaining <= 0.0:
            return 0.0
        idx = start_idx
        total_distance = 0.0
        total_turn = 0.0
        while remaining > 0.0:
            seg = float(self._segment_lengths[idx])
            if seg > 1e-6:
                used = min(seg, remaining)
                ratio = used / seg
                total_distance += used
                total_turn += float(self._segment_heading_deltas[idx]) * ratio
                remaining -= used
            if not self.closed_route and idx >= len(self.route) - 1:
                break
            idx = (idx + 1) % len(self.route)
            if idx == start_idx:
                break
        if total_distance <= 1e-6:
            return 0.0
        return total_turn / total_distance

    def _speed_limit(self, nearest_idx: int, abs_steer: float) -> float | None:
        if not self.curve_speed_enabled:
            return None

        speed_limit = self.curve_speed_cap_mps
        curvature = self._upcoming_curvature(nearest_idx)
        if curvature > 1e-6:
            speed_limit = min(speed_limit, math.sqrt(self.curve_speed_max_lat_acc_mps2 / curvature))

        steer_excess = max(0.0, abs_steer - self.steer_speed_threshold)
        if steer_excess > 0.0:
            speed_limit = min(speed_limit, self.curve_speed_cap_mps - self.steer_speed_gain_mps * steer_excess)

        return max(self.curve_speed_min_mps, min(self.curve_speed_cap_mps, speed_limit))

    def _reverse_lookahead_distance(self, speed_mps: float) -> float:
        return max(0.25, min(self.reverse_lookahead_m + 0.15 * max(0.0, speed_mps), 1.2))

    def _nearest_index(self, location: carla.Location) -> int:
        query = np.asarray([location.x, location.y], dtype=np.float64)
        n = len(self.route)
        if self.index is None:
            distances = np.sum((self._xy - query) ** 2, axis=1)
            return int(np.argmin(distances))

        if self.closed_route:
            candidates = [(self.index + offset) % n for offset in range(-self.search_back, self.search_ahead + 1)]
        else:
            start = max(0, self.index - self.search_back)
            end = min(n - 1, self.index + self.search_ahead)
            candidates = list(range(start, end + 1))
        candidate_xy = self._xy[np.asarray(candidates, dtype=np.int32)]
        distances = np.sum((candidate_xy - query) ** 2, axis=1)
        return int(candidates[int(np.argmin(distances))])

    def _advance_index(self, start_idx: int, distance_m: float) -> int:
        n = len(self.route)
        idx = start_idx
        remaining = distance_m
        while remaining > 0.0:
            if not self.closed_route and idx >= n - 1:
                return n - 1
            seg = float(self._segment_lengths[idx])
            remaining -= seg
            idx = (idx + 1) % n
            if idx == start_idx:
                break
        return idx

    def _end_speed_limit(self, location: carla.Location, *, reverse: bool = False) -> tuple[float | None, str | None]:
        final = self.route[-1]
        distance = math.hypot(final.x - location.x, final.y - location.y)
        if distance <= self.finish_distance_m:
            return 0.0, "route_stop"
        if distance >= self.end_slowdown_distance_m:
            return None, None
        ratio = distance / max(1e-6, self.end_slowdown_distance_m)
        min_speed = self.reverse_end_min_speed_mps if reverse else 0.4
        return max(min_speed, self.curve_speed_cap_mps * ratio), "route_approach"

    @staticmethod
    def _to_vehicle_local(transform: carla.Transform, point: RoutePoint) -> tuple[float, float]:
        dx = point.x - transform.location.x
        dy = point.y - transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        right_x = -math.sin(yaw)
        right_y = math.cos(yaw)
        local_x = dx * forward_x + dy * forward_y
        local_y = dx * right_x + dy * right_y
        return local_x, local_y
