"""Obstacle avoidance route presets for the SKKU track."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from carla_autodrive.control import RoutePoint
from carla_autodrive.maps import TrackPose, TrackSpec


DEFAULT_ELEMENTS: dict[str, Any] = {
    "start_line": {"s": 0, "lane": 2},
    "laptimer_T1": {"s": 0},
    "laptimer_T2": {"s": 9100},
    "laptimer_T3": {"s": 14500},
    "laptimer_T4": {"s": 23000},
    "crosswalk": {"s": 17000},
    "obstacle_1": {"s": 4100, "lane": 1},
    "obstacle_2_presets": [
        {"s": 6900, "lane": 1},
        {"s": 7600, "lane": 1},
        {"s": 8300, "lane": 1},
    ],
    "obstacle_3_presets": [
        {"s": 9300, "lane": 2},
        {"s": 10100, "lane": 2},
        {"s": 10900, "lane": 2},
    ],
}


@dataclass(slots=True)
class ObstaclePreset:
    label: str
    s_mm: float
    lane: int


def mission_elements(spec: TrackSpec) -> dict[str, Any]:
    elements = spec.cfg.get("elements")
    if not isinstance(elements, dict):
        elements = {}
    merged = dict(DEFAULT_ELEMENTS)
    merged.update({key: value for key, value in elements.items() if value})
    return merged


def selected_obstacle_presets(
    spec: TrackSpec,
    *,
    obstacle2_idx: int = 0,
    obstacle3_idx: int = 0,
) -> list[ObstaclePreset]:
    elements = mission_elements(spec)
    obstacle2 = elements["obstacle_2_presets"][obstacle2_idx]
    obstacle3 = elements["obstacle_3_presets"][obstacle3_idx]
    return [
        _preset("obstacle_1", elements["obstacle_1"]),
        _preset(f"obstacle_2_p{obstacle2_idx}", obstacle2),
        _preset(f"obstacle_3_p{obstacle3_idx}", obstacle3),
    ]


def build_obstacle_avoidance_route(
    spec: TrackSpec | None = None,
    *,
    drive_lane: int = 2,
    obstacle2_idx: int = 0,
    obstacle3_idx: int = 0,
    spacing_mm: float = 100.0,
    hold_mm: float = 1200.0,
    transition_mm: float = 1400.0,
    carla_coordinates: bool = True,
) -> list[RoutePoint]:
    """Build a closed route that changes lane around selected obstacle presets."""
    spec = spec or TrackSpec()
    lane_count = int(spec.cfg["lanes"]["count"])
    if drive_lane < 1 or drive_lane > lane_count:
        raise ValueError(f"drive_lane must be 1..{lane_count}: {drive_lane}")

    total_mm = spec.total_length() / spec.scale * 1000.0
    if total_mm <= 0:
        raise ValueError("track route length is zero")
    count = max(4, int(math.ceil(total_mm / spacing_mm)))
    obstacles = selected_obstacle_presets(
        spec,
        obstacle2_idx=obstacle2_idx,
        obstacle3_idx=obstacle3_idx,
    )
    avoid_lane = 1 if drive_lane != 1 else min(2, lane_count)

    route: list[RoutePoint] = []
    for idx in range(count):
        s_mm = idx * total_mm / count
        lane_blend = 0.0
        for obstacle in obstacles:
            if obstacle.lane == drive_lane:
                lane_blend = max(lane_blend, _avoidance_blend(s_mm, obstacle.s_mm, total_mm, hold_mm, transition_mm))

        lane_offset_m = ((drive_lane - 0.5) * (1.0 - lane_blend) + (avoid_lane - 0.5) * lane_blend) * spec.lane_width
        pose = spec.pose_with_right_offset(s_mm, lane_offset_m)
        route.append(_route_point(pose, carla_coordinates))
    return route


def _preset(label: str, item: dict[str, Any]) -> ObstaclePreset:
    return ObstaclePreset(label=label, s_mm=float(item["s"]), lane=int(item["lane"]))


def _route_point(pose: TrackPose, carla_coordinates: bool) -> RoutePoint:
    if carla_coordinates:
        return RoutePoint(pose.x, -pose.y, -pose.heading, pose.s)
    return RoutePoint(pose.x, pose.y, pose.heading, pose.s)


def _avoidance_blend(s_mm: float, obstacle_s_mm: float, total_mm: float, hold_mm: float, transition_mm: float) -> float:
    dist = abs(_signed_cyclic_delta(s_mm, obstacle_s_mm, total_mm))
    hold_half = max(0.0, hold_mm) / 2.0
    if dist <= hold_half:
        return 1.0
    if transition_mm <= 1e-6:
        return 0.0
    t = (dist - hold_half) / transition_mm
    if t >= 1.0:
        return 0.0
    return 1.0 - _smoothstep(t)


def _signed_cyclic_delta(s_mm: float, center_s_mm: float, total_mm: float) -> float:
    return ((s_mm - center_s_mm + total_mm / 2.0) % total_mm) - total_mm / 2.0


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
