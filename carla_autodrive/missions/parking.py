"""Parking maneuver route presets for the SKKU track."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from carla_autodrive.control import RoutePoint
from carla_autodrive.maps import TrackPose, TrackSpec


DEFAULT_PARKING_ELEMENTS: dict[str, Any] = {
    "parking_zone1": {"s": 10800, "lane": 1},
    "parking_zone2": {"s": 7600, "lane": 1},
}


@dataclass(slots=True)
class ParkingZone:
    label: str
    s_mm: float
    lane: int
    slot_x_mm: float | None = None
    slot_y_mm: float | None = None
    slot_heading_deg: float | None = None
    reverse_staging_distance_mm: float | None = None

    @property
    def has_explicit_slot(self) -> bool:
        return self.slot_x_mm is not None and self.slot_y_mm is not None and self.slot_heading_deg is not None


def parking_elements(spec: TrackSpec) -> dict[str, Any]:
    elements = spec.cfg.get("elements")
    if not isinstance(elements, dict):
        elements = {}
    merged = dict(DEFAULT_PARKING_ELEMENTS)
    merged.update({key: value for key, value in elements.items() if key.startswith("parking_zone") and value})
    return merged


def selected_parking_zone(spec: TrackSpec, zone_idx: int = 1) -> ParkingZone:
    if zone_idx not in (1, 2):
        raise ValueError(f"parking zone must be 1 or 2: {zone_idx}")
    item = parking_elements(spec)[f"parking_zone{zone_idx}"]
    slot = item.get("slot") if isinstance(item, dict) else None
    if isinstance(slot, dict):
        return ParkingZone(
            label=f"parking_zone{zone_idx}",
            s_mm=float(item["s"]),
            lane=int(item["lane"]),
            slot_x_mm=float(slot["x"]),
            slot_y_mm=float(slot["y"]),
            slot_heading_deg=float(slot["heading_deg"]),
            reverse_staging_distance_mm=float(slot.get("reverse_staging_distance_mm", 1800.0)),
        )
    return ParkingZone(label=f"parking_zone{zone_idx}", s_mm=float(item["s"]), lane=int(item["lane"]))


def parking_zone_pose(spec: TrackSpec, zone_idx: int = 1) -> TrackPose:
    """Return the explicit slot pose when present, otherwise the lane-center fallback."""
    return _slot_pose(spec, selected_parking_zone(spec, zone_idx))


def build_parking_maneuver_route(
    spec: TrackSpec | None = None,
    *,
    zone_idx: int = 2,
    drive_lane: int = 2,
    start_s_mm: float = 0.0,
    spacing_mm: float = 100.0,
    approach_mm: float = 3200.0,
    transition_mm: float = 2600.0,
    overshoot_mm: float = 0.0,
    carla_coordinates: bool = True,
) -> list[RoutePoint]:
    """Build an open forward pull-in route ending at the selected parking zone."""
    spec = spec or TrackSpec()
    lane_count = int(spec.cfg["lanes"]["count"])
    if drive_lane < 1 or drive_lane > lane_count:
        raise ValueError(f"drive_lane must be 1..{lane_count}: {drive_lane}")

    zone = selected_parking_zone(spec, zone_idx)
    if zone.lane < 1 or zone.lane > lane_count:
        raise ValueError(f"parking lane must be 1..{lane_count}: {zone.lane}")

    total_mm = spec.total_length() / spec.scale * 1000.0
    end_s_mm = zone.s_mm + max(0.0, overshoot_mm)
    if end_s_mm <= start_s_mm:
        end_s_mm += total_mm

    count = max(4, int(math.ceil((end_s_mm - start_s_mm) / spacing_mm)) + 1)
    route: list[RoutePoint] = []
    for idx in range(count):
        ratio = idx / max(1, count - 1)
        s_unwrapped = start_s_mm + (end_s_mm - start_s_mm) * ratio
        s_mm = s_unwrapped % total_mm
        blend = _parking_blend(s_unwrapped, zone.s_mm, approach_mm, transition_mm)
        lane_offset_m = (
            (drive_lane - 0.5) * (1.0 - blend)
            + (zone.lane - 0.5) * blend
        ) * spec.lane_width
        pose = spec.pose_with_right_offset(s_mm, lane_offset_m)
        route.append(_route_point(pose, carla_coordinates))

    final_pose = _slot_pose(spec, zone)
    final = _route_point(final_pose, carla_coordinates)
    if math.hypot(route[-1].x - final.x, route[-1].y - final.y) > 0.05:
        route.append(final)
    else:
        route[-1] = final
    return route


def build_reverse_parking_maneuver_route(
    spec: TrackSpec | None = None,
    *,
    zone_idx: int = 2,
    drive_lane: int = 2,
    start_s_mm: float = 0.0,
    spacing_mm: float = 100.0,
    staging_after_mm: float = 1800.0,
    reverse_transition_mm: float = 2200.0,
    reverse_speed_mps: float = 0.5,
    carla_coordinates: bool = True,
) -> list[RoutePoint]:
    """Build a forward approach plus reverse pull-in route ending in a parking zone."""
    spec = spec or TrackSpec()
    lane_count = int(spec.cfg["lanes"]["count"])
    if drive_lane < 1 or drive_lane > lane_count:
        raise ValueError(f"drive_lane must be 1..{lane_count}: {drive_lane}")

    zone = selected_parking_zone(spec, zone_idx)
    if zone.lane < 1 or zone.lane > lane_count:
        raise ValueError(f"parking lane must be 1..{lane_count}: {zone.lane}")

    total_mm = spec.total_length() / spec.scale * 1000.0
    staging_s_mm = zone.s_mm + max(spacing_mm, staging_after_mm)
    if staging_s_mm <= start_s_mm:
        staging_s_mm += total_mm

    route: list[RoutePoint] = []

    forward_count = max(4, int(math.ceil((staging_s_mm - start_s_mm) / spacing_mm)) + 1)
    for idx in range(forward_count):
        ratio = idx / max(1, forward_count - 1)
        s_unwrapped = start_s_mm + (staging_s_mm - start_s_mm) * ratio
        blend = _parking_blend(s_unwrapped, zone.s_mm, approach_mm=3200.0, transition_mm=2600.0)
        lane_offset_m = (
            (drive_lane - 0.5) * (1.0 - blend)
            + (zone.lane - 0.5) * blend
        ) * spec.lane_width
        pose = spec.pose_with_right_offset(s_unwrapped % total_mm, lane_offset_m)
        route.append(_route_point(pose, carla_coordinates, reverse=False))

    if zone.has_explicit_slot:
        reverse_distance_mm = zone.reverse_staging_distance_mm or staging_after_mm
        final_pose = _slot_pose(spec, zone)
        staging_pose = _offset_pose(final_pose, spec.mm(reverse_distance_mm), direction=1.0)
        route[-1] = _route_point(staging_pose, carla_coordinates, reverse=False)
        reverse_count = max(4, int(math.ceil(reverse_distance_mm / spacing_mm)) + 1)
        for idx in range(1, reverse_count):
            ratio = idx / max(1, reverse_count - 1)
            pose = _interpolate_pose(staging_pose, final_pose, ratio)
            point = _route_point(pose, carla_coordinates, reverse=True)
            point.speed_limit_mps = reverse_speed_mps
            route.append(point)
    else:
        reverse_count = max(4, int(math.ceil(max(spacing_mm, staging_after_mm) / spacing_mm)) + 1)
        for idx in range(reverse_count):
            ratio = idx / max(1, reverse_count - 1)
            s_unwrapped = staging_s_mm - max(spacing_mm, staging_after_mm) * ratio
            lane_offset_m = (zone.lane - 0.5) * spec.lane_width
            pose = spec.pose_with_right_offset(s_unwrapped % total_mm, lane_offset_m)
            point = _route_point(pose, carla_coordinates, reverse=True)
            point.speed_limit_mps = reverse_speed_mps
            route.append(point)
        final_pose = spec.lane_center_pose(zone.s_mm, zone.lane)

    final = _route_point(final_pose, carla_coordinates, reverse=True)
    final.speed_limit_mps = reverse_speed_mps
    route[-1] = final
    return route


def _parking_blend(s_unwrapped_mm: float, parking_s_mm: float, approach_mm: float, transition_mm: float) -> float:
    start = parking_s_mm - max(0.0, approach_mm)
    end = parking_s_mm - max(1.0, approach_mm - transition_mm)
    if s_unwrapped_mm <= start:
        return 0.0
    if s_unwrapped_mm >= end:
        return 1.0
    return _smoothstep((s_unwrapped_mm - start) / max(1.0, end - start))


def _route_point(pose: TrackPose, carla_coordinates: bool, *, reverse: bool = False) -> RoutePoint:
    if carla_coordinates:
        return RoutePoint(pose.x, -pose.y, -pose.heading, pose.s, reverse=reverse)
    return RoutePoint(pose.x, pose.y, pose.heading, pose.s, reverse=reverse)


def _slot_pose(spec: TrackSpec, zone: ParkingZone) -> TrackPose:
    if not zone.has_explicit_slot:
        return spec.lane_center_pose(zone.s_mm, zone.lane)
    return TrackPose(
        spec.mm(zone.s_mm),
        spec.mm(float(zone.slot_x_mm)),
        spec.mm(float(zone.slot_y_mm)),
        math.radians(float(zone.slot_heading_deg)),
        zone.lane,
    )


def _offset_pose(pose: TrackPose, distance_m: float, *, direction: float) -> TrackPose:
    return TrackPose(
        pose.s,
        pose.x + direction * distance_m * math.cos(pose.heading),
        pose.y + direction * distance_m * math.sin(pose.heading),
        pose.heading,
        pose.lane,
    )


def _interpolate_pose(start: TrackPose, end: TrackPose, ratio: float) -> TrackPose:
    t = max(0.0, min(1.0, ratio))
    return TrackPose(
        end.s,
        start.x + (end.x - start.x) * t,
        start.y + (end.y - start.y) * t,
        end.heading,
        end.lane,
    )


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
