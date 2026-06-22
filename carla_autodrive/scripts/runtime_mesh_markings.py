#!/usr/bin/env python
"""Spawn road markings as runtime static mesh actors.

The CARLA packaged build exposes ``static.prop.mesh`` and Unreal's basic Plane
mesh.  The markings below are generated from smoothed approximations of the
visible road edges, so the edge lines stay a fixed inset from both road edges
and the dashed divider follows the midpoint between them.
"""
from __future__ import annotations

import argparse
import bisect
import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import carla

from carla_autodrive.maps import TrackPose, TrackSpec
from carla_autodrive.missions import mission_elements
from carla_autodrive.utils import get_logger, load_config

log = get_logger()

ROLE_NAME = "skku_runtime_marking"
DEFAULT_MESH_PATH = "/Engine/BasicShapes/Plane.Plane"

Point = tuple[float, float]


@dataclass(frozen=True)
class RuntimeMarkingConfig:
    mesh_path: str = DEFAULT_MESH_PATH
    z: float = 0.055
    tile_size_m: float | None = None
    tile_spacing_m: float | None = None
    crosswalk_tile_size_m: float = 0.25
    edge_sample_spacing_m: float = 0.18
    line_smooth_window_m: float = 0.45
    line_smooth_passes: int = 1
    line_smooth_max_displacement_m: float = 0.12
    max_actors: int = 3600


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Spawn lane/crosswalk markings as CARLA static mesh actors")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--clear-only", action="store_true", help="Destroy previous runtime markings and exit.")
    ap.add_argument("--keep-existing", action="store_true", help="Do not clear previous runtime markings before spawning.")
    ap.add_argument("--no-lanes", action="store_true", help="Skip lane markings.")
    ap.add_argument("--no-crosswalk", action="store_true", help="Skip crosswalk markings.")
    ap.add_argument("--mesh-path", default=DEFAULT_MESH_PATH)
    ap.add_argument("--z", type=float, default=0.055)
    ap.add_argument("--tile-size", type=float, default=None, help="Square lane tile side length in meters.")
    ap.add_argument("--tile-spacing", type=float, default=None, help="Lane tile spacing in meters.")
    ap.add_argument("--crosswalk-tile-size", type=float, default=0.25)
    ap.add_argument("--edge-sample-spacing", type=float, default=0.18)
    ap.add_argument("--line-smooth-window", type=float, default=0.45)
    ap.add_argument("--line-smooth-passes", type=int, default=1)
    ap.add_argument("--line-smooth-max-displacement", type=float, default=0.12)
    ap.add_argument("--max-actors", type=int, default=3600)
    return ap.parse_args()


def _marking_transform(pose: TrackPose, z: float) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=pose.x, y=-pose.y, z=z),
        carla.Rotation(yaw=math.degrees(-pose.heading)),
    )


def _mesh_blueprint(world: carla.World, cfg: RuntimeMarkingConfig, tile_size_m: float):
    bp = world.get_blueprint_library().find("static.prop.mesh")
    bp.set_attribute("mesh_path", cfg.mesh_path)
    bp.set_attribute("scale", f"{tile_size_m:.4f}")
    bp.set_attribute("role_name", ROLE_NAME)
    return bp


def clear_runtime_markings(world: carla.World) -> int:
    destroyed = 0
    for actor in list(world.get_actors().filter("static.prop.mesh")):
        if actor.attributes.get("role_name") != ROLE_NAME:
            continue
        try:
            if actor.is_alive:
                actor.destroy()
                destroyed += 1
        except RuntimeError:
            pass
    return destroyed


def _spawn_batch(world: carla.World, client: carla.Client | None, bp, transforms: list[carla.Transform]) -> list[carla.Actor]:
    if not transforms:
        return []
    if client is None:
        actors = []
        for tf in transforms:
            actor = world.try_spawn_actor(bp, tf)
            if actor is not None:
                actors.append(actor)
        return actors

    commands = [carla.command.SpawnActor(bp, tf) for tf in transforms]
    responses = client.apply_batch_sync(commands, True)
    actors: list[carla.Actor] = []
    failed = 0
    for response in responses:
        if response.error:
            failed += 1
            continue
        actor = world.get_actor(response.actor_id)
        if actor is not None:
            actors.append(actor)
    if failed:
        log.warning("runtime mesh marking spawn failed for %d/%d tiles", failed, len(transforms))
    return actors


def _smooth_closed_points(points: list[Point], spacing_m: float, cfg: RuntimeMarkingConfig) -> list[Point]:
    if len(points) < 5 or cfg.line_smooth_passes <= 0:
        return points
    radius = max(1, int(round(cfg.line_smooth_window_m / max(spacing_m, 1e-6) / 2.0)))
    radius = min(radius, max(1, len(points) // 8))
    weights = [radius + 1 - abs(i) for i in range(-radius, radius + 1)]
    weight_sum = float(sum(weights))
    original = points[:]
    smoothed = points[:]
    n = len(points)
    for _ in range(cfg.line_smooth_passes):
        nxt: list[Point] = []
        for idx in range(n):
            sx = 0.0
            sy = 0.0
            for rel, weight in zip(range(-radius, radius + 1), weights):
                x, y = smoothed[(idx + rel) % n]
                sx += x * weight
                sy += y * weight
            nx = sx / weight_sum
            ny = sy / weight_sum
            ox, oy = original[idx]
            disp = math.hypot(nx - ox, ny - oy)
            if cfg.line_smooth_max_displacement_m > 0.0 and disp > cfg.line_smooth_max_displacement_m:
                ratio = cfg.line_smooth_max_displacement_m / disp
                nx = ox + (nx - ox) * ratio
                ny = oy + (ny - oy) * ratio
            nxt.append((nx, ny))
        smoothed = nxt
    return smoothed


def _road_edge_points(spec: TrackSpec, spacing_m: float) -> tuple[list[Point], list[Point]]:
    total = spec.total_length()
    count = max(32, int(math.ceil(total / max(0.05, spacing_m))))
    inner: list[Point] = []
    outer: list[Point] = []
    for idx in range(count):
        pose = spec.reference_pose_at(total * idx / count)
        rx = math.sin(pose.heading)
        ry = -math.cos(pose.heading)
        inner.append((pose.x, pose.y))
        outer.append((pose.x + spec.road_width * rx, pose.y + spec.road_width * ry))
    return inner, outer


def _interpolate_between_edges(inner: list[Point], outer: list[Point], fraction: float) -> list[Point]:
    return [
        (ix + (ox - ix) * fraction, iy + (oy - iy) * fraction)
        for (ix, iy), (ox, oy) in zip(inner, outer)
    ]


def _closed_cumulative(points: list[Point]) -> tuple[list[float], float]:
    cumulative = [0.0]
    total = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        total += math.hypot(x2 - x1, y2 - y1)
        cumulative.append(total)
    return cumulative, total


def _point_at(points: list[Point], cumulative: list[float], total: float, s_m: float) -> tuple[Point, float]:
    if total <= 0.0:
        return points[0], 0.0
    s_m = s_m % total
    idx = max(0, min(len(points) - 1, bisect.bisect_right(cumulative, s_m) - 1))
    seg_start = cumulative[idx]
    seg_len = max(1e-9, cumulative[idx + 1] - seg_start)
    t = (s_m - seg_start) / seg_len
    x1, y1 = points[idx]
    x2, y2 = points[(idx + 1) % len(points)]
    x = x1 + (x2 - x1) * t
    y = y1 + (y2 - y1) * t
    heading = math.atan2(y2 - y1, x2 - x1)
    return (x, y), heading


def _append_line_transforms(
    points: list[Point],
    spacing_m: float,
    z: float,
    transforms: list[carla.Transform],
    max_actors: int,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> None:
    cumulative, total = _closed_cumulative(points)
    if end_s is None:
        end_s = total
    s_m = start_s
    while s_m <= end_s + 1e-6:
        if len(transforms) >= max_actors:
            raise RuntimeError(f"runtime marking actor limit reached: {max_actors}")
        (x, y), heading = _point_at(points, cumulative, total, s_m)
        transforms.append(_marking_transform(TrackPose(s_m, x, y, heading), z))
        s_m += max(0.05, spacing_m)


def spawn_lane_markings(
    world: carla.World,
    spec: TrackSpec,
    cfg: RuntimeMarkingConfig = RuntimeMarkingConfig(),
    client: carla.Client | None = None,
) -> list[carla.Actor]:
    tile_size = cfg.tile_size_m if cfg.tile_size_m is not None else max(0.12, spec.lane_mark_width)
    base_spacing = cfg.tile_spacing_m if cfg.tile_spacing_m is not None else tile_size * 0.85
    solid_spacing = base_spacing * 0.68
    dashed_spacing = tile_size * 0.92
    edge_inset = min(spec.lane_width * 0.35, max(spec.lane_mark_width * 0.5, tile_size * 0.60))
    inner, outer = _road_edge_points(spec, cfg.edge_sample_spacing_m)
    inner_line = _smooth_closed_points(_interpolate_between_edges(inner, outer, edge_inset / spec.road_width), cfg.edge_sample_spacing_m, cfg)
    outer_line = _smooth_closed_points(_interpolate_between_edges(inner, outer, 1.0 - edge_inset / spec.road_width), cfg.edge_sample_spacing_m, cfg)
    divider_line = _smooth_closed_points(_interpolate_between_edges(inner, outer, 0.5), cfg.edge_sample_spacing_m, cfg)

    bp = _mesh_blueprint(world, cfg, tile_size)
    marks = spec.cfg.get("lanes", {}).get("marks", {})
    transforms: list[carla.Transform] = []

    if marks.get("inner", "solid") == "solid":
        _append_line_transforms(inner_line, solid_spacing, cfg.z, transforms, cfg.max_actors)
    if marks.get("outer", "solid") == "solid":
        _append_line_transforms(outer_line, solid_spacing, cfg.z, transforms, cfg.max_actors)

    if marks.get("divider", "broken") == "broken":
        dash = max(0.1, spec.mm(spec.cfg.get("lanes", {}).get("dash_length_mm", 300)))
        gap = max(0.1, spec.mm(spec.cfg.get("lanes", {}).get("dash_gap_mm", 300)))
        cumulative, total = _closed_cumulative(divider_line)
        dash_start = 0.0
        while dash_start < total:
            dash_end = min(dash_start + dash, total)
            s_m = dash_start + dashed_spacing * 0.5
            while s_m <= dash_end - dashed_spacing * 0.25:
                if len(transforms) >= cfg.max_actors:
                    raise RuntimeError(f"runtime marking actor limit reached: {cfg.max_actors}")
                (x, y), heading = _point_at(divider_line, cumulative, total, s_m)
                transforms.append(_marking_transform(TrackPose(s_m, x, y, heading), cfg.z + 0.002))
                s_m += dashed_spacing
            dash_start += dash + gap

    return _spawn_batch(world, client, bp, transforms)


def spawn_crosswalk_markings(
    world: carla.World,
    spec: TrackSpec,
    s_mm: float,
    cfg: RuntimeMarkingConfig = RuntimeMarkingConfig(),
    client: carla.Client | None = None,
) -> list[carla.Actor]:
    dims = spec.cfg["dimensions"]
    crosswalk_length = spec.mm(dims["crosswalk_mm"][1])
    crosswalk_width = spec.road_width + 0.6
    stripe_count = max(1, int(spec.cfg.get("lanes", {}).get("crosswalk_stripes", 4)))
    stripe_length = min(0.18, crosswalk_length / (stripe_count * 1.5))
    gap = 0.0 if stripe_count == 1 else max(0.05, (crosswalk_length - stripe_count * stripe_length) / (stripe_count - 1))
    tile_size = max(0.08, cfg.crosswalk_tile_size_m)
    spacing = tile_size * 0.85
    bp = _mesh_blueprint(world, cfg, tile_size)
    transforms: list[carla.Transform] = []

    start_s_m = spec.mm(s_mm) - crosswalk_length / 2.0
    for idx in range(stripe_count):
        stripe_center_s = start_s_m + idx * (stripe_length + gap) + stripe_length / 2.0
        lateral = -0.3
        while lateral <= crosswalk_width:
            if len(transforms) >= cfg.max_actors:
                raise RuntimeError(f"runtime marking actor limit reached: {cfg.max_actors}")
            pose = spec.reference_pose_at(stripe_center_s)
            x = pose.x + lateral * math.sin(pose.heading)
            y = pose.y - lateral * math.cos(pose.heading)
            transforms.append(_marking_transform(TrackPose(pose.s, x, y, pose.heading), cfg.z + 0.003))
            lateral += spacing

    return _spawn_batch(world, client, bp, transforms)


def spawn_runtime_markings(
    world: carla.World,
    spec: TrackSpec,
    cfg: RuntimeMarkingConfig = RuntimeMarkingConfig(),
    include_lanes: bool = True,
    include_crosswalk: bool = True,
    clear_existing: bool = True,
    client: carla.Client | None = None,
) -> list[carla.Actor]:
    if clear_existing:
        removed = clear_runtime_markings(world)
        if removed:
            log.info("runtime markings cleared: actors=%d", removed)

    actors: list[carla.Actor] = []
    if include_lanes:
        actors.extend(spawn_lane_markings(world, spec, cfg, client=client))
    if include_crosswalk:
        crosswalk_s = mission_elements(spec)["crosswalk"]["s"]
        actors.extend(spawn_crosswalk_markings(world, spec, crosswalk_s, cfg, client=client))
    log.info("runtime mesh markings spawned: actors=%d mesh=%s", len(actors), cfg.mesh_path)
    return actors


def main() -> int:
    args = parse_args()
    client_cfg = load_config("sim")["client"]
    host = args.host or client_cfg["host"]
    port = args.port or client_cfg["port"]
    client = carla.Client(host, port)
    client.set_timeout(client_cfg["timeout"])
    world = client.get_world()
    spec = TrackSpec()
    removed = 0 if args.keep_existing else clear_runtime_markings(world)
    if removed:
        log.info("runtime markings cleared: actors=%d", removed)
    if args.clear_only:
        return 0
    cfg = RuntimeMarkingConfig(
        mesh_path=args.mesh_path,
        z=args.z,
        tile_size_m=args.tile_size,
        tile_spacing_m=args.tile_spacing,
        crosswalk_tile_size_m=args.crosswalk_tile_size,
        edge_sample_spacing_m=args.edge_sample_spacing,
        line_smooth_window_m=args.line_smooth_window,
        line_smooth_passes=args.line_smooth_passes,
        line_smooth_max_displacement_m=args.line_smooth_max_displacement,
        max_actors=args.max_actors,
    )
    spawn_runtime_markings(
        world,
        spec,
        cfg,
        include_lanes=not args.no_lanes,
        include_crosswalk=not args.no_crosswalk,
        clear_existing=False,
        client=client,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
