#!/usr/bin/env python
"""Phase 1 runtime layout: spawn marker actors and monitor logical trigger zones.

This is the Python-runtime counterpart to permanent Unreal TriggerVolume/StaticMesh
placement. It places visible CARLA prop actors at mission element boundaries and
checks vehicle positions against the same oriented boxes.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import carla

from carla_autodrive.maps import TrackPose, TrackSpec
from carla_autodrive.missions import mission_elements, parking_zone_pose
from carla_autodrive.scripts.phase1_draw_elements import draw_box, selected_obstacle_poses, transform_from_pose
from carla_autodrive.utils import get_logger, load_config

log = get_logger()


@dataclass(frozen=True)
class Zone:
    name: str
    pose: TrackPose
    length: float
    width: float
    kind: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 1: mission prop placement and logical trigger monitoring")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--duration", type=float, default=120.0, help="Placement/monitoring duration in seconds.")
    ap.add_argument("--tick", type=float, default=0.1, help="Trigger polling interval in seconds.")
    ap.add_argument("--ego-id", type=int, default=None, help="Monitor only this vehicle actor id.")
    ap.add_argument("--no-monitor", action="store_true", help="Skip trigger monitoring.")
    ap.add_argument("--spawn-obstacles", action="store_true", help="Spawn three obstacle vehicles.")
    ap.add_argument("--obstacle2", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--obstacle3", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--keep", action="store_true", help="Keep spawned actors on exit.")
    ap.add_argument("--marker-blueprint", default="static.prop.trafficcone01")
    ap.add_argument("--barrier-blueprint", default="static.prop.streetbarrier")
    ap.add_argument("--signal-blueprint", default="static.prop.trafficwarning")
    ap.add_argument("--obstacle-blueprint", default="vehicle.tesla.model3")
    ap.add_argument("--spawn-test-vehicle-zone", default=None,
                    help="Spawn a trigger test vehicle at the center of the selected zone.")
    ap.add_argument("--test-vehicle-blueprint", default="vehicle.tesla.model3")
    return ap.parse_args()


def zone_specs(spec: TrackSpec) -> list[Zone]:
    dims = spec.cfg["dimensions"]
    elements = mission_elements(spec)
    start = elements["start_line"]
    t4 = elements["laptimer_T4"]
    crosswalk = elements["crosswalk"]

    zones = [
        Zone("start_line", spec.road_center_pose(start["s"]), spec.mm(dims["start_line_mm"]), spec.road_width, "lap"),
        Zone("laptimer_T4", spec.road_center_pose(t4["s"]), spec.mm(dims["start_line_mm"]), spec.road_width, "lap"),
        Zone("crosswalk", spec.road_center_pose(crosswalk["s"]), spec.mm(dims["crosswalk_mm"][1]),
             spec.mm(dims["crosswalk_mm"][0]), "traffic"),
    ]

    for key in ("parking_zone1", "parking_zone2"):
        item = elements[key]
        zone_idx = int(key.removeprefix("parking_zone"))
        slot = item.get("slot", {}) if isinstance(item, dict) else {}
        parking_length = spec.mm(slot.get("length_mm", dims["parking_mm"][1]))
        parking_width = spec.mm(slot.get("width_mm", dims["parking_mm"][0]))
        zones.append(Zone(key, parking_zone_pose(spec, zone_idx), parking_length, parking_width, "parking"))
    return zones


def oriented_corners(pose: TrackPose, length: float, width: float) -> list[tuple[float, float]]:
    hx = length / 2.0
    hy = width / 2.0
    c = math.cos(pose.heading)
    s = math.sin(pose.heading)
    corners = []
    for lx, ly in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)):
        x = pose.x + lx * c - ly * s
        y = pose.y + lx * s + ly * c
        corners.append((x, y))
    return corners


def contains(zone: Zone, loc: carla.Location) -> bool:
    dx = loc.x - zone.pose.x
    dy = loc.y - zone.pose.y
    c = math.cos(zone.pose.heading)
    s = math.sin(zone.pose.heading)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return abs(local_x) <= zone.length / 2.0 and abs(local_y) <= zone.width / 2.0


def spawn_actor(world: carla.World, blueprint_id: str, transform: carla.Transform, label: str):
    try:
        bp = world.get_blueprint_library().find(blueprint_id)
        actor = world.try_spawn_actor(bp, transform)
    except RuntimeError as exc:
        log.warning("actor spawn failed: %s blueprint=%s err=%s", label, blueprint_id, exc)
        return None
    if actor is None:
        log.warning("actor spawn failed: %s blueprint=%s", label, blueprint_id)
        return None
    log.info("actor spawned: %-18s id=%d blueprint=%s", label, actor.id, blueprint_id)
    return actor


def spawn_zone_markers(world: carla.World, zones: list[Zone], args: argparse.Namespace) -> list[carla.Actor]:
    actors: list[carla.Actor] = []
    for zone in zones:
        color = carla.Color(255, 255, 255)
        if zone.kind == "traffic":
            color = carla.Color(255, 220, 0)
        elif zone.kind == "parking":
            color = carla.Color(0, 220, 80)
        elif zone.name == "laptimer_T4":
            color = carla.Color(0, 220, 220)
        draw_box(world, zone.pose, zone.name.upper(), zone.length, zone.width, 0.12, color, args.duration)

        marker_bp = args.marker_blueprint
        if zone.kind == "lap":
            marker_bp = args.barrier_blueprint
        for idx, (x, y) in enumerate(oriented_corners(zone.pose, zone.length, zone.width)):
            tf = carla.Transform(carla.Location(x=x, y=y, z=0.15), carla.Rotation(yaw=math.degrees(zone.pose.heading)))
            actor = spawn_actor(world, marker_bp, tf, f"{zone.name}_corner{idx}")
            if actor is not None:
                actors.append(actor)

        if zone.kind == "traffic":
            # Put a visible warning sign beside the crosswalk, offset to the road right side.
            x, y = oriented_corners(zone.pose, zone.length + 2.0, zone.width + 2.0)[0]
            tf = carla.Transform(carla.Location(x=x, y=y, z=0.4), carla.Rotation(yaw=math.degrees(zone.pose.heading)))
            actor = spawn_actor(world, args.signal_blueprint, tf, "traffic_signal_marker")
            if actor is not None:
                actors.append(actor)
    return actors


def spawn_obstacles(world: carla.World, spec: TrackSpec, args: argparse.Namespace) -> list[carla.Actor]:
    actors: list[carla.Actor] = []
    bp = world.get_blueprint_library().find(args.obstacle_blueprint)
    for label, pose in selected_obstacle_poses(spec, args.obstacle2, args.obstacle3):
        actor = world.try_spawn_actor(bp, transform_from_pose(pose, z=0.35))
        if actor is None:
            log.warning("obstacle spawn failed: %s", label)
            continue
        try:
            actor.set_simulate_physics(False)
        except RuntimeError:
            pass
        actors.append(actor)
        log.info("obstacle actor spawned: %-14s id=%d", label, actor.id)
    return actors


def spawn_test_vehicle(world: carla.World, zones: list[Zone], args: argparse.Namespace) -> carla.Actor | None:
    if not args.spawn_test_vehicle_zone:
        return None
    zone_by_name = {zone.name: zone for zone in zones}
    zone = zone_by_name.get(args.spawn_test_vehicle_zone)
    if zone is None:
        raise ValueError(f"unknown zone: {args.spawn_test_vehicle_zone}. candidates={sorted(zone_by_name)}")
    bp = world.get_blueprint_library().find(args.test_vehicle_blueprint)
    actor = world.try_spawn_actor(bp, transform_from_pose(zone.pose, z=0.35))
    if actor is None:
        log.warning("trigger test vehicle spawn failed: zone=%s", zone.name)
        return None
    try:
        actor.set_simulate_physics(False)
    except RuntimeError:
        pass
    args.ego_id = actor.id
    log.info("trigger test vehicle spawned: zone=%s id=%d", zone.name, actor.id)
    return actor


def watched_vehicles(world: carla.World, ego_id: int | None) -> list[carla.Actor]:
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if ego_id is None:
        return vehicles
    return [actor for actor in vehicles if actor.id == ego_id]


def monitor_triggers(world: carla.World, zones: list[Zone], args: argparse.Namespace) -> None:
    log.info("trigger monitoring started: duration=%.1fs tick=%.2fs ego_id=%s",
             args.duration, args.tick, args.ego_id)
    active: set[tuple[int, str]] = set()
    end_at = time.time() + args.duration
    while time.time() < end_at:
        current: set[tuple[int, str]] = set()
        for actor in watched_vehicles(world, args.ego_id):
            loc = actor.get_location()
            for zone in zones:
                key = (actor.id, zone.name)
                if contains(zone, loc):
                    current.add(key)
                    if key not in active:
                        log.info("TRIGGER ENTER actor=%d zone=%s kind=%s", actor.id, zone.name, zone.kind)
                elif key in active:
                    log.info("TRIGGER EXIT  actor=%d zone=%s kind=%s", actor.id, zone.name, zone.kind)
        active = current
        time.sleep(args.tick)


def main() -> int:
    args = parse_args()
    cfg = load_config("sim")["client"]
    client = carla.Client(args.host or cfg["host"], args.port or cfg["port"])
    client.set_timeout(cfg["timeout"])
    world = client.get_world()
    spec = TrackSpec()
    zones = zone_specs(spec)

    log.info("current map: %s", world.get_map().name)
    log.info("zone count: %d", len(zones))
    for zone in zones:
        log.info("zone: %-14s kind=%-7s s=%7.2fm xy=(%8.2f,%8.2f) size=(%.2f x %.2f)",
                 zone.name, zone.kind, zone.pose.s, zone.pose.x, zone.pose.y, zone.length, zone.width)

    actors = spawn_zone_markers(world, zones, args)
    if args.spawn_obstacles:
        actors.extend(spawn_obstacles(world, spec, args))
    test_vehicle = spawn_test_vehicle(world, zones, args)
    if test_vehicle is not None:
        actors.append(test_vehicle)

    try:
        if args.no_monitor:
            time.sleep(args.duration)
        else:
            monitor_triggers(world, zones, args)
    finally:
        if actors and not args.keep:
            log.info("cleaning up spawned actors: %d", len(actors))
            for actor in actors:
                try:
                    if actor.is_alive:
                        actor.destroy()
                except RuntimeError:
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
