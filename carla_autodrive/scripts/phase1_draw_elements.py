#!/usr/bin/env python
"""Phase 1 helper: draw mission elements and optionally spawn obstacle vehicles.

Assumes the SKKU_AD_Track OpenDRIVE world is already loaded, for example:
    python -m carla_autodrive.scripts.build_track --load
    python -m carla_autodrive.scripts.runtime_mesh_markings
    python -m carla_autodrive.scripts.phase1_draw_elements --duration 60
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import carla

from carla_autodrive.maps import TrackPose, TrackSpec
from carla_autodrive.missions import mission_elements, parking_zone_pose
from carla_autodrive.utils import get_logger, load_config

log = get_logger()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 1: visualize track mission elements and optionally spawn obstacles")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--duration", type=float, default=60.0, help="Display/spawn duration in seconds.")
    ap.add_argument("--spawn-obstacles", action="store_true", help="Spawn three obstacle vehicles as real actors.")
    ap.add_argument("--obstacle2", type=int, default=0, choices=(0, 1, 2), help="Obstacle 2 preset index.")
    ap.add_argument("--obstacle3", type=int, default=0, choices=(0, 1, 2), help="Obstacle 3 preset index.")
    ap.add_argument("--obstacle-blueprint", default="vehicle.tesla.model3")
    ap.add_argument("--keep-obstacles", action="store_true", help="Keep obstacle actors on exit.")
    return ap.parse_args()


def color(name: str) -> carla.Color:
    colors = {
        "white": carla.Color(255, 255, 255),
        "yellow": carla.Color(255, 220, 0),
        "green": carla.Color(0, 220, 80),
        "red": carla.Color(255, 40, 40),
        "blue": carla.Color(40, 140, 255),
        "cyan": carla.Color(0, 220, 220),
        "magenta": carla.Color(255, 80, 255),
    }
    return colors[name]


def transform_from_pose(pose: TrackPose, z: float = 0.15) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=pose.x, y=-pose.y, z=z),
        carla.Rotation(yaw=math.degrees(-pose.heading)),
    )


def draw_box(world: carla.World, pose: TrackPose, label: str, length: float,
             width: float, height: float, box_color: carla.Color,
             life_time: float) -> None:
    tf = transform_from_pose(pose, z=height / 2.0)
    box = carla.BoundingBox(tf.location, carla.Vector3D(length / 2.0, width / 2.0, height / 2.0))
    world.debug.draw_box(box, tf.rotation, thickness=0.06, color=box_color, life_time=life_time)
    label_loc = carla.Location(x=pose.x, y=-pose.y, z=height + 0.8)
    world.debug.draw_string(label_loc, label, draw_shadow=False, color=box_color, life_time=life_time)


def draw_elements(world: carla.World, spec: TrackSpec, life_time: float) -> list[tuple[str, TrackPose]]:
    dims = spec.cfg["dimensions"]
    elements = mission_elements(spec)
    drawn: list[tuple[str, TrackPose]] = []

    start = elements["start_line"]
    start_pose = spec.road_center_pose(start["s"])
    draw_box(world, start_pose, "START / T1", spec.mm(dims["start_line_mm"]),
             spec.road_width, 0.12, color("white"), life_time)
    drawn.append(("start_line", start_pose))

    t4 = elements["laptimer_T4"]
    t4_pose = spec.road_center_pose(t4["s"])
    draw_box(world, t4_pose, "T4 LAP", spec.mm(dims["start_line_mm"]),
             spec.road_width, 0.12, color("cyan"), life_time)
    drawn.append(("laptimer_T4", t4_pose))

    crosswalk = elements["crosswalk"]
    crosswalk_pose = spec.road_center_pose(crosswalk["s"])
    world.debug.draw_string(
        carla.Location(x=crosswalk_pose.x, y=-crosswalk_pose.y, z=0.8),
        "CROSSWALK / SIGNAL",
        draw_shadow=False,
        color=color("white"),
        life_time=life_time,
    )
    drawn.append(("crosswalk", crosswalk_pose))

    for key in ("parking_zone1", "parking_zone2"):
        item = elements[key]
        zone_idx = int(key.removeprefix("parking_zone"))
        slot = item.get("slot", {}) if isinstance(item, dict) else {}
        parking_length = spec.mm(slot.get("length_mm", dims["parking_mm"][1]))
        parking_width = spec.mm(slot.get("width_mm", dims["parking_mm"][0]))
        pose = parking_zone_pose(spec, zone_idx)
        draw_box(world, pose, key.upper(), parking_length, parking_width,
                 0.10, color("green"), life_time)
        drawn.append((key, pose))

    obstacle_items = [("obstacle_1", elements["obstacle_1"])]
    for idx, item in enumerate(elements["obstacle_2_presets"]):
        obstacle_items.append((f"obstacle_2_p{idx}", item))
    for idx, item in enumerate(elements["obstacle_3_presets"]):
        obstacle_items.append((f"obstacle_3_p{idx}", item))

    for label, item in obstacle_items:
        pose = spec.lane_center_pose(item["s"], item["lane"])
        draw_box(world, pose, label.upper(), 2.2, 1.0, 0.35, color("red"), life_time)
        drawn.append((label, pose))

    return drawn


def selected_obstacle_poses(spec: TrackSpec, obstacle2_idx: int, obstacle3_idx: int) -> list[tuple[str, TrackPose]]:
    elements = mission_elements(spec)
    selected = [
        ("obstacle_1", elements["obstacle_1"]),
        (f"obstacle_2_p{obstacle2_idx}", elements["obstacle_2_presets"][obstacle2_idx]),
        (f"obstacle_3_p{obstacle3_idx}", elements["obstacle_3_presets"][obstacle3_idx]),
    ]
    return [(label, spec.lane_center_pose(item["s"], item["lane"])) for label, item in selected]


def spawn_obstacles(world: carla.World, spec: TrackSpec, blueprint_id: str,
                    obstacle2_idx: int, obstacle3_idx: int) -> list[carla.Actor]:
    bp = world.get_blueprint_library().find(blueprint_id)
    actors: list[carla.Actor] = []
    for label, pose in selected_obstacle_poses(spec, obstacle2_idx, obstacle3_idx):
        tf = transform_from_pose(pose, z=0.35)
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            log.warning("obstacle spawn failed: %s @ s=%.2fm lane=%s", label, pose.s, pose.lane)
            continue
        try:
            actor.set_simulate_physics(False)
        except RuntimeError:
            pass
        actors.append(actor)
        log.info("obstacle spawned: %s -> %s id=%d @ (%.2f, %.2f)", label, blueprint_id,
                 actor.id, pose.x, pose.y)
    return actors


def main() -> int:
    args = parse_args()
    cfg = load_config("sim")["client"]
    host = args.host or cfg["host"]
    port = args.port or cfg["port"]

    client = carla.Client(host, port)
    client.set_timeout(cfg["timeout"])
    world = client.get_world()
    spec = TrackSpec()

    log.info("current map: %s", world.get_map().name)
    log.info("drawing track elements: duration=%.1fs, total_length=%.1fm", args.duration, spec.total_length())
    drawn = draw_elements(world, spec, args.duration)
    for label, pose in drawn:
        log.info("element position: %-16s s=%7.2fm lane=%s xy=(%8.2f,%8.2f) yaw=%6.1f",
                 label, pose.s, pose.lane or "road", pose.x, pose.y, math.degrees(pose.heading))

    actors: list[carla.Actor] = []
    if args.spawn_obstacles:
        actors = spawn_obstacles(world, spec, args.obstacle_blueprint, args.obstacle2, args.obstacle3)

    if args.duration > 0:
        time.sleep(args.duration)

    if actors and not args.keep_obstacles:
        log.info("cleaning up obstacle actors: %d", len(actors))
        for actor in actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
