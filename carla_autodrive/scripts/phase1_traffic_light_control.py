#!/usr/bin/env python
"""Phase 1 traffic light control for the custom OpenDRIVE track.

The custom map exposes a real ``traffic.traffic_light`` actor when the
OpenDRIVE ``<signal>`` entry is loaded by CARLA. This script can lock that
actor to one state or run a Red/Green/Yellow cycle.
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

from carla_autodrive.maps import TrackSpec
from carla_autodrive.missions import mission_elements
from carla_autodrive.utils import get_logger, load_config

log = get_logger()


STATE_BY_NAME = {
    "red": carla.TrafficLightState.Red,
    "yellow": carla.TrafficLightState.Yellow,
    "green": carla.TrafficLightState.Green,
    "off": carla.TrafficLightState.Off,
    "unknown": carla.TrafficLightState.Unknown,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 1: real traffic-light actor state control")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--traffic-light-id", type=int, default=None, help="traffic-light actor id to control")
    ap.add_argument("--nearest-crosswalk", action="store_true", default=True,
                    help="Select the traffic light closest to the crosswalk position. Default.")
    ap.add_argument("--state", choices=sorted(STATE_BY_NAME), default="red",
                    help="Fixed signal state when --cycle is not used.")
    ap.add_argument("--cycle", action="store_true", help="Red→Green→Yellow repeat")
    ap.add_argument("--duration", type=float, default=30.0, help="Control duration in seconds.")
    ap.add_argument("--red-time", type=float, default=5.0)
    ap.add_argument("--green-time", type=float, default=5.0)
    ap.add_argument("--yellow-time", type=float, default=2.0)
    ap.add_argument("--no-freeze", action="store_true", help="Do not freeze the traffic light.")
    ap.add_argument("--logical-fallback", action="store_true",
                    help="If no real actor exists, only print debug text and act as a logical signal.")
    return ap.parse_args()


def crosswalk_pose(spec: TrackSpec):
    return spec.road_center_pose(mission_elements(spec)["crosswalk"]["s"])


def distance_2d(actor: carla.Actor, x: float, y: float) -> float:
    loc = actor.get_transform().location
    return math.hypot(loc.x - x, loc.y - y)


def select_traffic_light(world: carla.World, args: argparse.Namespace, spec: TrackSpec):
    tls = list(world.get_actors().filter("traffic.traffic_light*"))
    log.info("traffic-light actor count: %d", len(tls))
    if not tls:
        return None

    if args.traffic_light_id is not None:
        for tl in tls:
            if tl.id == args.traffic_light_id:
                return tl
        raise ValueError(f"traffic light id={args.traffic_light_id}  was not found.")

    pose = crosswalk_pose(spec)
    return min(tls, key=lambda tl: distance_2d(tl, pose.x, pose.y))


def configure_durations(tl: carla.TrafficLight, args: argparse.Namespace) -> None:
    setters = (
        ("set_red_time", args.red_time),
        ("set_green_time", args.green_time),
        ("set_yellow_time", args.yellow_time),
    )
    for name, value in setters:
        fn = getattr(tl, name, None)
        if callable(fn):
            fn(value)


def set_state(tl: carla.TrafficLight, state_name: str) -> None:
    state = STATE_BY_NAME[state_name]
    tl.set_state(state)
    log.info("TRAFFIC_LIGHT id=%d state=%s", tl.id, state_name.upper())


def run_actor_control(tl: carla.TrafficLight, args: argparse.Namespace) -> None:
    configure_durations(tl, args)
    if not args.no_freeze:
        try:
            tl.freeze(True)
            log.info("traffic light freeze=ON")
        except RuntimeError as exc:
            log.warning("traffic-light freeze failed: %s", exc)

    end_at = time.time() + args.duration
    if not args.cycle:
        set_state(tl, args.state)
        while time.time() < end_at:
            time.sleep(0.2)
        return

    cycle = (
        ("red", args.red_time),
        ("green", args.green_time),
        ("yellow", args.yellow_time),
    )
    while time.time() < end_at:
        for state_name, hold in cycle:
            set_state(tl, state_name)
            until = min(end_at, time.time() + hold)
            while time.time() < until:
                time.sleep(0.2)


def run_logical_fallback(world: carla.World, args: argparse.Namespace, spec: TrackSpec) -> None:
    pose = crosswalk_pose(spec)
    loc = carla.Location(x=pose.x, y=pose.y, z=2.0)
    end_at = time.time() + args.duration
    log.warning("Current map has no real traffic.traffic_light actor. Falling back to the logical signal.")
    if not args.cycle:
        while time.time() < end_at:
            world.debug.draw_string(loc, f"LOGICAL SIGNAL: {args.state.upper()}",
                                    draw_shadow=False, color=carla.Color(255, 0, 0), life_time=0.5)
            log.info("LOGICAL_SIGNAL state=%s", args.state.upper())
            time.sleep(1.0)
        return

    cycle = (("red", args.red_time), ("green", args.green_time), ("yellow", args.yellow_time))
    colors = {
        "red": carla.Color(255, 0, 0),
        "green": carla.Color(0, 255, 0),
        "yellow": carla.Color(255, 220, 0),
    }
    while time.time() < end_at:
        for state_name, hold in cycle:
            log.info("LOGICAL_SIGNAL state=%s", state_name.upper())
            until = min(end_at, time.time() + hold)
            while time.time() < until:
                world.debug.draw_string(loc, f"LOGICAL SIGNAL: {state_name.upper()}",
                                        draw_shadow=False, color=colors[state_name], life_time=0.5)
                time.sleep(0.5)


def main() -> int:
    args = parse_args()
    cfg = load_config("sim")["client"]
    client = carla.Client(args.host or cfg["host"], args.port or cfg["port"])
    client.set_timeout(cfg["timeout"])
    world = client.get_world()
    spec = TrackSpec()
    log.info("current map: %s", world.get_map().name)

    tl = select_traffic_light(world, args, spec)
    if tl is None:
        if args.logical_fallback:
            run_logical_fallback(world, args, spec)
            return 0
        log.error("Current map has no real traffic.traffic_light actor.")
        log.error("Reload the custom OpenDRIVE map or use --logical-fallback.")
        return 1

    loc = tl.get_transform().location
    log.info("traffic light target: id=%d type=%s xyz=(%.2f, %.2f, %.2f)",
             tl.id, tl.type_id, loc.x, loc.y, loc.z)
    run_actor_control(tl, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
