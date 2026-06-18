#!/usr/bin/env python
"""Phase 3 synthetic dataset recorder for lane, traffic-light, and obstacle learning."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import carla

from carla_autodrive.control import (
    HardwareLimitConfig,
    HardwareLimiter,
    RoutePurePursuitController,
    VehicleController,
    VehicleControllerConfig,
    build_track_lane_route,
)
from carla_autodrive.maps import TrackSpec
from carla_autodrive.perception import PerceptionGroundTruthLabeler, append_label_jsonl
from carla_autodrive.scripts.phase4_control_demo import (
    auto_load_track_map,
    nearest_driving_waypoint,
    spawn_preset_obstacles,
)
from carla_autodrive.sensors import SensorStack, save_snapshot_npz
from carla_autodrive.utils import CarlaSession, get_logger, load_config

log = get_logger()

_TRAFFIC_STATES = {
    "red": carla.TrafficLightState.Red,
    "yellow": carla.TrafficLightState.Yellow,
    "green": carla.TrafficLightState.Green,
    "off": carla.TrafficLightState.Off,
}


def parse_args(default_duration: float, default_dt: float) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: record synthetic sensor snapshots and CARLA ground-truth labels"
    )
    parser.add_argument("--host", default=None, help="CARLA server host")
    parser.add_argument("--port", type=int, default=None, help="CARLA server port")
    parser.add_argument("--map", default=None, help="Map to load before spawning")
    parser.add_argument("--spawn-index", type=int, default=None)
    parser.add_argument("--duration", type=float, default=default_duration)
    parser.add_argument("--ticks", type=int, default=1200, help="Run for N simulator ticks")
    parser.add_argument("--dt", type=float, default=default_dt)
    parser.add_argument("--save-dir", default="carla_autodrive/datasets/phase3_synthetic")
    parser.add_argument("--save-every", type=int, default=5, help="Save every N simulator ticks")
    parser.add_argument("--warmup-ticks", type=int, default=10, help="Skip initial ticks before saving samples")
    parser.add_argument("--target-speed", type=float, default=2.0)
    parser.add_argument("--route-source", choices=("track", "map"), default="track")
    parser.add_argument("--route-lane", type=int, default=2)
    parser.add_argument("--route-spacing-mm", type=float, default=100.0)
    parser.add_argument("--no-auto-load-track-map", action="store_true")
    parser.add_argument("--track-map-load-timeout", type=float, default=180.0)
    parser.add_argument("--max-throttle", type=float, default=0.45)
    parser.add_argument("--max-brake", type=float, default=0.75)
    parser.add_argument("--brake-overspeed-margin", type=float, default=0.35)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--spawn-preset-obstacles", action="store_true")
    parser.add_argument("--obstacle2", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--obstacle3", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--obstacle-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--obstacle-actor-filter", default="vehicle.*")
    parser.add_argument("--max-obstacle-distance", type=float, default=8.0)
    parser.add_argument(
        "--traffic-cycle",
        default="red,yellow,green",
        help="Comma-separated traffic light states to cycle. Empty string disables control.",
    )
    parser.add_argument("--traffic-cycle-ticks", type=int, default=120)
    parser.add_argument("--no-freeze-traffic-light", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.host is not None:
        cfg["client"]["host"] = args.host
    if args.port is not None:
        cfg["client"]["port"] = args.port
    if args.map is not None:
        cfg["world"]["map"] = args.map
    if args.spawn_index is not None:
        cfg["vehicle"]["spawn_index"] = args.spawn_index


def select_nearest_traffic_light(world: carla.World, vehicle: carla.Vehicle):
    vehicle_location = vehicle.get_location()
    nearest = None
    nearest_distance = float("inf")
    for actor in world.get_actors().filter("traffic.traffic_light*"):
        distance = actor.get_location().distance(vehicle_location)
        if distance < nearest_distance:
            nearest = actor
            nearest_distance = distance
    return nearest


def parse_traffic_cycle(raw: str) -> list[str]:
    states = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [state for state in states if state not in _TRAFFIC_STATES]
    if unknown:
        raise ValueError(f"unknown traffic states: {unknown}")
    return states


def maybe_set_traffic_state(tl, states: list[str], tick_count: int, cycle_ticks: int) -> str | None:
    if tl is None or not states:
        return None
    index = (tick_count // max(1, cycle_ticks)) % len(states)
    state_name = states[index]
    tl.set_state(_TRAFFIC_STATES[state_name])
    return state_name


def write_dataset_info(save_dir: Path, args: argparse.Namespace, sensor_cfg: dict) -> None:
    payload = {
        "script": "phase3_dataset_recorder",
        "args": vars(args),
        "sensor_config": sensor_cfg,
        "label_format": {
            "lane": "track-geometry center/heading error label",
            "traffic_light": "nearest CARLA traffic.traffic_light actor state",
            "obstacles": "CARLA actor local position and bounding-box dimensions",
        },
    }
    (save_dir / "dataset_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    cfg = load_config("sim")
    args = parse_args(cfg["phase0"]["duration"], cfg["world"].get("fixed_delta_seconds", 0.05))
    apply_overrides(cfg, args)
    sensor_cfg = load_config("sensors")
    hardware_limits = HardwareLimitConfig.from_dict(load_config("hardware_limits"))

    if args.route_source == "track" and not args.no_auto_load_track_map:
        try:
            auto_load_track_map(cfg, args.track_map_load_timeout)
        except RuntimeError as exc:
            log.error("track route map auto-load failed: %s", exc)
            log.error("Run build_track --load first or retry with --track-map-load-timeout 300.")
            return 1

    spec = TrackSpec() if args.route_source == "track" else None
    route_follower = None
    if spec is not None:
        route_follower = RoutePurePursuitController(
            build_track_lane_route(spec, lane=args.route_lane, spacing_mm=args.route_spacing_mm),
            curve_speed_enabled=True,
            curve_speed_cap_mps=args.target_speed,
        )

    controller = VehicleController(
        VehicleControllerConfig(
            target_speed_mps=args.target_speed,
            max_throttle=args.max_throttle,
            max_brake=args.max_brake,
            brake_overspeed_margin_mps=args.brake_overspeed_margin,
        ),
        hardware_limiter=HardwareLimiter(hardware_limits),
    )
    labeler = PerceptionGroundTruthLabeler(
        spec=spec,
        route_lane=args.route_lane,
        route_spacing_mm=args.route_spacing_mm,
        obstacle_actor_filter=args.obstacle_actor_filter,
        max_obstacle_distance_m=args.max_obstacle_distance,
    )
    traffic_states = parse_traffic_cycle(args.traffic_cycle)

    save_dir = Path(args.save_dir)
    label_path = save_dir / "labels.jsonl"
    save_dir.mkdir(parents=True, exist_ok=True)
    if label_path.exists():
        label_path.unlink()
    write_dataset_info(save_dir, args, sensor_cfg)

    try:
        with CarlaSession(cfg) as session:
            world = session.world
            if args.route_source == "track" and "OpenDriveMap" not in world.get_map().name:
                raise RuntimeError(f"--route-source track requires OpenDriveMap, current map={world.get_map().name}")
            vehicle = session.spawn_vehicle()
            if args.spawn_preset_obstacles:
                if spec is None:
                    raise RuntimeError("--spawn-preset-obstacles requires --route-source track")
                spawn_preset_obstacles(session, spec, args)

            stack = SensorStack(
                sensor_cfg,
                enable_camera=not args.no_camera,
                enable_lidar=not args.no_lidar,
                enable_radar=not args.no_radar,
            )
            for actor in stack.spawn(world, vehicle):
                session.register(actor)

            traffic_light = select_nearest_traffic_light(world, vehicle)
            if traffic_light is not None and not args.no_freeze_traffic_light:
                traffic_light.freeze(True)
                log.info("traffic light freeze=ON id=%d", traffic_light.id)

            start = time.time()
            tick_count = 0
            saved = 0
            next_log = 0.0
            log.info(
                "Phase 3 dataset recorder started: ticks=%s save_every=%d out=%s",
                args.ticks,
                args.save_every,
                save_dir,
            )

            while True:
                if args.ticks is not None and tick_count >= args.ticks:
                    break
                if args.ticks is None and time.time() - start >= args.duration:
                    break

                session.tick()
                tick_count += 1
                elapsed = time.time() - start
                sim_elapsed = tick_count * args.dt
                frame = world.get_snapshot().frame

                traffic_state = maybe_set_traffic_state(
                    traffic_light,
                    traffic_states,
                    tick_count,
                    args.traffic_cycle_ticks,
                )

                snapshot = stack.capture(vehicle, sim_frame=frame, timestamp=sim_elapsed)
                if tick_count > max(0, args.warmup_ticks) and tick_count % max(1, args.save_every) == 0:
                    snapshot_rel = Path("snapshots") / f"snapshot_{snapshot.sim_frame:08d}.npz"
                    snapshot_path = save_snapshot_npz(snapshot, save_dir / snapshot_rel)
                    label = labeler.label(world=world, vehicle=vehicle, snapshot=snapshot)
                    label["snapshot_path"] = str(snapshot_rel)
                    append_label_jsonl(label_path, label)
                    saved += 1
                    log.debug("saved %s", snapshot_path)

                if route_follower is not None:
                    control, command = controller.run_route_step(vehicle, route_follower, None, args.dt)
                else:
                    waypoint = nearest_driving_waypoint(world, vehicle)
                    control, command = controller.run_step(vehicle, waypoint, None, args.dt)
                vehicle.apply_control(control)

                if sim_elapsed >= next_log:
                    log.info(
                        "sim_t=%4.1fs tick=%d saved=%d speed=%.2fm/s reason=%s traffic=%s",
                        sim_elapsed,
                        tick_count,
                        saved,
                        command.current_speed_mps,
                        command.reason,
                        traffic_state,
                    )
                    next_log += cfg["phase0"]["log_interval"]

            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            log.info("Phase 3 dataset recorder finished: saved=%d labels=%s", saved, label_path)
        return 0

    except KeyboardInterrupt:
        log.warning("interrupted by user")
        return 130
    except (RuntimeError, ValueError) as exc:
        log.error("runtime error: %s", exc)
        log.error("Check that CARLA server is running and the current map/sensors are available.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
