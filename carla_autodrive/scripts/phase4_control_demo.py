#!/usr/bin/env python
"""Phase 4 demo: waypoint following with Phase 3 perception feedback."""
from __future__ import annotations

import argparse
import math
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
from carla_autodrive.maps import TrackSpec, generate_xodr
from carla_autodrive.missions import (
    build_obstacle_avoidance_route,
    build_parking_maneuver_route,
    build_reverse_parking_maneuver_route,
    selected_obstacle_presets,
    selected_parking_zone,
)
from carla_autodrive.perception import PerceptionPipeline
from carla_autodrive.sensors import SensorStack
from carla_autodrive.utils import CarlaSession, get_logger, load_config

log = get_logger()


def parse_args(default_duration: float, default_dt: float) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: Pure Pursuit + PID control demo")
    parser.add_argument("--host", default=None, help="CARLA server host")
    parser.add_argument("--port", type=int, default=None, help="CARLA server port")
    parser.add_argument("--map", default=None, help="Map to load before spawning")
    parser.add_argument("--spawn-index", type=int, default=None, help="Vehicle spawn point index")
    parser.add_argument("--duration", type=float, default=default_duration)
    parser.add_argument("--ticks", type=int, default=None, help="Run for N simulator ticks instead of wall-clock duration")
    parser.add_argument("--target-speed", type=float, default=3.0, help="Target speed in m/s")
    parser.add_argument("--route-source", choices=("map", "track"), default="map")
    parser.add_argument("--route-lane", type=int, default=2, help="track route lane when --route-source track")
    parser.add_argument("--route-spacing-mm", type=float, default=100.0)
    parser.add_argument("--avoid-obstacles", action="store_true",
                        help="Use preset lane-change route for the selected obstacle scenario")
    parser.add_argument("--obstacle2", type=int, default=0, choices=(0, 1, 2), help="Obstacle 2 preset index")
    parser.add_argument("--obstacle3", type=int, default=0, choices=(0, 1, 2), help="Obstacle 3 preset index")
    parser.add_argument("--avoid-hold-mm", type=float, default=1200.0)
    parser.add_argument("--avoid-transition-mm", type=float, default=1400.0)
    parser.add_argument("--spawn-preset-obstacles", action="store_true",
                        help="Spawn static obstacle actors for the selected preset scenario")
    parser.add_argument("--obstacle-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--parking-maneuver", action="store_true",
                        help="Use an open route that pulls into a selected parking zone and stops")
    parser.add_argument("--parking-zone", type=int, default=2, choices=(1, 2))
    parser.add_argument("--reverse-parking", action="store_true",
                        help="Use forward staging plus reverse pull-in parking maneuver")
    parser.add_argument("--parking-approach-mm", type=float, default=3200.0)
    parser.add_argument("--parking-transition-mm", type=float, default=2600.0)
    parser.add_argument("--parking-overshoot-mm", type=float, default=0.0)
    parser.add_argument("--parking-staging-after-mm", type=float, default=1800.0)
    parser.add_argument("--parking-reverse-transition-mm", type=float, default=2200.0)
    parser.add_argument("--parking-reverse-speed", type=float, default=0.35)
    parser.add_argument("--parking-reverse-lookahead", type=float, default=0.6)
    parser.add_argument("--parking-reverse-steer-scale", type=float, default=0.55)
    parser.add_argument("--parking-reverse-min-throttle", type=float, default=0.16)
    parser.add_argument("--parking-reverse-end-min-speed", type=float, default=0.12)
    parser.add_argument("--parking-finish-distance", type=float, default=0.8)
    parser.add_argument("--no-stop-at-route-end", action="store_true")
    parser.add_argument("--no-auto-load-track-map", action="store_true",
                        help="Do not auto-load config/track.yaml OpenDRIVE when --route-source track")
    parser.add_argument("--track-map-load-timeout", type=float, default=180.0,
                        help="Timeout seconds for auto-loading config/track.yaml OpenDRIVE")
    parser.add_argument("--no-curve-speed", action="store_true", help="Disable curvature/steer based route speed caps")
    parser.add_argument("--curve-min-speed", type=float, default=1.2, help="Minimum speed cap in sharp curve sections")
    parser.add_argument("--curve-max-lat-acc", type=float, default=0.45, help="Lateral acceleration limit for curve speed cap")
    parser.add_argument("--curve-lookahead", type=float, default=8.0, help="Meters ahead used for route curvature speed cap")
    parser.add_argument("--steer-speed-gain", type=float, default=2.2, help="Speed cap reduction per normalized steer excess")
    parser.add_argument("--max-throttle", type=float, default=0.45)
    parser.add_argument("--max-brake", type=float, default=0.75)
    parser.add_argument("--brake-overspeed-margin", type=float, default=0.35,
                        help="Apply brake when speed exceeds desired speed by this margin")
    parser.add_argument("--obstacle-min-x", type=float, default=1.0)
    parser.add_argument("--obstacle-stop-distance", type=float, default=1.0)
    parser.add_argument("--obstacle-slow-distance", type=float, default=2.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--no-perception", action="store_true")
    parser.add_argument("--dt", type=float, default=default_dt, help="Controller dt in seconds")
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


def nearest_driving_waypoint(world, vehicle) -> carla.Waypoint:
    waypoint = world.get_map().get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise RuntimeError("No driving waypoint was found near the current vehicle position.")
    return waypoint


def auto_load_track_map(cfg: dict, timeout: float) -> None:
    """Load the custom OpenDRIVE map before using TrackSpec route coordinates."""
    client_cfg = cfg["client"]
    spec = TrackSpec()
    xodr = generate_xodr(spec)
    out = Path(__file__).resolve().parent.parent / "maps" / f"{spec.name}.xodr"
    out.write_text(xodr, encoding="utf-8")

    log.info(
        "track route requested: auto-loading config/track.yaml OpenDRIVE (%s, %.1fm)",
        spec.name,
        spec.total_length(),
    )

    client = carla.Client(client_cfg["host"], client_cfg["port"])
    client.set_timeout(max(float(client_cfg.get("timeout", 20.0)), timeout))
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
    world = client.generate_opendrive_world(xodr, params)
    log.info(
        "track route map loaded: %s waypoints=%d",
        world.get_map().name,
        len(world.get_map().generate_waypoints(2.0)),
    )


def track_pose_to_carla_transform(pose, z: float = 0.35) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=pose.x, y=-pose.y, z=z),
        carla.Rotation(yaw=math.degrees(-pose.heading)),
    )


def spawn_preset_obstacles(session: CarlaSession, spec: TrackSpec, args: argparse.Namespace) -> None:
    world = session.world
    bp = world.get_blueprint_library().find(args.obstacle_blueprint)
    for obstacle in selected_obstacle_presets(spec, obstacle2_idx=args.obstacle2, obstacle3_idx=args.obstacle3):
        pose = spec.lane_center_pose(obstacle.s_mm, obstacle.lane)
        actor = world.try_spawn_actor(bp, track_pose_to_carla_transform(pose))
        if actor is None:
            log.warning("preset obstacle spawn failed: %s s=%.0f lane=%d", obstacle.label, obstacle.s_mm, obstacle.lane)
            continue
        session.register(actor)
        try:
            actor.set_simulate_physics(False)
        except RuntimeError:
            pass
        log.info("preset obstacle spawned: %s id=%d s=%.0f lane=%d", obstacle.label, actor.id, obstacle.s_mm, obstacle.lane)


def main() -> int:
    cfg = load_config("sim")
    args = parse_args(cfg["phase0"]["duration"], cfg["world"].get("fixed_delta_seconds", 0.05))
    apply_overrides(cfg, args)
    sensor_cfg = load_config("sensors")
    hardware_cfg = load_config("hardware_limits")
    hardware_limits = HardwareLimitConfig.from_dict(hardware_cfg)
    if hardware_limits.allow_voltage_boost:
        log.warning("hardware_limits allows voltage boost; contest rules require SMPS 12.0V without boost")
    log.info(
        "hardware limits: smps=%.1fV throttle<=%.2f reverse<=%.2f brake<=%.2f accel_delta<=%.2f/s pwm=%d..%d",
        hardware_limits.smps_voltage_v,
        hardware_limits.max_throttle_cmd,
        hardware_limits.max_reverse_cmd,
        hardware_limits.max_brake_cmd,
        hardware_limits.max_accel_delta_per_sec,
        hardware_limits.pwm_min,
        hardware_limits.pwm_max,
    )

    if args.route_source == "track" and not args.no_auto_load_track_map:
        try:
            auto_load_track_map(cfg, args.track_map_load_timeout)
        except RuntimeError as exc:
            log.error("automatic track-route map load failed: %s", exc)
            log.error("Restart the CARLA server, or first run `python -m carla_autodrive.scripts.build_track --load --timeout 180`.")
            return 1

    controller = VehicleController(
        VehicleControllerConfig(
            target_speed_mps=args.target_speed,
            max_throttle=args.max_throttle,
            max_brake=args.max_brake,
            brake_overspeed_margin_mps=args.brake_overspeed_margin,
            obstacle_min_x_m=args.obstacle_min_x,
            obstacle_stop_distance_m=args.obstacle_stop_distance,
            obstacle_slow_distance_m=args.obstacle_slow_distance,
            reverse_min_throttle=args.parking_reverse_min_throttle,
        ),
        hardware_limiter=HardwareLimiter(hardware_limits),
    )
    perception_pipeline = None if args.no_perception else PerceptionPipeline()
    route_follower = None
    spec = TrackSpec() if args.route_source == "track" else None
    stop_at_route_end = False
    if args.route_source == "track":
        if args.parking_maneuver:
            zone = selected_parking_zone(spec, args.parking_zone)
            if args.reverse_parking:
                route = build_reverse_parking_maneuver_route(
                    spec,
                    zone_idx=args.parking_zone,
                    drive_lane=args.route_lane,
                    spacing_mm=args.route_spacing_mm,
                    staging_after_mm=args.parking_staging_after_mm,
                    reverse_transition_mm=args.parking_reverse_transition_mm,
                    reverse_speed_mps=args.parking_reverse_speed,
                )
            else:
                route = build_parking_maneuver_route(
                    spec,
                    zone_idx=args.parking_zone,
                    drive_lane=args.route_lane,
                    spacing_mm=args.route_spacing_mm,
                    approach_mm=args.parking_approach_mm,
                    transition_mm=args.parking_transition_mm,
                    overshoot_mm=args.parking_overshoot_mm,
                )
            stop_at_route_end = not args.no_stop_at_route_end
            log.info(
                "parking maneuver route: mode=%s zone=%s s=%.0f lane=%d drive_lane=%d points=%d stop_at_end=%s",
                "reverse" if args.reverse_parking else "forward",
                zone.label,
                zone.s_mm,
                zone.lane,
                args.route_lane,
                len(route),
                stop_at_route_end,
            )
        elif args.avoid_obstacles:
            route = build_obstacle_avoidance_route(
                spec,
                drive_lane=args.route_lane,
                obstacle2_idx=args.obstacle2,
                obstacle3_idx=args.obstacle3,
                spacing_mm=args.route_spacing_mm,
                hold_mm=args.avoid_hold_mm,
                transition_mm=args.avoid_transition_mm,
            )
            log.info(
                "obstacle avoidance route: lane=%d obstacle2=%d obstacle3=%d points=%d",
                args.route_lane,
                args.obstacle2,
                args.obstacle3,
                len(route),
            )
        else:
            route = build_track_lane_route(spec, lane=args.route_lane, spacing_mm=args.route_spacing_mm)
        route_follower = RoutePurePursuitController(
            route,
            closed_route=not args.parking_maneuver,
            curve_speed_enabled=not args.no_curve_speed,
            curve_speed_cap_mps=args.target_speed,
            curve_speed_min_mps=args.curve_min_speed,
            curve_speed_max_lat_acc_mps2=args.curve_max_lat_acc,
            curve_speed_lookahead_m=args.curve_lookahead,
            steer_speed_gain_mps=args.steer_speed_gain,
            stop_at_end=stop_at_route_end,
            finish_distance_m=args.parking_finish_distance,
            reverse_lookahead_m=args.parking_reverse_lookahead,
            reverse_steer_scale=args.parking_reverse_steer_scale,
            reverse_end_min_speed_mps=args.parking_reverse_end_min_speed,
        )

    try:
        with CarlaSession(cfg) as session:
            world = session.world
            if args.route_source == "track" and "OpenDriveMap" not in world.get_map().name:
                raise RuntimeError(
                    f"--route-source track needs the custom OpenDRIVE map. "
                    f"current map: {world.get_map().name}"
                )
            vehicle = session.spawn_vehicle()
            if args.spawn_preset_obstacles:
                if spec is None:
                    raise RuntimeError("--spawn-preset-obstacles requires --route-source track")
                spawn_preset_obstacles(session, spec, args)

            stack = None
            if not args.no_perception:
                stack = SensorStack(
                    sensor_cfg,
                    enable_camera=not args.no_camera,
                    enable_lidar=not args.no_lidar,
                    enable_radar=not args.no_radar,
                )
                for actor in stack.spawn(world, vehicle):
                    session.register(actor)

            start = time.time()
            next_log = 0.0
            tick_count = 0
            sim_elapsed = 0.0
            speeds: list[float] = []
            abs_ctes: list[float] = []
            abs_headings: list[float] = []
            reasons: dict[str, int] = {}
            distance_m = 0.0
            prev_location = vehicle.get_location()
            log.info(
                "Phase 4 control demo started for %s target_speed=%.2fm/s",
                f"{args.ticks} ticks" if args.ticks is not None else f"{args.duration:.1f}s",
                args.target_speed,
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

                perception = None
                if stack is not None and perception_pipeline is not None:
                    snapshot = stack.capture(vehicle, sim_frame=frame, timestamp=elapsed)
                    perception = perception_pipeline.process(snapshot)

                if route_follower is not None:
                    control, command = controller.run_route_step(vehicle, route_follower, perception, args.dt)
                else:
                    waypoint = nearest_driving_waypoint(world, vehicle)
                    control, command = controller.run_step(vehicle, waypoint, perception, args.dt)
                vehicle.apply_control(control)
                speeds.append(command.current_speed_mps)
                abs_ctes.append(abs(command.cross_track_error_m))
                abs_headings.append(abs(command.heading_error_rad))
                reasons[command.reason] = reasons.get(command.reason, 0) + 1

                current_location = vehicle.get_location()
                distance_m += math.hypot(
                    current_location.x - prev_location.x,
                    current_location.y - prev_location.y,
                )
                prev_location = current_location

                if sim_elapsed >= next_log:
                    perception_text = perception.summary() if perception is not None else "perception=off"
                    log.info(
                        "sim_t=%4.1fs wall_t=%4.1fs tick=%d | %s | %s",
                        sim_elapsed,
                        elapsed,
                        tick_count,
                        command.summary(),
                        perception_text,
                    )
                    next_log += cfg["phase0"]["log_interval"]

                if route_follower is not None and route_follower.is_finished(vehicle.get_location(), command.current_speed_mps):
                    log.info("route endpoint reached: tick=%d sim_t=%.1fs", tick_count, sim_elapsed)
                    break

            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            if speeds:
                log.info(
                    "Phase 4 metrics: sim_t=%.1fs distance=%.1fm avg_speed=%.2fm/s "
                    "max_speed=%.2fm/s mean_abs_cte=%.2fm max_abs_cte=%.2fm "
                    "max_abs_heading=%.2frad reasons=%s",
                    sim_elapsed,
                    distance_m,
                    sum(speeds) / len(speeds),
                    max(speeds),
                    sum(abs_ctes) / len(abs_ctes),
                    max(abs_ctes),
                    max(abs_headings),
                    reasons,
                )
            log.info("Phase 4 control demo finished")
        return 0

    except KeyboardInterrupt:
        log.warning("interrupted by user")
        return 130
    except RuntimeError as exc:
        log.error("runtime error: %s", exc)
        log.error("Check that CARLA server is running and the current map has driving waypoints.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
