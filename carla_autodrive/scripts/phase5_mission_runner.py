#!/usr/bin/env python
"""Phase 5 runner: mission FSM + existing route following controller."""
from __future__ import annotations

import argparse
import math
import select
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
from carla_autodrive.missions import (
    build_obstacle_avoidance_route,
    build_parking_maneuver_route,
    build_reverse_parking_maneuver_route,
    selected_parking_zone,
)
from carla_autodrive.perception import PerceptionPipeline
from carla_autodrive.scripts.phase4_control_demo import auto_load_track_map, spawn_preset_obstacles
from carla_autodrive.sensors import SensorStack
from carla_autodrive.simulator import CollisionMonitor, CompetitionScorer, TickRecord
from carla_autodrive.state_machine import MissionContext, MissionFSM, MissionFSMConfig, MissionMode
from carla_autodrive.utils import CarlaSession, get_logger, load_config

log = get_logger()


def parse_args(default_duration: float, default_dt: float) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: mission FSM runner")
    parser.add_argument("--host", default=None, help="CARLA server host")
    parser.add_argument("--port", type=int, default=None, help="CARLA server port")
    parser.add_argument("--map", default=None, help="Map to load before spawning")
    parser.add_argument("--spawn-index", type=int, default=None, help="Vehicle spawn point index")
    parser.add_argument("--mission", choices=[mode.value for mode in MissionMode], default=MissionMode.TIME_TRIAL.value)
    parser.add_argument("--duration", type=float, default=default_duration)
    parser.add_argument("--ticks", type=int, default=None, help="Run for N simulator ticks instead of wall-clock duration")
    parser.add_argument("--target-speed", type=float, default=2.0, help="Base target speed in m/s")
    parser.add_argument("--route-lane", type=int, default=2)
    parser.add_argument("--route-spacing-mm", type=float, default=100.0)
    parser.add_argument("--total-laps", type=int, default=2)
    parser.add_argument("--no-auto-load-track-map", action="store_true")
    parser.add_argument("--track-map-load-timeout", type=float, default=180.0)

    parser.add_argument("--obstacle2", type=int, default=0, choices=(0, 1, 2), help="Obstacle 2 preset index")
    parser.add_argument("--obstacle3", type=int, default=0, choices=(0, 1, 2), help="Obstacle 3 preset index")
    parser.add_argument("--avoid-hold-mm", type=float, default=1200.0)
    parser.add_argument("--avoid-transition-mm", type=float, default=1400.0)
    parser.add_argument("--spawn-preset-obstacles", action="store_true")
    parser.add_argument("--obstacle-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--green-after-sec", type=float, default=3.0)
    parser.add_argument("--manual-green", action="store_true", help="Press Enter after stopping at the red light")

    parser.add_argument("--parking-zone", type=int, default=2, choices=(1, 2))
    parser.set_defaults(forward_parking=False)
    parser.add_argument("--reverse-parking", dest="forward_parking", action="store_false",
                        help="Use forward staging plus reverse pull-in parking route")
    parser.add_argument("--forward-parking", dest="forward_parking", action="store_true",
                        help="Use the older forward pull-in parking route")
    parser.add_argument("--parking-approach-mm", type=float, default=3200.0)
    parser.add_argument("--parking-transition-mm", type=float, default=2600.0)
    parser.add_argument("--parking-overshoot-mm", type=float, default=0.0)
    parser.add_argument("--parking-staging-after-mm", type=float, default=1800.0)
    parser.add_argument("--parking-reverse-transition-mm", type=float, default=2200.0)
    parser.add_argument("--parking-reverse-speed", type=float, default=0.35)
    parser.add_argument("--parking-hold-sec", type=float, default=3.2)
    parser.add_argument("--parking-reverse-lookahead", type=float, default=0.6)
    parser.add_argument("--parking-reverse-steer-scale", type=float, default=0.55)
    parser.add_argument("--parking-reverse-min-throttle", type=float, default=0.16)
    parser.add_argument("--parking-reverse-end-min-speed", type=float, default=0.12)
    parser.add_argument("--parking-finish-distance", type=float, default=0.8)

    parser.add_argument("--no-curve-speed", action="store_true")
    parser.add_argument("--curve-min-speed", type=float, default=1.2)
    parser.add_argument("--curve-max-lat-acc", type=float, default=0.45)
    parser.add_argument("--curve-lookahead", type=float, default=8.0)
    parser.add_argument("--steer-speed-gain", type=float, default=2.2)
    parser.add_argument("--max-throttle", type=float, default=0.45)
    parser.add_argument("--max-brake", type=float, default=0.75)
    parser.add_argument("--brake-overspeed-margin", type=float, default=0.35)
    parser.add_argument("--obstacle-min-x", type=float, default=1.0)
    parser.add_argument("--obstacle-stop-distance", type=float, default=1.0)
    parser.add_argument("--obstacle-slow-distance", type=float, default=2.0)

    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--no-perception", action="store_true")
    parser.add_argument("--dt", type=float, default=default_dt)
    parser.add_argument("--no-collision-sensor", action="store_true")
    parser.add_argument("--cte-warning", type=float, default=0.75)
    parser.add_argument("--lane-intrusion-cte", type=float, default=0.45)
    parser.add_argument("--lane-departure-cte", type=float, default=0.85)
    parser.set_defaults(lane_corridor_scoring=True)
    parser.add_argument("--lane-corridor-scoring", dest="lane_corridor_scoring", action="store_true",
                        help="Score lane events against the current lane corridor instead of raw route CTE")
    parser.add_argument("--no-lane-corridor-scoring", dest="lane_corridor_scoring", action="store_false",
                        help="Use the legacy raw route-CTE thresholds for lane events")
    parser.add_argument("--lane-boundary-margin", type=float, default=0.0,
                        help="Extra safety margin, in meters, subtracted from the lane-corridor threshold")
    parser.add_argument("--stop-violation-speed", type=float, default=0.35)
    parser.add_argument("--report-path", default=None, help="Write detailed run report JSON")
    parser.add_argument("--csv-path", default=None, help="Write per-tick telemetry CSV")
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


def build_route(spec: TrackSpec, args: argparse.Namespace):
    mission = MissionMode(args.mission)
    if mission == MissionMode.TIME_TRIAL:
        route = build_track_lane_route(spec, lane=args.route_lane, spacing_mm=args.route_spacing_mm)
        return route, True, False, "time trial lane route"

    if mission == MissionMode.OBSTACLE_SIGNAL:
        route = build_obstacle_avoidance_route(
            spec,
            drive_lane=args.route_lane,
            obstacle2_idx=args.obstacle2,
            obstacle3_idx=args.obstacle3,
            spacing_mm=args.route_spacing_mm,
            hold_mm=args.avoid_hold_mm,
            transition_mm=args.avoid_transition_mm,
        )
        return route, True, False, "obstacle/signal preset route"

    zone = selected_parking_zone(spec, args.parking_zone)
    if args.forward_parking:
        route = build_parking_maneuver_route(
            spec,
            zone_idx=args.parking_zone,
            drive_lane=args.route_lane,
            spacing_mm=args.route_spacing_mm,
            approach_mm=args.parking_approach_mm,
            transition_mm=args.parking_transition_mm,
            overshoot_mm=args.parking_overshoot_mm,
        )
        label = f"forward parking route {zone.label}"
    else:
        route = build_reverse_parking_maneuver_route(
            spec,
            zone_idx=args.parking_zone,
            drive_lane=args.route_lane,
            spacing_mm=args.route_spacing_mm,
            staging_after_mm=args.parking_staging_after_mm,
            reverse_transition_mm=args.parking_reverse_transition_mm,
            reverse_speed_mps=args.parking_reverse_speed,
        )
        label = f"reverse parking route {zone.label}"
    return route, False, True, label


def lane_event_flags(spec: TrackSpec, args: argparse.Namespace, decision, abs_cte: float) -> tuple[bool, bool]:
    """Return lane intrusion/departure flags for the current tick.

    For lane-following runs, the route is the virtual line inside the selected
    lane. Raw route CTE is still useful for tuning, but lane penalties should
    only start once the vehicle leaves the lane corridor. During the obstacle
    mission's avoidance state, lane penalties are intentionally suppressed
    because the rules allow using the adjacent lane in that section.
    """
    if MissionMode(args.mission) == MissionMode.OBSTACLE_SIGNAL and decision.state.value == "OBSTACLE_AVOID":
        return False, False

    if not args.lane_corridor_scoring:
        return abs_cte >= args.lane_intrusion_cte, abs_cte >= args.lane_departure_cte

    half_lane = spec.lane_width / 2.0
    marking = spec.lane_mark_width
    margin = max(0.0, float(args.lane_boundary_margin))
    intrusion_threshold = max(0.0, half_lane - marking - margin)
    departure_threshold = max(intrusion_threshold, half_lane - margin)
    return abs_cte >= intrusion_threshold, abs_cte >= departure_threshold


def build_route_follower(route, closed_route: bool, stop_at_end: bool, args: argparse.Namespace):
    return RoutePurePursuitController(
        route,
        closed_route=closed_route,
        curve_speed_enabled=not args.no_curve_speed,
        curve_speed_cap_mps=args.target_speed,
        curve_speed_min_mps=args.curve_min_speed,
        curve_speed_max_lat_acc_mps2=args.curve_max_lat_acc,
        curve_speed_lookahead_m=args.curve_lookahead,
        steer_speed_gain_mps=args.steer_speed_gain,
        stop_at_end=stop_at_end,
        finish_distance_m=args.parking_finish_distance,
        reverse_lookahead_m=args.parking_reverse_lookahead,
        reverse_steer_scale=args.parking_reverse_steer_scale,
        reverse_end_min_speed_mps=args.parking_reverse_end_min_speed,
    )


def manual_green_pressed(enabled: bool) -> bool:
    if not enabled:
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return False
    sys.stdin.readline()
    return True


def vehicle_speed(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return float(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))


def main() -> int:
    cfg = load_config("sim")
    args = parse_args(cfg["phase0"]["duration"], cfg["world"].get("fixed_delta_seconds", 0.05))
    apply_overrides(cfg, args)
    sensor_cfg = load_config("sensors")
    hardware_cfg = load_config("hardware_limits")
    spec = TrackSpec()
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

    if not args.no_auto_load_track_map:
        try:
            auto_load_track_map(cfg, args.track_map_load_timeout)
        except RuntimeError as exc:
            log.error("automatic track-route map load failed: %s", exc)
            log.error("Run `python -m carla_autodrive.scripts.build_track --load --timeout 180` first, then retry with --no-auto-load-track-map.")
            return 1

    route, closed_route, stop_at_end, route_label = build_route(spec, args)
    time_limit_s = 240.0 if MissionMode(args.mission) == MissionMode.TIME_TRIAL else 240.0
    route_follower = build_route_follower(route, closed_route, stop_at_end, args)
    mission_mode = MissionMode(args.mission)
    green_after_sec = float("inf") if args.manual_green else args.green_after_sec
    fsm = MissionFSM(
        spec,
        MissionFSMConfig(
            mode=mission_mode,
            target_speed_mps=args.target_speed,
            total_laps=args.total_laps,
            green_after_sec=green_after_sec,
            parking_hold_sec=args.parking_hold_sec,
        ),
    )

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
    scorer = CompetitionScorer(
        args.mission,
        time_limit_s=time_limit_s,
        cte_warning_m=args.cte_warning,
        lane_intrusion_cte_m=args.lane_intrusion_cte,
        lane_departure_cte_m=args.lane_departure_cte,
        stop_violation_speed_mps=args.stop_violation_speed,
    )

    try:
        with CarlaSession(cfg) as session:
            world = session.world
            if "OpenDriveMap" not in world.get_map().name:
                raise RuntimeError(f"Phase 5 track routing needs the custom OpenDRIVE map. Current map: {world.get_map().name}")

            vehicle = session.spawn_vehicle()
            if args.spawn_preset_obstacles:
                spawn_preset_obstacles(session, spec, args)

            collision_monitor = None
            if not args.no_collision_sensor:
                collision_monitor = CollisionMonitor()
                session.register(collision_monitor.spawn(world, vehicle))

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

            log.info(
                "Phase 5 mission runner started: mission=%s route=%s points=%d target=%.2fm/s",
                mission_mode.value,
                route_label,
                len(route),
                args.target_speed,
            )
            if args.manual_green:
                log.info("manual green enabled: press Enter after the vehicle stops at the red-light zone")

            start = time.time()
            tick_count = 0
            sim_elapsed = 0.0
            next_log = 0.0
            speeds: list[float] = []
            abs_ctes: list[float] = []
            abs_headings: list[float] = []
            distance_m = 0.0
            prev_location = vehicle.get_location()

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
                speed = vehicle_speed(vehicle)
                route_finished = route_follower.is_finished(vehicle.get_location(), speed)
                decision = fsm.update(
                    MissionContext(
                        tick=tick_count,
                        sim_time_s=sim_elapsed,
                        speed_mps=speed,
                        route_index=route_follower.current_index,
                        route_length=len(route),
                        route_s_m=route_follower.current_s_m,
                        route_finished=route_finished,
                        green_signal=manual_green_pressed(args.manual_green),
                    )
                )
                controller.cfg.target_speed_mps = decision.target_speed_mps
                if decision.finished:
                    scorer.finish(completed=True, reason=decision.reason, sim_time_s=sim_elapsed)
                    log.info("mission finished: tick=%d sim_t=%.1fs reason=%s", tick_count, sim_elapsed, decision.reason)
                    break

                perception = None
                if stack is not None and perception_pipeline is not None:
                    snapshot = stack.capture(vehicle, sim_frame=frame, timestamp=elapsed)
                    perception = perception_pipeline.process(snapshot)

                control, command = controller.run_route_step(vehicle, route_follower, perception, args.dt)
                vehicle.apply_control(control)
                speeds.append(command.current_speed_mps)
                abs_ctes.append(abs(command.cross_track_error_m))
                abs_headings.append(abs(command.heading_error_rad))
                abs_cte = abs(command.cross_track_error_m)
                collision = collision_monitor.snapshot() if collision_monitor is not None else None
                stop_required = bool(decision.force_stop or decision.state.value == "TRAFFIC_STOP")
                stop_violation = bool(stop_required and command.current_speed_mps > args.stop_violation_speed)
                lane_intrusion, lane_departure = lane_event_flags(spec, args, decision, abs_cte)
                scorer.add_tick(
                    TickRecord(
                        tick=tick_count,
                        sim_time_s=sim_elapsed,
                        state=decision.state.value,
                        decision_reason=decision.reason,
                        control_reason=command.reason,
                        speed_mps=command.current_speed_mps,
                        desired_speed_mps=command.desired_speed_mps,
                        target_speed_mps=command.target_speed_mps,
                        throttle=command.throttle,
                        brake=command.brake,
                        steer=command.steer,
                        reverse=command.reverse,
                        cross_track_error_m=command.cross_track_error_m,
                        heading_error_rad=command.heading_error_rad,
                        collision_count=collision.count if collision is not None else 0,
                        collision_impulse=collision.max_impulse if collision is not None else 0.0,
                        lane_intrusion=lane_intrusion,
                        lane_departure=lane_departure,
                        stop_required=stop_required,
                        stop_violation=stop_violation,
                        parking_hold=decision.reason == "parking_hold",
                        route_index=route_follower.current_index,
                        route_s_m=route_follower.current_s_m,
                    )
                )

                current_location = vehicle.get_location()
                step_distance = math.hypot(current_location.x - prev_location.x, current_location.y - prev_location.y)
                if step_distance <= 5.0:
                    distance_m += step_distance
                prev_location = current_location

                if sim_elapsed >= next_log:
                    perception_text = perception.summary() if perception is not None else "perception=off"
                    log.info(
                        "sim_t=%4.1fs tick=%d | %s decision=%s speed=%.2f | %s | %s",
                        sim_elapsed,
                        tick_count,
                        fsm.summary(),
                        decision.reason,
                        decision.target_speed_mps,
                        command.summary(),
                        perception_text,
                    )
                    next_log += cfg["phase0"]["log_interval"]

            if not scorer.completed:
                scorer.finish(completed=False, reason="tick_or_duration_limit", sim_time_s=sim_elapsed)
            scorer.set_distance(distance_m)
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            summary = scorer.summary()
            if args.report_path:
                scorer.write_json(args.report_path, metadata={
                    "route_label": route_label,
                    "route_points": len(route),
                    "target_speed_mps": args.target_speed,
                    "route_lane": args.route_lane,
                    "obstacle2": args.obstacle2,
                    "obstacle3": args.obstacle3,
                    "parking_zone": args.parking_zone,
                    "cte_warning_m": args.cte_warning,
                    "lane_intrusion_cte_m": args.lane_intrusion_cte,
                    "lane_departure_cte_m": args.lane_departure_cte,
                    "lane_corridor_scoring": args.lane_corridor_scoring,
                    "route_lane_width_m": spec.lane_width,
                    "route_lane_mark_width_m": spec.lane_mark_width,
                    "lane_boundary_margin_m": args.lane_boundary_margin,
                    "stop_violation_speed_mps": args.stop_violation_speed,
                })
                log.info("Phase 6 JSON report written: %s", args.report_path)
            if args.csv_path:
                scorer.write_csv(args.csv_path)
                log.info("Phase 6 CSV telemetry written: %s", args.csv_path)
            if speeds:
                log.info(
                    "Phase 5 metrics: sim_t=%.1fs distance=%.1fm avg_speed=%.2fm/s max_speed=%.2fm/s "
                    "mean_abs_cte=%.2fm max_abs_cte=%.2fm max_abs_heading=%.2frad states=%s reasons=%s",
                    sim_elapsed,
                    distance_m,
                    sum(speeds) / len(speeds),
                    max(speeds),
                    sum(abs_ctes) / len(abs_ctes),
                    max(abs_ctes),
                    max(abs_headings),
                    summary.states,
                    summary.reasons,
                )
                log.info("Phase 6 events=%s", summary.events)
            if summary.penalties:
                log.info("Phase 6 scorer penalties=%s score=%.2f", summary.penalties, summary.score)
            else:
                log.info("Phase 6 scorer penalties={} score=0.00")
            log.info("Phase 5 mission runner finished")
        return 0

    except KeyboardInterrupt:
        log.warning("interrupted by user")
        return 130
    except RuntimeError as exc:
        log.error("runtime error: %s", exc)
        log.error("Check that CARLA server is running and the custom OpenDRIVE map is loaded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
