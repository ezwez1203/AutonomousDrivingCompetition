#!/usr/bin/env python
"""Phase 0 demo: spawn vehicle, attach RGB/LiDAR/Radar, print sensor data.

Usage with the carla conda environment:
    # 1) start the CARLA server in a separate terminal first
    #    ./CarlaUE4.sh -quality-level=Low
    # 2) run the demo
    python -m carla_autodrive.scripts.phase0_spawn_sensors
    python -m carla_autodrive.scripts.phase0_spawn_sensors --map Town01 --duration 30 --autopilot
    python -m carla_autodrive.scripts.phase0_spawn_sensors --throttle 0.2 --steer 0.1 --brake 0.0

Validation items from the Phase 0 project plan:
    - carla.Client connection
    - vehicle spawn
    - RGB camera / LiDAR / Radar attachment and data reception
    - carla.VehicleControl manual control(throttle/steer/brake) behavior
"""
from __future__ import annotations

import argparse
import sys
import time

# Path adjustment for direct execution without python -m.
if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import carla

from carla_autodrive.sensors import RGBCamera, Lidar, Radar
from carla_autodrive.utils import CarlaSession, get_logger, load_config

log = get_logger()


def parse_args(p0_cfg: dict) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 0: vehicle spawn + sensor data output")
    ap.add_argument("--host", default=None, help="CARLA server host")
    ap.add_argument("--port", type=int, default=None, help="CARLA server port")
    ap.add_argument("--map", default=None, help="map to load (for example: Town01)")
    ap.add_argument("--duration", type=float, default=p0_cfg["duration"],
                    help="Demo duration in seconds.")
    ap.add_argument("--autopilot", action="store_true",
                    help="Traffic Manager autopilot. Default is fixed throttle.")
    ap.add_argument("--throttle", type=float, default=None,
                    help="manual throttle override (0.0~1.0)")
    ap.add_argument("--steer", type=float, default=None,
                    help="manual steer override (-1.0~1.0)")
    ap.add_argument("--brake", type=float, default=None,
                    help="manual brake override (0.0~1.0)")
    ap.add_argument("--no-radar", action="store_true",
                    help="Skip attaching the radar sensor.")
    return ap.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.host is not None:
        cfg["client"]["host"] = args.host
    if args.port is not None:
        cfg["client"]["port"] = args.port
    if args.map is not None:
        cfg["world"]["map"] = args.map
    if args.autopilot:
        cfg["phase0"]["control_mode"] = "autopilot"
    if args.throttle is not None:
        cfg["phase0"]["throttle"] = args.throttle
    if args.steer is not None:
        cfg["phase0"]["steer"] = args.steer
    if args.brake is not None:
        cfg["phase0"]["brake"] = args.brake
    if args.no_radar:
        cfg["phase0"]["attach_radar"] = False


def radar_front_config(radar_cfg: dict) -> dict:
    """Build a single front radar config for the Phase 0 smoke test."""
    cfg = dict(radar_cfg)
    variants = cfg.pop("position_variants", None)
    if variants:
        first = variants[0]
        if isinstance(first, dict):
            cfg["position"] = first["position"]
            cfg["rotation"] = first.get("rotation", [0.0, 0.0, 0.0])
        else:
            cfg["position"] = first
            cfg["rotation"] = [0.0, 0.0, 0.0]
    cfg.setdefault("position", [0.4, 0.0, 0.2])
    cfg.setdefault("rotation", [0.0, 0.0, 0.0])
    return cfg


def main() -> int:
    cfg = load_config("sim")
    args = parse_args(cfg["phase0"])
    apply_cli_overrides(cfg, args)

    sensor_cfg = load_config("sensors")
    p0 = cfg["phase0"]

    try:
        with CarlaSession(cfg) as session:
            world = session.world

            # 1) vehicle spawn
            vehicle = session.spawn_vehicle()

            # 2) attach sensors
            camera = RGBCamera(sensor_cfg["camera"])
            lidar = Lidar(sensor_cfg["lidar"])
            radar = None
            session.register(camera.spawn(world, vehicle))
            session.register(lidar.spawn(world, vehicle))
            if p0.get("attach_radar", True):
                radar = Radar(radar_front_config(sensor_cfg["radar"]), name="radar_front")
                session.register(radar.spawn(world, vehicle))

            # 3) control mode setup (manual control = carla.VehicleControl validation)
            mode = p0["control_mode"]
            if mode == "autopilot":
                tm = session.client.get_trafficmanager()
                if session.is_sync:
                    tm.set_synchronous_mode(True)
                vehicle.set_autopilot(True, tm.get_port())
                log.info("control mode: autopilot (Traffic Manager)")
            else:
                control = carla.VehicleControl(
                    throttle=float(p0["throttle"]),
                    steer=float(p0.get("steer", 0.0)),
                    brake=float(p0.get("brake", 0.0)),
                )
                vehicle.apply_control(control)
                log.info("control mode: throttle=%.2f steer=%.2f brake=%.2f (manual control)",
                         control.throttle, control.steer, control.brake)

            # 4) main loop: print sensor data summaries
            log.info("demo started: %.0f seconds of sensor output (Ctrl+C to stop)",
                     args.duration)
            start = time.time()
            next_log = 0.0
            while time.time() - start < args.duration:
                session.tick()
                elapsed = time.time() - start
                if elapsed >= next_log:
                    v = vehicle.get_velocity()
                    speed_kmh = 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5
                    summaries = [camera.summary(), lidar.summary()]
                    if radar is not None:
                        summaries.append(radar.summary())
                    log.info("t=%4.1fs | speed=%5.1f km/h | %s",
                             elapsed, speed_kmh, " | ".join(summaries))
                    next_log += p0["log_interval"]

            log.info("demo finished normally")
        return 0

    except KeyboardInterrupt:
        log.warning("interrupted by user (Ctrl+C)")
        return 130
    except RuntimeError as e:
        log.error("run error: %s", e)
        log.error("Check that the CARLA server is running (./CarlaUE4.sh).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
