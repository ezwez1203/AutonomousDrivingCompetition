#!/usr/bin/env python
"""Phase 2 demo: spawn the full sensor stack and optionally record snapshots."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import carla

from carla_autodrive.sensors import SensorStack, save_snapshot_npz
from carla_autodrive.utils import CarlaSession, get_logger, load_config

log = get_logger()


def parse_args(default_duration: float) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: camera/LiDAR/4-way radar stack + PerceptionInput"
    )
    parser.add_argument("--host", default=None, help="CARLA server host")
    parser.add_argument("--port", type=int, default=None, help="CARLA server port")
    parser.add_argument("--map", default=None, help="Map to load before spawning")
    parser.add_argument("--duration", type=float, default=default_duration)
    parser.add_argument("--throttle", type=float, default=None)
    parser.add_argument("--steer", type=float, default=None)
    parser.add_argument("--brake", type=float, default=None)
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-dir", default="carla_autodrive/reports/phase2_sensor_samples")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save every N simulator ticks")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.host is not None:
        cfg["client"]["host"] = args.host
    if args.port is not None:
        cfg["client"]["port"] = args.port
    if args.map is not None:
        cfg["world"]["map"] = args.map
    if args.autopilot:
        cfg["phase0"]["control_mode"] = "autopilot"
    for key in ("throttle", "steer", "brake"):
        value = getattr(args, key)
        if value is not None:
            cfg["phase0"][key] = value


def apply_control(session: CarlaSession, vehicle, cfg: dict) -> None:
    mode = cfg["phase0"]["control_mode"]
    if mode == "autopilot":
        tm = session.client.get_trafficmanager()
        if session.is_sync:
            tm.set_synchronous_mode(True)
        vehicle.set_autopilot(True, tm.get_port())
        log.info("control mode: autopilot")
        return

    control = carla.VehicleControl(
        throttle=float(cfg["phase0"]["throttle"]),
        steer=float(cfg["phase0"].get("steer", 0.0)),
        brake=float(cfg["phase0"].get("brake", 0.0)),
    )
    vehicle.apply_control(control)
    log.info(
        "control mode: throttle=%.2f steer=%.2f brake=%.2f",
        control.throttle,
        control.steer,
        control.brake,
    )


def main() -> int:
    cfg = load_config("sim")
    args = parse_args(cfg["phase0"]["duration"])
    apply_overrides(cfg, args)
    sensor_cfg = load_config("sensors")

    try:
        with CarlaSession(cfg) as session:
            world = session.world
            vehicle = session.spawn_vehicle()

            stack = SensorStack(
                sensor_cfg,
                enable_camera=not args.no_camera,
                enable_lidar=not args.no_lidar,
                enable_radar=not args.no_radar,
            )
            for actor in stack.spawn(world, vehicle):
                session.register(actor)

            apply_control(session, vehicle, cfg)

            save_dir = Path(args.save_dir)
            saved = 0
            start = time.time()
            tick_count = 0
            next_log = 0.0
            log.info("Phase 2 sensor stack started for %.1fs", args.duration)

            while time.time() - start < args.duration:
                session.tick()
                tick_count += 1
                elapsed = time.time() - start
                frame = world.get_snapshot().frame
                snapshot = stack.capture(vehicle, sim_frame=frame, timestamp=elapsed)

                if not args.no_save and tick_count % max(1, args.save_every) == 0:
                    out = save_snapshot_npz(
                        snapshot,
                        save_dir / f"snapshot_{snapshot.sim_frame:08d}.npz",
                    )
                    saved += 1
                    log.info("saved %s", out)

                if elapsed >= next_log:
                    log.info(
                        "t=%4.1fs | %s | %s",
                        elapsed,
                        snapshot.summary(),
                        " | ".join(stack.summaries()),
                    )
                    next_log += cfg["phase0"]["log_interval"]

            log.info("Phase 2 sensor stack finished, saved=%d", saved)
        return 0

    except KeyboardInterrupt:
        log.warning("interrupted by user")
        return 130
    except RuntimeError as exc:
        log.error("runtime error: %s", exc)
        log.error("Check that CARLA server is running.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
