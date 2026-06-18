#!/usr/bin/env python
"""Phase 1: generate the track OpenDRIVE file and optionally validate it in CARLA.

    # Generate .xodr and check loop closure. A server is not required.
    python -m carla_autodrive.scripts.build_track

    # Load into the CARLA server after generation and inspect topology/spawns/waypoints.
    python -m carla_autodrive.scripts.build_track --load
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from carla_autodrive.maps import TrackSpec, generate_xodr
from carla_autodrive.utils import get_logger, load_config

log = get_logger()
OUT_DIR = Path(__file__).resolve().parent.parent / "maps"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 1: generate and load the track OpenDRIVE")
    ap.add_argument("--load", action="store_true", help="Load into the CARLA server after generation.")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=120.0, help="CARLA OpenDRIVE load timeout in seconds.")
    return ap.parse_args()


def build() -> tuple[TrackSpec, str, Path]:
    spec = TrackSpec()
    err = spec.loop_closure_error()
    log.info("track '%s' | scale=×%g | road width=%.2fm lane width=%.2fm | total length=%.1fm",
             spec.name, spec.scale, spec.road_width, spec.lane_width, spec.total_length())
    if err > 0.5:
        log.warning("loop-closure error %.3fm: the centerline does not return to its start. "
                    "Check that the turn angles sum to roughly 360 degrees.", err)
    else:
        log.info("loop closure OK (error %.4fm)", err)

    xodr = generate_xodr(spec)
    out = OUT_DIR / f"{spec.name}.xodr"
    out.write_text(xodr, encoding="utf-8")
    log.info("OpenDRIVE written: %s (%d bytes)", out, len(xodr))
    return spec, xodr, out


def load_into_carla(xodr: str, host: str, port: int, timeout: float = 120.0) -> int:
    import carla
    log.info("CARLA connection: %s:%d", host, port)
    client = carla.Client(host, port)
    client.set_timeout(timeout)

    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,        # no track boundary wall, flat driving surface
        additional_width=0.6,   # extra width beside each lane
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
    log.info("generating OpenDRIVE world...")
    world = client.generate_opendrive_world(xodr, params)
    cmap = world.get_map()

    spawns = cmap.get_spawn_points()
    topo = cmap.get_topology()
    wps = cmap.generate_waypoints(2.0)
    log.info("✅ map load succeeded: %s", cmap.name)
    log.info("   spawn_points=%d  topology_edges=%d  waypoints=%d",
             len(spawns), len(topo), len(wps))
    if not spawns and not wps:
        log.error("No drivable lane was generated. Check lane/link settings.")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    _, xodr, _ = build()
    if not args.load:
        log.info("load validation is available with '--load' when a CARLA server is running.")
        return 0

    cfg = load_config("sim")["client"]
    host = args.host or cfg["host"]
    port = args.port or cfg["port"]
    try:
        return load_into_carla(xodr, host, port, args.timeout)
    except RuntimeError as e:
        log.error("load failed: %s", e)
        log.error("Check that the CARLA server is running.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
