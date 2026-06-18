#!/usr/bin/env python
"""Quick helper for checking only the CARLA server connection.

    python -m carla_autodrive.scripts.check_connection
If the server is available, print versions, map, and spawn-point count. Otherwise print a short hint.
"""
from __future__ import annotations

import sys

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import carla

from carla_autodrive.utils import get_logger, load_config

log = get_logger()


def main() -> int:
    cfg = load_config("sim")["client"]
    try:
        client = carla.Client(cfg["host"], cfg["port"])
        client.set_timeout(cfg["timeout"])
        world = client.get_world()
        cmap = world.get_map()
        log.info("connection OK")
        log.info("   server=%s  client=%s",
                 client.get_server_version(), client.get_client_version())
        log.info("   map=%s  spawn_points=%d",
                 cmap.name, len(cmap.get_spawn_points()))
        return 0
    except RuntimeError as e:
        log.error("connection failed: %s", e)
        log.error("   Start the CARLA server first: ./CarlaUE4.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
