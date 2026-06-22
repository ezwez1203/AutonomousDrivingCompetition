"""CARLA server connection and actor lifecycle management.

with CarlaSession(cfg) as session:
    vehicle = session.spawn_vehicle(...)
    ...
When the session ends, all spawned actors are cleaned up and modified world settings are restored.
"""
from __future__ import annotations

import carla

from .logger import get_logger

log = get_logger()


class CarlaSession:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client: carla.Client | None = None
        self.world: carla.World | None = None
        self._actors: list[carla.Actor] = []
        self._original_settings: carla.WorldSettings | None = None

    # ---- connect / cleanup ---------------------------------------------------
    def __enter__(self) -> "CarlaSession":
        c = self.cfg["client"]
        log.info("CARLA server connection attempt: %s:%s", c["host"], c["port"])
        self.client = carla.Client(c["host"], c["port"])
        self.client.set_timeout(c["timeout"])

        version = self.client.get_server_version()
        log.info("server connected (server %s / client %s)",
                 version, self.client.get_client_version())

        wcfg = self.cfg["world"]
        if wcfg.get("map"):
            log.info("map load: %s", wcfg["map"])
            self.world = self.client.load_world(wcfg["map"])
        else:
            self.world = self.client.get_world()
            log.info("using current map: %s", self.world.get_map().name)

        self._original_settings = self.world.get_settings()
        if wcfg.get("synchronous_mode"):
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = wcfg["fixed_delta_seconds"]
            self.world.apply_settings(settings)
            log.info("synchronous mode ON (Δt=%.3fs)", wcfg["fixed_delta_seconds"])
        return self

    def __exit__(self, exc_type, exc, tb):
        log.info("cleaning up session: destroying %d actors", len(self._actors))
        for actor in reversed(self._actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        if self.world is not None and self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            log.info("world settings restored")
        return False  # do not suppress exceptions

    # ---- helpers ----------------------------------------------------------
    @property
    def is_sync(self) -> bool:
        return bool(self.cfg["world"].get("synchronous_mode"))

    def tick(self):
        """Tick the world and return the snapshot used as the timing source."""
        if self.is_sync:
            self.world.tick()
            return self.world.get_snapshot()
        return self.world.wait_for_tick()

    def register(self, actor: carla.Actor) -> carla.Actor:
        """Register an actor for cleanup when the session exits."""
        self._actors.append(actor)
        return actor

    def spawn_vehicle(self) -> carla.Vehicle:
        """spawn a vehicle from the configured blueprint and spawn point.

        If the map has no recommended spawn points, as with some custom
        OpenDRIVE tracks, a lane waypoint is used as a rough fallback.
        """
        vcfg = self.cfg["vehicle"]
        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.find(vcfg["blueprint"])

        idx = vcfg.get("spawn_index") or 0
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            transform = spawn_points[idx % len(spawn_points)]
            origin = f"spawn[{idx}]"
        else:
            transform = self._waypoint_spawn(idx)
            origin = f"waypoint[{idx}] (no recommended spawn point, lane fallback)"

        vehicle = self.world.try_spawn_actor(bp, transform)
        if vehicle is None:
            raise RuntimeError(f"vehicle spawn failed ({origin} possible collision)")
        self.register(vehicle)
        log.info("vehicle spawn: %s (id=%d) @ %s", vcfg["blueprint"], vehicle.id, origin)
        return vehicle

    def _waypoint_spawn(self, idx: int) -> carla.Transform:
        """Build a spawn transform from a lane waypoint, raised 0.3 m above the road."""
        waypoints = self.world.get_map().generate_waypoints(2.0)
        if not waypoints:
            raise RuntimeError("No driving-lane waypoint exists in this map.")
        wp = waypoints[idx % len(waypoints)]
        tf = wp.transform
        tf.location.z += 0.3
        return tf
