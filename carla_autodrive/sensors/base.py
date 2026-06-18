"""Shared base class for sensor wrappers.

Sensor callbacks run on a separate thread and keep only the latest payload. The main loop reads that payload safely through get_latest(). Phase 2/3 camera, LiDAR, and radar wrappers share this interface.
"""
from __future__ import annotations

import threading

import carla

from ..utils.logger import get_logger

log = get_logger()


def make_transform(position, rotation) -> carla.Transform:
    """Convert [x, y, z] and [roll, pitch, yaw] into a CARLA transform."""
    loc = carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2]))
    rot = carla.Rotation(
        roll=float(rotation[0]),
        pitch=float(rotation[1]),
        yaw=float(rotation[2]),
    )
    return carla.Transform(loc, rot)


class SensorBase:
    """Encapsulate sensor blueprint creation, vehicle attachment, and callback registration."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.sensor: carla.Sensor | None = None
        self._latest = None
        self._frame = 0
        self._lock = threading.Lock()

    # implemented by subclasses ------------------------------------------------
    def _configure_blueprint(self, bp: carla.ActorBlueprint) -> None:
        """Set blueprint attributes."""
        raise NotImplementedError

    def _on_data(self, data) -> None:
        """Server callback. Store processed results with self._store(...)."""
        raise NotImplementedError

    # shared ---------------------------------------------------------------
    def spawn(self, world: carla.World, parent: carla.Actor) -> carla.Sensor:
        bp = world.get_blueprint_library().find(self.cfg["type"])
        self._configure_blueprint(bp)
        transform = make_transform(self.cfg["position"], self.cfg["rotation"])
        self.sensor = world.spawn_actor(bp, transform, attach_to=parent)
        self.sensor.listen(self._on_data)
        log.info("sensor attached: %s (%s, id=%d)", self.name, self.cfg["type"], self.sensor.id)
        return self.sensor

    def _store(self, frame: int, payload) -> None:
        with self._lock:
            self._frame = frame
            self._latest = payload

    def get_latest(self):
        """Return (frame, payload), or (0, None) before the first callback."""
        with self._lock:
            return self._frame, self._latest

    def destroy(self) -> None:
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
