"""CARLA runtime interface for the LQR/double-integrator controller."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import carla

from .controller import CarlaLQRControllerConfig, CarlaLQRVehicleController, ControllerDebug


@dataclass(slots=True)
class CarlaInterfaceConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 20.0
    map_name: str | None = None
    vehicle_blueprint: str = "vehicle.tesla.model3"
    spawn_index: int = 0
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.05


@dataclass(slots=True)
class SensorReadings:
    transform: carla.Transform
    velocity: carla.Vector3D
    acceleration: carla.Vector3D
    angular_velocity: carla.Vector3D
    waypoint: carla.Waypoint
    timestamp_s: float
    frame: int


@dataclass(slots=True)
class ControlLoopSample:
    readings: SensorReadings
    control: carla.VehicleControl
    debug: ControllerDebug


@dataclass(slots=True)
class CarlaLQRRuntime:
    cfg: CarlaInterfaceConfig = field(default_factory=CarlaInterfaceConfig)
    controller_cfg: CarlaLQRControllerConfig = field(default_factory=CarlaLQRControllerConfig)
    client: carla.Client | None = None
    world: carla.World | None = None
    vehicle: carla.Vehicle | None = None
    controller: CarlaLQRVehicleController | None = None
    _original_settings: carla.WorldSettings | None = None
    _actors: list[carla.Actor] = field(default_factory=list)

    def __enter__(self) -> "CarlaLQRRuntime":
        self.connect()
        self.setup_world()
        self.spawn_vehicle()
        if self.vehicle is None:
            raise RuntimeError("vehicle was not spawned")
        self.controller = CarlaLQRVehicleController(self.vehicle, self.controller_cfg)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def connect(self) -> None:
        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(self.cfg.timeout_s)
        self.world = self.client.load_world(self.cfg.map_name) if self.cfg.map_name else self.client.get_world()

    def setup_world(self) -> None:
        if self.world is None:
            raise RuntimeError("connect() must be called before setup_world()")
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(self.cfg.synchronous_mode)
        if self.cfg.synchronous_mode:
            settings.fixed_delta_seconds = float(self.cfg.fixed_delta_seconds)
        self.world.apply_settings(settings)

    def spawn_vehicle(self) -> carla.Vehicle:
        if self.world is None:
            raise RuntimeError("connect() must be called before spawn_vehicle()")
        blueprint = self.world.get_blueprint_library().find(self.cfg.vehicle_blueprint)
        spawn_points = self.world.get_map().get_spawn_points()
        if spawn_points:
            transform = spawn_points[self.cfg.spawn_index % len(spawn_points)]
        else:
            waypoints = self.world.get_map().generate_waypoints(2.0)
            if not waypoints:
                raise RuntimeError("No spawn points or driving waypoints are available.")
            transform = waypoints[self.cfg.spawn_index % len(waypoints)].transform
            transform.location.z += 0.35

        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is None:
            raise RuntimeError("vehicle spawn failed, likely due to a collision at the spawn point")
        self.vehicle = vehicle
        self._actors.append(vehicle)
        return vehicle

    def tick(self) -> carla.WorldSnapshot:
        if self.world is None:
            raise RuntimeError("world is not connected")
        if self.cfg.synchronous_mode:
            self.world.tick()
            return self.world.get_snapshot()
        return self.world.wait_for_tick()

    def read_sensor_data(self) -> SensorReadings:
        """Read proprioceptive vehicle state and nearest map waypoint."""

        if self.world is None or self.vehicle is None:
            raise RuntimeError("runtime is not initialized")
        snapshot = self.world.get_snapshot()
        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError("No driving waypoint was found near the vehicle.")
        return SensorReadings(
            transform=self.vehicle.get_transform(),
            velocity=self.vehicle.get_velocity(),
            acceleration=self.vehicle.get_acceleration(),
            angular_velocity=self.vehicle.get_angular_velocity(),
            waypoint=waypoint,
            timestamp_s=float(snapshot.timestamp.elapsed_seconds),
            frame=int(snapshot.frame),
        )

    def run_step(self, target_speed_mps: float | None = None) -> ControlLoopSample:
        if self.vehicle is None or self.controller is None:
            raise RuntimeError("runtime is not initialized")
        self.tick()
        readings = self.read_sensor_data()
        control, debug = self.controller.run_step(
            self.vehicle,
            readings.waypoint,
            self.cfg.fixed_delta_seconds,
            target_speed_mps=target_speed_mps,
        )
        self.vehicle.apply_control(control)
        return ControlLoopSample(readings=readings, control=control, debug=debug)

    def run_for(self, duration_s: float, target_speed_mps: float | None = None) -> list[ControlLoopSample]:
        deadline = time.monotonic() + max(float(duration_s), 0.0)
        samples: list[ControlLoopSample] = []
        while time.monotonic() < deadline:
            samples.append(self.run_step(target_speed_mps=target_speed_mps))
        return samples

    def close(self) -> None:
        for actor in reversed(self._actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        self._actors.clear()
        if self.world is not None and self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
        self.vehicle = None
        self.controller = None
