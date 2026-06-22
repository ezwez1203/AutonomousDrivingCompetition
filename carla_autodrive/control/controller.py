"""Longitudinal and lateral controllers for CARLA vehicles.

The controller uses double-integrator and linearized-bicycle models only for
control-law design.  It always returns a carla.VehicleControl command intended
for vehicle.apply_control(); it never disables CARLA's native vehicle physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import carla
import numpy as np

from .vehicle_model import (
    VehicleParameters,
    lateral_lqr_gain,
    longitudinal_lqr_gain,
    vehicle_parameters_from_actor,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    return angle


@dataclass(slots=True)
class LongitudinalControllerConfig:
    mode: str = "pid"
    kp: float = 0.45
    ki: float = 0.04
    kd: float = 0.02
    integral_limit: float = 10.0
    max_accel_mps2: float = 3.0
    max_decel_mps2: float = 6.0
    max_throttle: float = 1.0
    max_brake: float = 1.0
    lqr_q_position: float = 0.5
    lqr_q_velocity: float = 2.0
    lqr_r: float = 0.8


@dataclass(slots=True)
class LateralLQRControllerConfig:
    q_lateral_error: float = 2.5
    q_lateral_rate: float = 0.25
    q_heading_error: float = 4.0
    q_heading_rate: float = 0.2
    r_steer: float = 1.0
    min_design_speed_mps: float = 0.5
    caf_n_per_rad: float = 80000.0
    car_n_per_rad: float = 80000.0


@dataclass(slots=True)
class CarlaLQRControllerConfig:
    target_speed_mps: float = 5.0
    longitudinal: LongitudinalControllerConfig = field(default_factory=LongitudinalControllerConfig)
    lateral: LateralLQRControllerConfig = field(default_factory=LateralLQRControllerConfig)


@dataclass(slots=True)
class LongitudinalDebug:
    target_speed_mps: float
    speed_mps: float
    speed_error_mps: float
    accel_command_mps2: float
    throttle: float
    brake: float


@dataclass(slots=True)
class LateralDebug:
    cross_track_error_m: float
    cross_track_rate_mps: float
    heading_error_rad: float
    heading_rate_radps: float
    steer_angle_rad: float
    steer_normalized: float
    design_speed_mps: float
    gain: tuple[float, float, float, float]


@dataclass(slots=True)
class ControllerDebug:
    longitudinal: LongitudinalDebug
    lateral: LateralDebug
    waypoint: carla.Waypoint


class LongitudinalController:
    """PID or LQR acceleration controller for a double-integrator model."""

    def __init__(self, cfg: LongitudinalControllerConfig | None = None):
        self.cfg = cfg or LongitudinalControllerConfig()
        self.integral = 0.0
        self.prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def step(
        self,
        *,
        speed_mps: float,
        target_speed_mps: float,
        dt: float,
        position_error_m: float = 0.0,
    ) -> LongitudinalDebug:
        dt = max(float(dt), 1e-3)
        speed_error = float(target_speed_mps) - float(speed_mps)
        if self.cfg.mode.lower() == "lqr":
            gain = longitudinal_lqr_gain(
                dt,
                q=(self.cfg.lqr_q_position, self.cfg.lqr_q_velocity),
                r=self.cfg.lqr_r,
            )
            state = np.array([[float(position_error_m)], [float(speed_mps) - float(target_speed_mps)]])
            accel = float((-gain @ state).item())
        else:
            self.integral += speed_error * dt
            self.integral = clamp(self.integral, -self.cfg.integral_limit, self.cfg.integral_limit)
            derivative = 0.0 if self.prev_error is None else (speed_error - self.prev_error) / dt
            self.prev_error = speed_error
            accel = self.cfg.kp * speed_error + self.cfg.ki * self.integral + self.cfg.kd * derivative

        accel = clamp(accel, -self.cfg.max_decel_mps2, self.cfg.max_accel_mps2)
        throttle, brake = self._accel_to_throttle_brake(accel)
        return LongitudinalDebug(
            target_speed_mps=float(target_speed_mps),
            speed_mps=float(speed_mps),
            speed_error_mps=float(speed_error),
            accel_command_mps2=float(accel),
            throttle=float(throttle),
            brake=float(brake),
        )

    def _accel_to_throttle_brake(self, accel_mps2: float) -> tuple[float, float]:
        if accel_mps2 >= 0.0:
            throttle = accel_mps2 / max(self.cfg.max_accel_mps2, 1e-6)
            return clamp(throttle, 0.0, self.cfg.max_throttle), 0.0
        brake = -accel_mps2 / max(self.cfg.max_decel_mps2, 1e-6)
        return 0.0, clamp(brake, 0.0, self.cfg.max_brake)


class LateralLQRController:
    """LQR steering controller using a linearized bicycle design model."""

    def __init__(
        self,
        params: VehicleParameters,
        cfg: LateralLQRControllerConfig | None = None,
    ):
        self.params = params
        self.cfg = cfg or LateralLQRControllerConfig()

    def step(self, vehicle: carla.Vehicle, waypoint: carla.Waypoint, dt: float) -> LateralDebug:
        speed = vehicle_speed_mps(vehicle)
        design_speed = max(speed, self.cfg.min_design_speed_mps)
        state = self._lateral_state(vehicle, waypoint)
        gain = lateral_lqr_gain(
            self.params,
            design_speed,
            max(float(dt), 1e-3),
            q=(
                self.cfg.q_lateral_error,
                self.cfg.q_lateral_rate,
                self.cfg.q_heading_error,
                self.cfg.q_heading_rate,
            ),
            r=self.cfg.r_steer,
        )
        delta_left_rad = float((-gain @ state).item())
        max_steer = max(self.params.max_steer_angle_rad, 1e-3)

        # CARLA's positive normalized steer is rightward for the project
        # convention.  The bicycle model above uses left-positive steering.
        steer = clamp(-delta_left_rad / max_steer, -1.0, 1.0)
        delta_left_rad = clamp(delta_left_rad, -max_steer, max_steer)
        return LateralDebug(
            cross_track_error_m=float(state[0, 0]),
            cross_track_rate_mps=float(state[1, 0]),
            heading_error_rad=float(state[2, 0]),
            heading_rate_radps=float(state[3, 0]),
            steer_angle_rad=float(delta_left_rad),
            steer_normalized=float(steer),
            design_speed_mps=float(design_speed),
            gain=tuple(float(x) for x in gain.reshape(-1)),
        )

    @staticmethod
    def _lateral_state(vehicle: carla.Vehicle, waypoint: carla.Waypoint) -> np.ndarray:
        transform = vehicle.get_transform()
        wp_transform = waypoint.transform
        path_yaw = math.radians(wp_transform.rotation.yaw)
        vehicle_yaw = math.radians(transform.rotation.yaw)

        dx = transform.location.x - wp_transform.location.x
        dy = transform.location.y - wp_transform.location.y
        right_x = -math.sin(path_yaw)
        right_y = math.cos(path_yaw)
        error_right = dx * right_x + dy * right_y

        velocity = vehicle.get_velocity()
        lateral_rate_right = velocity.x * right_x + velocity.y * right_y
        angular_velocity = vehicle.get_angular_velocity()
        yaw_rate_radps = math.radians(float(angular_velocity.z))

        error_left = -error_right
        lateral_rate_left = -lateral_rate_right
        heading_error_left = wrap_pi(path_yaw - vehicle_yaw)
        return np.array(
            [[error_left], [lateral_rate_left], [heading_error_left], [yaw_rate_radps]],
            dtype=float,
        )


class CarlaLQRVehicleController:
    """Combined longitudinal/lateral CARLA controller."""

    def __init__(
        self,
        vehicle: carla.Vehicle,
        cfg: CarlaLQRControllerConfig | None = None,
        params: VehicleParameters | None = None,
    ):
        self.cfg = cfg or CarlaLQRControllerConfig()
        self.params = params or vehicle_parameters_from_actor(
            vehicle,
            caf_n_per_rad=self.cfg.lateral.caf_n_per_rad,
            car_n_per_rad=self.cfg.lateral.car_n_per_rad,
        )
        self.longitudinal = LongitudinalController(self.cfg.longitudinal)
        self.lateral = LateralLQRController(self.params, self.cfg.lateral)

    def run_step(
        self,
        vehicle: carla.Vehicle,
        waypoint: carla.Waypoint,
        dt: float,
        *,
        target_speed_mps: float | None = None,
        position_error_m: float = 0.0,
    ) -> tuple[carla.VehicleControl, ControllerDebug]:
        target_speed = self.cfg.target_speed_mps if target_speed_mps is None else float(target_speed_mps)
        long_dbg = self.longitudinal.step(
            speed_mps=vehicle_speed_mps(vehicle),
            target_speed_mps=target_speed,
            dt=dt,
            position_error_m=position_error_m,
        )
        lat_dbg = self.lateral.step(vehicle, waypoint, dt)
        control = carla.VehicleControl(
            throttle=long_dbg.throttle,
            steer=lat_dbg.steer_normalized,
            brake=long_dbg.brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        return control, ControllerDebug(longitudinal=long_dbg, lateral=lat_dbg, waypoint=waypoint)


def vehicle_speed_mps(vehicle: carla.Vehicle) -> float:
    velocity = vehicle.get_velocity()
    return float((velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z) ** 0.5)
