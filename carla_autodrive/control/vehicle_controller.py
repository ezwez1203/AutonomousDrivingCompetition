"""Phase 4 vehicle controller combining Pure Pursuit and speed PID."""
from __future__ import annotations

from dataclasses import dataclass

import carla

from carla_autodrive.perception.types import PerceptionOutput

from .hardware_limits import HardwareLimiter
from .pid import PIDConfig, PIDController
from .pure_pursuit import PurePursuitConfig, PurePursuitController
from .route_following import RoutePurePursuitController
from .types import ControlCommand


@dataclass(slots=True)
class VehicleControllerConfig:
    target_speed_mps: float = 3.0
    max_throttle: float = 0.45
    max_brake: float = 0.75
    brake_overspeed_margin_mps: float = 1.0
    obstacle_min_x_m: float = 1.0
    obstacle_stop_distance_m: float = 1.0
    obstacle_slow_distance_m: float = 2.0
    obstacle_y_abs_m: float = 0.8
    reverse_brake_speed_mps: float = 0.2
    reverse_min_throttle: float = 0.16
    parking_line_steer_gain: float = 0.18
    parking_line_max_steer_correction: float = 0.12


class VehicleController:
    """Map waypoint following controller."""

    def __init__(
        self,
        cfg: VehicleControllerConfig | None = None,
        pursuit_cfg: PurePursuitConfig | None = None,
        speed_pid_cfg: PIDConfig | None = None,
        hardware_limiter: HardwareLimiter | None = None,
    ):
        self.cfg = cfg or VehicleControllerConfig()
        self.pursuit = PurePursuitController(pursuit_cfg)
        self.speed_pid = PIDController(speed_pid_cfg)
        self.hardware_limiter = hardware_limiter or HardwareLimiter()
        self._reverse_mode = False

    def run_step(
        self,
        vehicle,
        current_waypoint: carla.Waypoint,
        perception: PerceptionOutput | None,
        dt: float,
    ) -> tuple[carla.VehicleControl, ControlCommand]:
        speed = self._vehicle_speed(vehicle)
        target = self.pursuit.target(vehicle.get_transform(), current_waypoint, speed)
        return self._control_from_target(vehicle, speed, target, perception, dt)

    def run_route_step(
        self,
        vehicle,
        route_follower: RoutePurePursuitController,
        perception: PerceptionOutput | None,
        dt: float,
    ) -> tuple[carla.VehicleControl, ControlCommand]:
        speed = self._vehicle_speed(vehicle)
        target = route_follower.target(vehicle.get_transform(), speed)
        return self._control_from_target(vehicle, speed, target, perception, dt)

    def _control_from_target(
        self,
        vehicle,
        speed: float,
        target,
        perception: PerceptionOutput | None,
        dt: float,
    ) -> tuple[carla.VehicleControl, ControlCommand]:
        desired_speed, reason = self._desired_speed(perception)
        speed_limit = getattr(target, "speed_limit_mps", None)
        if speed_limit is not None and speed_limit < desired_speed:
            desired_speed = max(0.0, float(speed_limit))
            route_reason = getattr(target, "reason", None)
            if route_reason:
                reason = route_reason if reason == "lane_follow" else f"{reason}+{route_reason}"
            else:
                reason = "curve_slow" if reason == "lane_follow" else f"{reason}+curve_slow"
        desired_reverse = bool(getattr(target, "reverse", False))
        if desired_reverse != self._reverse_mode:
            if speed > self.cfg.reverse_brake_speed_mps:
                throttle, brake = self.hardware_limiter.clamp(
                    throttle=0.0,
                    brake=float(self.cfg.max_brake),
                    reverse=self._reverse_mode,
                    dt=dt,
                )
                vehicle_control = carla.VehicleControl(
                    throttle=throttle,
                    steer=0.0,
                    brake=brake,
                    reverse=self._reverse_mode,
                )
                command = ControlCommand(
                    throttle=throttle,
                    steer=0.0,
                    brake=brake,
                    current_speed_mps=float(speed),
                    target_speed_mps=float(self.cfg.target_speed_mps),
                    desired_speed_mps=0.0,
                    target_distance_m=float(target.distance_m),
                    cross_track_error_m=float(target.cross_track_error_m),
                    heading_error_rad=float(target.heading_error_rad),
                    reason="gear_shift_to_reverse" if desired_reverse else "gear_shift_to_drive",
                    reverse=self._reverse_mode,
                )
                return vehicle_control, command
            self._reverse_mode = desired_reverse
            self.speed_pid.reset()
            self.hardware_limiter.reset()

        steer = float(target.steer)
        if perception is not None and perception.parking.detected:
            route_reason = getattr(target, "reason", None)
            if desired_reverse or route_reason in {"route_approach", "route_stop"}:
                correction = -perception.parking.center_error_norm * self.cfg.parking_line_steer_gain
                correction = max(
                    -self.cfg.parking_line_max_steer_correction,
                    min(self.cfg.parking_line_max_steer_correction, correction),
                )
                steer = max(-1.0, min(1.0, steer + correction))

        speed_error = desired_speed - speed
        accel_cmd = self.speed_pid.step(speed_error, dt)

        throttle = min(self.cfg.max_throttle, max(0.0, accel_cmd))
        if self._reverse_mode and desired_speed > 0.05 and speed < desired_speed:
            throttle = max(throttle, min(self.cfg.max_throttle, self.cfg.reverse_min_throttle))
        brake = 0.0
        if speed > desired_speed + self.cfg.brake_overspeed_margin_mps:
            brake = min(self.cfg.max_brake, max(0.0, -accel_cmd))
        if desired_speed <= 0.05:
            throttle = 0.0
            if speed > self.cfg.reverse_brake_speed_mps:
                brake = self.cfg.max_brake

        throttle, brake = self.hardware_limiter.clamp(
            throttle=throttle,
            brake=brake,
            reverse=self._reverse_mode,
            dt=dt,
        )

        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            reverse=self._reverse_mode,
        )
        command = ControlCommand(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            current_speed_mps=float(speed),
            target_speed_mps=float(self.cfg.target_speed_mps),
            desired_speed_mps=float(desired_speed),
            target_distance_m=float(target.distance_m),
            cross_track_error_m=float(target.cross_track_error_m),
            heading_error_rad=float(target.heading_error_rad),
            reason=reason,
            reverse=self._reverse_mode,
        )
        return vehicle_control, command

    def _desired_speed(self, perception: PerceptionOutput | None) -> tuple[float, str]:
        if perception is None:
            return self.cfg.target_speed_mps, "lane_follow"
        nearest = perception.obstacles.nearest
        if nearest is None:
            return self.cfg.target_speed_mps, "lane_follow"
        if nearest.x < self.cfg.obstacle_min_x_m or abs(nearest.y) > self.cfg.obstacle_y_abs_m:
            return self.cfg.target_speed_mps, "lane_follow"
        if nearest.distance <= self.cfg.obstacle_stop_distance_m:
            return 0.0, f"obstacle_stop:{nearest.source}"
        if nearest.distance <= self.cfg.obstacle_slow_distance_m:
            ratio = (
                (nearest.distance - self.cfg.obstacle_stop_distance_m)
                / max(1e-6, self.cfg.obstacle_slow_distance_m - self.cfg.obstacle_stop_distance_m)
            )
            return max(0.5, self.cfg.target_speed_mps * ratio), f"obstacle_slow:{nearest.source}"
        return self.cfg.target_speed_mps, "lane_follow"

    @staticmethod
    def _vehicle_speed(vehicle) -> float:
        velocity = vehicle.get_velocity()
        return float((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5)
