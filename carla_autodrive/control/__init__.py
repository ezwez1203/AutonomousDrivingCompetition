"""control module (implemented in later phases)."""
"""Phase 4 control modules."""

from .pid import PIDConfig, PIDController
from .hardware_limits import HardwareLimitConfig, HardwareLimiter
from .pure_pursuit import PurePursuitConfig, PurePursuitController, PursuitTarget
from .route_following import RoutePoint, RoutePurePursuitController, RouteTarget, build_track_lane_route
from .types import ControlCommand
from .vehicle_controller import VehicleController, VehicleControllerConfig
from .vehicle_model import (
    VehicleParameters,
    lateral_bicycle_matrices,
    lateral_lqr_gain,
    longitudinal_double_integrator_matrices,
    longitudinal_lqr_gain,
    vehicle_parameters_from_actor,
    vehicle_parameters_from_physics_control,
)
from .controller import (
    CarlaLQRControllerConfig,
    CarlaLQRVehicleController,
    LateralLQRController,
    LateralLQRControllerConfig,
    LongitudinalController,
    LongitudinalControllerConfig,
)
from .carla_interface import (
    CarlaInterfaceConfig,
    CarlaLQRRuntime,
    ControlLoopSample,
    SensorReadings,
)

__all__ = [
    "ControlCommand",
    "HardwareLimitConfig",
    "HardwareLimiter",
    "PIDConfig",
    "PIDController",
    "PurePursuitConfig",
    "PurePursuitController",
    "PursuitTarget",
    "RoutePoint",
    "RoutePurePursuitController",
    "RouteTarget",
    "VehicleController",
    "VehicleControllerConfig",
    "CarlaLQRControllerConfig",
    "CarlaLQRVehicleController",
    "CarlaInterfaceConfig",
    "CarlaLQRRuntime",
    "ControlLoopSample",
    "LateralLQRController",
    "LateralLQRControllerConfig",
    "LongitudinalController",
    "LongitudinalControllerConfig",
    "SensorReadings",
    "VehicleParameters",
    "build_track_lane_route",
    "lateral_bicycle_matrices",
    "lateral_lqr_gain",
    "longitudinal_double_integrator_matrices",
    "longitudinal_lqr_gain",
    "vehicle_parameters_from_actor",
    "vehicle_parameters_from_physics_control",
]
