"""Linear design models for CARLA vehicle control.

The functions here build controller design models only.  They do not modify
CARLA vehicle physics; the simulated plant remains CARLA's native PhysX model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_FRONT_CORNERING_STIFFNESS_N_PER_RAD = 80000.0
DEFAULT_REAR_CORNERING_STIFFNESS_N_PER_RAD = 80000.0
DEFAULT_LF_M = 1.35
DEFAULT_LR_M = 1.45
DEFAULT_MAX_STEER_ANGLE_RAD = math.radians(35.0)


@dataclass(slots=True)
class VehicleParameters:
    """Parameters used by the linearized bicycle design model."""

    mass_kg: float = 1500.0
    yaw_inertia_kg_m2: float = 2500.0
    lf_m: float = DEFAULT_LF_M
    lr_m: float = DEFAULT_LR_M
    caf_n_per_rad: float = DEFAULT_FRONT_CORNERING_STIFFNESS_N_PER_RAD
    car_n_per_rad: float = DEFAULT_REAR_CORNERING_STIFFNESS_N_PER_RAD
    max_steer_angle_rad: float = DEFAULT_MAX_STEER_ANGLE_RAD

    @property
    def wheel_base_m(self) -> float:
        return self.lf_m + self.lr_m


def longitudinal_double_integrator_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return continuous-time x_dot = A x + B u for x=[position, velocity]."""

    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    b = np.array([[0.0], [1.0]], dtype=float)
    return a, b


def lateral_bicycle_matrices(params: VehicleParameters, vx_mps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Rajamani-style continuous linear bicycle matrices.

    State is x = [e, e_dot, psi_e, psi_dot_e], where positive lateral error and
    steer angle follow the standard vehicle-dynamics left-positive convention.
    CARLA's normalized steering sign is handled in controller.py.
    """

    vx = max(abs(float(vx_mps)), 0.1)
    m = max(float(params.mass_kg), 1.0)
    iz = max(float(params.yaw_inertia_kg_m2), 1.0)
    lf = max(float(params.lf_m), 1e-3)
    lr = max(float(params.lr_m), 1e-3)
    caf = max(float(params.caf_n_per_rad), 1.0)
    car = max(float(params.car_n_per_rad), 1.0)

    a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [
                0.0,
                -(2.0 * caf + 2.0 * car) / (m * vx),
                (2.0 * caf + 2.0 * car) / m,
                (-2.0 * caf * lf + 2.0 * car * lr) / (m * vx),
            ],
            [0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                (-2.0 * caf * lf + 2.0 * car * lr) / (iz * vx),
                (2.0 * caf * lf - 2.0 * car * lr) / iz,
                -(2.0 * caf * lf * lf + 2.0 * car * lr * lr) / (iz * vx),
            ],
        ],
        dtype=float,
    )
    b = np.array([[0.0], [2.0 * caf / m], [0.0], [2.0 * caf * lf / iz]], dtype=float)
    return a, b


def discretize_forward_euler(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a continuous design model with forward Euler."""

    dt = max(float(dt), 1e-4)
    eye = np.eye(a.shape[0], dtype=float)
    return eye + a * dt, b * dt


def solve_discrete_lqr(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    max_iterations: int = 150,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Solve a discrete LQR gain using Riccati iteration.

    Returns K for u = -K x.  This avoids adding a scipy dependency to the
    project while remaining sufficient for the small 2x2 and 4x4 systems used
    by these design models.
    """

    p = np.array(q, dtype=float, copy=True)
    r = np.array(r, dtype=float, copy=True)
    for _ in range(max_iterations):
        bt_p = b.T @ p
        gain = np.linalg.solve(r + bt_p @ b, bt_p @ a)
        next_p = a.T @ p @ a - a.T @ p @ b @ gain + q
        if np.max(np.abs(next_p - p)) < tolerance:
            p = next_p
            break
        p = next_p
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


def longitudinal_lqr_gain(
    dt: float,
    q: tuple[float, float] = (0.5, 2.0),
    r: float = 0.8,
) -> np.ndarray:
    """Return discrete LQR gain for the double-integrator longitudinal model."""

    a, b = longitudinal_double_integrator_matrices()
    ad, bd = discretize_forward_euler(a, b, dt)
    return solve_discrete_lqr(ad, bd, np.diag(q), np.array([[float(r)]], dtype=float))


def lateral_lqr_gain(
    params: VehicleParameters,
    vx_mps: float,
    dt: float,
    q: tuple[float, float, float, float] = (2.5, 0.25, 4.0, 0.2),
    r: float = 1.0,
) -> np.ndarray:
    """Return discrete LQR gain for the linearized bicycle lateral model."""

    a, b = lateral_bicycle_matrices(params, vx_mps)
    ad, bd = discretize_forward_euler(a, b, dt)
    return solve_discrete_lqr(ad, bd, np.diag(q), np.array([[float(r)]], dtype=float))


def vehicle_parameters_from_physics_control(
    physics_control,
    *,
    caf_n_per_rad: float = DEFAULT_FRONT_CORNERING_STIFFNESS_N_PER_RAD,
    car_n_per_rad: float = DEFAULT_REAR_CORNERING_STIFFNESS_N_PER_RAD,
) -> VehicleParameters:
    """Build design-model parameters from carla.VehiclePhysicsControl."""

    mass = float(getattr(physics_control, "mass", 1500.0) or 1500.0)
    wheels = list(getattr(physics_control, "wheels", []) or [])
    lf, lr = _axle_distances_from_wheels(wheels)
    wheel_base = lf + lr
    yaw_inertia = mass * max(wheel_base, 1.0) ** 2 / 12.0
    max_steer = _max_steer_from_wheels(wheels)
    return VehicleParameters(
        mass_kg=mass,
        yaw_inertia_kg_m2=yaw_inertia,
        lf_m=lf,
        lr_m=lr,
        caf_n_per_rad=float(caf_n_per_rad),
        car_n_per_rad=float(car_n_per_rad),
        max_steer_angle_rad=max_steer,
    )


def vehicle_parameters_from_actor(
    vehicle,
    *,
    caf_n_per_rad: float = DEFAULT_FRONT_CORNERING_STIFFNESS_N_PER_RAD,
    car_n_per_rad: float = DEFAULT_REAR_CORNERING_STIFFNESS_N_PER_RAD,
) -> VehicleParameters:
    """Read CARLA physics control and return design-model parameters."""

    return vehicle_parameters_from_physics_control(
        vehicle.get_physics_control(),
        caf_n_per_rad=caf_n_per_rad,
        car_n_per_rad=car_n_per_rad,
    )


def _axle_distances_from_wheels(wheels: list) -> tuple[float, float]:
    xs: list[float] = []
    for wheel in wheels:
        position = getattr(wheel, "position", None)
        if position is None:
            continue
        xs.append(float(getattr(position, "x", 0.0)))
    if not xs:
        return DEFAULT_LF_M, DEFAULT_LR_M

    scale = 0.01 if max(abs(x) for x in xs) > 20.0 else 1.0
    front_x = max(xs) * scale
    rear_x = min(xs) * scale
    lf = front_x if front_x > 0.05 else DEFAULT_LF_M
    lr = -rear_x if rear_x < -0.05 else DEFAULT_LR_M
    return lf, lr


def _max_steer_from_wheels(wheels: list) -> float:
    angles = [abs(float(getattr(wheel, "max_steer_angle", 0.0) or 0.0)) for wheel in wheels]
    max_angle_deg = max(angles, default=0.0)
    if max_angle_deg <= 0.0:
        return DEFAULT_MAX_STEER_ANGLE_RAD
    return math.radians(max_angle_deg)
