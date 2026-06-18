"""Sensor calibration and coordinate conversion helpers."""
from __future__ import annotations

import numpy as np

from .base import make_transform


def sensor_to_vehicle_matrix(sensor_cfg: dict) -> np.ndarray:
    """Return the 4x4 transform from a sensor-local frame to the vehicle frame."""
    transform = make_transform(sensor_cfg["position"], sensor_cfg["rotation"])
    return np.asarray(transform.get_matrix(), dtype=np.float32)


def transform_points(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 matrix to an (N, 3) point array."""
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
    ones = np.ones((len(points), 1), dtype=np.float32)
    hom = np.concatenate((points, ones), axis=1)
    return (hom @ matrix.T)[:, :3].astype(np.float32, copy=False)


def points_sensor_to_vehicle(points_xyz: np.ndarray, sensor_cfg: dict) -> np.ndarray:
    """Transform sensor-local points into the vehicle frame."""
    return transform_points(points_xyz, sensor_to_vehicle_matrix(sensor_cfg))


def transform_to_dict(transform) -> dict:
    """Serialize a carla.Transform-like object into plain Python values."""
    loc = transform.location
    rot = transform.rotation
    return {
        "location": {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)},
        "rotation": {
            "roll": float(rot.roll),
            "pitch": float(rot.pitch),
            "yaw": float(rot.yaw),
        },
    }
