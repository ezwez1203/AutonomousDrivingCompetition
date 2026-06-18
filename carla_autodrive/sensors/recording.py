"""Recording utilities for Phase 2 sensor snapshots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .frames import PerceptionInput


def save_snapshot_npz(snapshot: PerceptionInput, path: str | Path) -> Path:
    """Save one PerceptionInput snapshot as a compressed NPZ file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sim_frame": snapshot.sim_frame,
        "timestamp": snapshot.timestamp,
        "vehicle_transform": snapshot.vehicle_transform,
        "speed_mps": snapshot.speed_mps,
        "sensor_frames": snapshot.sensor_frames,
        "radar_names": list(snapshot.radar_by_name.keys()),
    }
    payload = {
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        "lidar_points": snapshot.lidar_points,
        "radar_points": snapshot.radar_points,
    }
    if snapshot.camera_bgra is not None:
        payload["camera_bgra"] = snapshot.camera_bgra
    for name, points in snapshot.radar_by_name.items():
        payload[f"{name}_points"] = points
    np.savez_compressed(out, **payload)
    return out
