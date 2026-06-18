"""Phase 2 sensor stack: camera, LiDAR, and multi-direction radar."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import points_sensor_to_vehicle, transform_to_dict
from .camera import RGBCamera
from .frames import PerceptionInput
from .lidar import Lidar
from .radar import Radar


_DEFAULT_RADAR_VARIANTS = (
    ("front", [0.4, 0.0, 0.2], [0.0, 0.0, 0.0]),
    ("rear", [-0.4, 0.0, 0.2], [0.0, 0.0, 180.0]),
    ("right", [0.0, 0.3, 0.2], [0.0, 0.0, 90.0]),
    ("left", [0.0, -0.3, 0.2], [0.0, 0.0, -90.0]),
)


def expand_radar_configs(radar_cfg: dict) -> list[tuple[str, dict]]:
    """Build named radar configs from position_variants."""
    base = {k: v for k, v in radar_cfg.items() if k != "position_variants"}
    variants = radar_cfg.get("position_variants") or [
        {"name": name, "position": pos, "rotation": rot}
        for name, pos, rot in _DEFAULT_RADAR_VARIANTS
    ]

    configs: list[tuple[str, dict]] = []
    for index, variant in enumerate(variants):
        default_name, _default_pos, default_rot = _DEFAULT_RADAR_VARIANTS[
            index % len(_DEFAULT_RADAR_VARIANTS)
        ]
        cfg = dict(base)
        if isinstance(variant, dict):
            name = str(variant.get("name") or default_name)
            cfg["position"] = variant.get("position", cfg.get("position"))
            cfg["rotation"] = variant.get("rotation", default_rot)
        else:
            name = default_name
            cfg["position"] = variant
            cfg["rotation"] = default_rot
        cfg.setdefault("type", "sensor.other.radar")
        cfg.setdefault("position", _DEFAULT_RADAR_VARIANTS[index % 4][1])
        cfg.setdefault("rotation", default_rot)
        configs.append((f"radar_{name}", cfg))
    return configs


@dataclass(slots=True)
class MultiRadar:
    """Spawn and read multiple short-range radar sensors."""

    radar_cfg: dict
    radars: list[Radar] = field(init=False)

    def __post_init__(self) -> None:
        self.radars = [
            Radar(cfg, name=name) for name, cfg in expand_radar_configs(self.radar_cfg)
        ]

    def spawn(self, world, parent) -> list:
        return [radar.spawn(world, parent) for radar in self.radars]

    def summaries(self) -> list[str]:
        return [radar.summary() for radar in self.radars]

    def latest_vehicle_points(self) -> tuple[dict[str, int], dict[str, np.ndarray], np.ndarray]:
        frames: dict[str, int] = {}
        by_name: dict[str, np.ndarray] = {}
        fused: list[np.ndarray] = []

        for sensor_index, radar in enumerate(self.radars):
            frame, detections = radar.get_latest()
            frames[radar.name] = int(frame)
            if detections is None or len(detections) == 0:
                converted = np.empty((0, 8), dtype=np.float32)
            else:
                xyz = points_sensor_to_vehicle(detections[:, 4:7], radar.cfg)
                sensor_id = np.full((len(detections), 1), sensor_index, dtype=np.float32)
                converted = np.concatenate(
                    (
                        xyz,
                        detections[:, 0:4].astype(np.float32, copy=False),
                        sensor_id,
                    ),
                    axis=1,
                )
            by_name[radar.name] = converted
            fused.append(converted)

        if fused:
            fused_points = np.concatenate(fused, axis=0)
        else:
            fused_points = np.empty((0, 8), dtype=np.float32)
        return frames, by_name, fused_points


class SensorStack:
    """Reusable Phase 2 stack that produces PerceptionInput snapshots."""

    def __init__(
        self,
        sensor_cfg: dict,
        *,
        enable_camera: bool = True,
        enable_lidar: bool = True,
        enable_radar: bool = True,
    ):
        self.camera = RGBCamera(sensor_cfg["camera"]) if enable_camera else None
        self.lidar = Lidar(sensor_cfg["lidar"]) if enable_lidar else None
        self.radar = MultiRadar(sensor_cfg["radar"]) if enable_radar else None

    def spawn(self, world, parent) -> list:
        actors = []
        if self.camera is not None:
            actors.append(self.camera.spawn(world, parent))
        if self.lidar is not None:
            actors.append(self.lidar.spawn(world, parent))
        if self.radar is not None:
            actors.extend(self.radar.spawn(world, parent))
        return actors

    def summaries(self) -> list[str]:
        rows: list[str] = []
        if self.camera is not None:
            rows.append(self.camera.summary())
        if self.lidar is not None:
            rows.append(self.lidar.summary())
        if self.radar is not None:
            rows.extend(self.radar.summaries())
        return rows

    def capture(self, vehicle, sim_frame: int, timestamp: float) -> PerceptionInput:
        sensor_frames: dict[str, int] = {}

        camera_bgra = None
        if self.camera is not None:
            frame, camera_bgra = self.camera.get_latest()
            sensor_frames["camera"] = int(frame)

        lidar_points = np.empty((0, 4), dtype=np.float32)
        if self.lidar is not None:
            frame, raw_lidar = self.lidar.get_latest()
            sensor_frames["lidar"] = int(frame)
            if raw_lidar is not None and len(raw_lidar) > 0:
                xyz = points_sensor_to_vehicle(raw_lidar[:, :3], self.lidar.cfg)
                lidar_points = np.concatenate(
                    (xyz, raw_lidar[:, 3:4].astype(np.float32, copy=False)),
                    axis=1,
                )

        radar_by_name: dict[str, np.ndarray] = {}
        radar_points = np.empty((0, 8), dtype=np.float32)
        if self.radar is not None:
            radar_frames, radar_by_name, radar_points = self.radar.latest_vehicle_points()
            sensor_frames.update(radar_frames)

        velocity = vehicle.get_velocity()
        speed_mps = float((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5)

        return PerceptionInput(
            sim_frame=int(sim_frame),
            timestamp=float(timestamp),
            vehicle_transform=transform_to_dict(vehicle.get_transform()),
            speed_mps=speed_mps,
            camera_bgra=camera_bgra,
            lidar_points=lidar_points,
            radar_points=radar_points,
            radar_by_name=radar_by_name,
            sensor_frames=sensor_frames,
        )
