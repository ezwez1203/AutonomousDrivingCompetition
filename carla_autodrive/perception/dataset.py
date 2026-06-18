"""Ground-truth label helpers for Phase 3 synthetic perception datasets."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import carla

from carla_autodrive.control import RoutePurePursuitController, build_track_lane_route
from carla_autodrive.maps import TrackSpec
from carla_autodrive.sensors.frames import PerceptionInput


class PerceptionGroundTruthLabeler:
    """Create lane, traffic-light, and obstacle labels from CARLA ground truth."""

    def __init__(
        self,
        *,
        spec: TrackSpec | None = None,
        route_lane: int = 2,
        route_spacing_mm: float = 100.0,
        obstacle_actor_filter: str = "vehicle.*",
        max_obstacle_distance_m: float = 8.0,
    ):
        self.spec = spec
        self.route_lane = int(route_lane)
        self.max_obstacle_distance_m = float(max_obstacle_distance_m)
        self.obstacle_actor_filter = obstacle_actor_filter
        self.route_follower: RoutePurePursuitController | None = None
        if spec is not None:
            route = build_track_lane_route(
                spec,
                lane=self.route_lane,
                spacing_mm=route_spacing_mm,
            )
            self.route_follower = RoutePurePursuitController(
                route,
                curve_speed_enabled=False,
            )

    def label(self, *, world: carla.World, vehicle: carla.Vehicle, snapshot: PerceptionInput) -> dict[str, Any]:
        vehicle_tf = vehicle.get_transform()
        return {
            "sim_frame": int(snapshot.sim_frame),
            "timestamp": float(snapshot.timestamp),
            "snapshot": {
                "speed_mps": float(snapshot.speed_mps),
                "sensor_frames": dict(snapshot.sensor_frames),
            },
            "vehicle": {
                "id": int(vehicle.id),
                "type_id": vehicle.type_id,
                "transform": snapshot.vehicle_transform,
            },
            "lane": self._lane_label(vehicle_tf, snapshot.speed_mps),
            "traffic_light": self._traffic_light_label(world, vehicle_tf),
            "obstacles": self._obstacle_labels(world, vehicle, vehicle_tf),
        }

    def _lane_label(self, vehicle_tf: carla.Transform, speed_mps: float) -> dict[str, Any]:
        if self.route_follower is None:
            return {
                "detected": False,
                "message": "track route unavailable",
            }

        target = self.route_follower.target(vehicle_tf, speed_mps)
        lane_width_m = None if self.spec is None else float(self.spec.lane_width)
        center_error_norm = (
            0.0
            if lane_width_m is None or lane_width_m <= 0.0
            else float(target.cross_track_error_m / (lane_width_m * 0.5))
        )
        return {
            "detected": True,
            "lane": self.route_lane,
            "route_index": self.route_follower.current_index,
            "route_s_m": self.route_follower.current_s_m,
            "center_error_m": float(target.cross_track_error_m),
            "center_error_norm": center_error_norm,
            "heading_error_rad": float(target.heading_error_rad),
            "lane_width_m": lane_width_m,
        }

    def _traffic_light_label(self, world: carla.World, vehicle_tf: carla.Transform) -> dict[str, Any]:
        nearest = None
        nearest_local = None
        nearest_distance = float("inf")
        for actor in world.get_actors().filter("traffic.traffic_light*"):
            local = _location_to_vehicle_local(vehicle_tf, actor.get_transform().location)
            distance = math.hypot(local["x"], local["y"])
            if local["x"] >= -1.0 and distance < nearest_distance:
                nearest = actor
                nearest_local = local
                nearest_distance = distance

        if nearest is None or nearest_local is None:
            return {
                "detected": False,
                "message": "traffic light actor unavailable",
            }

        state = nearest.get_state()
        return {
            "detected": True,
            "actor_id": int(nearest.id),
            "state": _traffic_state_name(state),
            "distance_m": float(nearest_distance),
            "vehicle_local": nearest_local,
        }

    def _obstacle_labels(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        vehicle_tf: carla.Transform,
    ) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []
        for actor in world.get_actors().filter(self.obstacle_actor_filter):
            if actor.id == vehicle.id:
                continue
            local = _location_to_vehicle_local(vehicle_tf, actor.get_transform().location)
            distance = math.hypot(local["x"], local["y"])
            if distance > self.max_obstacle_distance_m:
                continue
            extent = actor.bounding_box.extent
            labels.append(
                {
                    "actor_id": int(actor.id),
                    "type_id": actor.type_id,
                    "distance_m": float(distance),
                    "vehicle_local": local,
                    "bbox_m": {
                        "length": float(extent.x * 2.0),
                        "width": float(extent.y * 2.0),
                        "height": float(extent.z * 2.0),
                    },
                }
            )
        labels.sort(key=lambda item: item["distance_m"])
        return labels


def append_label_jsonl(path: str | Path, label: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(label, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _location_to_vehicle_local(vehicle_tf: carla.Transform, location: carla.Location) -> dict[str, float]:
    dx = location.x - vehicle_tf.location.x
    dy = location.y - vehicle_tf.location.y
    dz = location.z - vehicle_tf.location.z
    yaw = math.radians(vehicle_tf.rotation.yaw)
    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)
    return {
        "x": float(dx * forward_x + dy * forward_y),
        "y": float(dx * right_x + dy * right_y),
        "z": float(dz),
    }


def _traffic_state_name(state: carla.TrafficLightState) -> str:
    text = str(state)
    return text.rsplit(".", maxsplit=1)[-1].lower()
