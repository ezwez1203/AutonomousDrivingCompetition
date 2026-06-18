"""Track specification utilities for config/track.yaml.

The config values are written in real-world millimeters and converted to CARLA
meters with the configured scale. The track supports three centerline modes:
legacy explicit line/arc segments, image-like closed Catmull-Rom control
points, and dense polyline point lists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..utils.config import load_config


@dataclass
class Segment:
    """Centerline segment. Units are scaled meters and radians."""
    type: str
    length: float
    curvature: float


@dataclass
class Geometry:
    """OpenDRIVE geometry record for pose lookup and generation."""
    s: float
    x: float
    y: float
    heading: float
    length: float
    curvature: float = 0.0


@dataclass
class TrackPose:
    """Track pose on the generated reference line/lane center."""
    s: float
    x: float
    y: float
    heading: float
    lane: int | None = None


class TrackSpec:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config("track")
        self.scale: float = float(self.cfg["meta"]["scale"])
        self.name: str = self.cfg["meta"]["name"]
        self._geometries: list[Geometry] | None = None

    # ---- unit conversion ---------------------------------------------
    def mm(self, value_mm: float) -> float:
        """Real mm -> CARLA meters with scale applied."""
        return value_mm / 1000.0 * self.scale

    # ---- road dimensions ---------------------------------------------
    @property
    def road_width(self) -> float:
        return self.mm(self.cfg["dimensions"]["road_width_mm"])

    @property
    def lane_width(self) -> float:
        return self.road_width / self.cfg["lanes"]["count"]

    @property
    def start_pose(self) -> tuple[float, float, float]:
        s = self.cfg["centerline"].get("start", {"x": 0, "y": 0, "heading_deg": 0})
        return self.mm(s["x"]), self.mm(s["y"]), math.radians(s["heading_deg"])

    # ---- centerline construction -------------------------------------
    def segments(self) -> list[Segment]:
        """Legacy line/arc segments from config, if present."""
        out: list[Segment] = []
        for seg in self.cfg["centerline"].get("segments", []):
            if seg["type"] == "line":
                out.append(Segment("line", self.mm(seg["length"]), 0.0))
            elif seg["type"] == "arc":
                radius = self.mm(seg["radius"])
                angle = math.radians(seg["angle_deg"])
                sign = 1.0 if angle >= 0.0 else -1.0
                out.append(Segment("arc", abs(angle) * radius, sign / radius))
            else:
                raise ValueError(f"unknown segment type: {seg['type']}")
        return out

    def geometries(self) -> list[Geometry]:
        """Return OpenDRIVE geometry records for the configured centerline."""
        if self._geometries is not None:
            return self._geometries

        centerline = self.cfg["centerline"]
        mode = centerline.get("mode")
        if mode == "spline":
            self._geometries = self._spline_geometries()
        elif mode == "polyline" or "points" in centerline:
            self._geometries = self._polyline_geometries()
        else:
            self._geometries = self._segment_geometries()
        return self._geometries

    def _segment_geometries(self) -> list[Geometry]:
        x, y, hdg = self.start_pose
        s = 0.0
        geoms: list[Geometry] = []
        for seg in self.segments():
            geoms.append(Geometry(s, x, y, hdg, seg.length, seg.curvature))
            x, y, hdg = integrate(x, y, hdg, seg.length, seg.curvature)
            s += seg.length
        return geoms

    def _spline_geometries(self) -> list[Geometry]:
        sampled = self._sample_spline_points()
        geoms: list[Geometry] = []
        s = 0.0
        for idx, (x1, y1) in enumerate(sampled):
            x2, y2 = sampled[(idx + 1) % len(sampled)]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 1e-6:
                continue
            hdg = math.atan2(y2 - y1, x2 - x1)
            geoms.append(Geometry(s, x1, y1, hdg, length, 0.0))
            s += length
        return geoms

    def _polyline_points(self) -> list[tuple[float, float]]:
        centerline = self.cfg["centerline"]
        raw = centerline.get("points")
        if not isinstance(raw, list) or len(raw) < 3:
            raise ValueError("polyline centerline needs at least 3 points.")
        pts: list[tuple[float, float]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"invalid polyline point format: index={idx}, item={item!r}")
            pts.append((self.mm(float(item["x"])), self.mm(float(item["y"]))))
        return pts

    def _polyline_geometries(self) -> list[Geometry]:
        pts = self._polyline_points()
        geoms: list[Geometry] = []
        s = 0.0
        for idx, (x1, y1) in enumerate(pts):
            x2, y2 = pts[(idx + 1) % len(pts)]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 1e-6:
                continue
            hdg = math.atan2(y2 - y1, x2 - x1)
            geoms.append(Geometry(s, x1, y1, hdg, length, 0.0))
            s += length
        return geoms

    def _sample_spline_points(self) -> list[tuple[float, float]]:
        centerline = self.cfg["centerline"]
        raw = centerline["control_points"]
        pts = [(self.mm(p["x"]), self.mm(p["y"])) for p in raw]
        if len(pts) < 4:
            raise ValueError("spline centerline needs at least 4 control points.")
        samples_per = int(centerline.get("samples_per_segment", 10))
        tension = float(centerline.get("tension", 0.5))
        sampled: list[tuple[float, float]] = []
        n = len(pts)
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            for j in range(samples_per):
                t = j / samples_per
                sampled.append(catmull_rom(p0, p1, p2, p3, t, tension))
        return sampled

    def total_length(self) -> float:
        return sum(g.length for g in self.geometries())

    # ---- pose lookup --------------------------------------------------
    def reference_pose_at(self, s_m: float) -> TrackPose:
        """Return pose on the reference line at scaled distance s_m."""
        total = self.total_length()
        if total <= 0:
            raise ValueError("track length is zero.")
        remaining = s_m % total
        for geom in self.geometries():
            if remaining <= geom.length:
                x, y, hdg = integrate(geom.x, geom.y, geom.heading, remaining, geom.curvature)
                return TrackPose(geom.s + remaining, x, y, hdg)
            remaining -= geom.length
        last = self.geometries()[-1]
        return TrackPose(total, last.x, last.y, last.heading)

    def pose_with_right_offset(self, s_mm: float, right_offset_m: float) -> TrackPose:
        """Return pose at YAML s(mm), shifted to the right of the reference line."""
        pose = self.reference_pose_at(self.mm(s_mm))
        x = pose.x + right_offset_m * math.sin(pose.heading)
        y = pose.y - right_offset_m * math.cos(pose.heading)
        return TrackPose(pose.s, x, y, pose.heading, pose.lane)

    def lane_center_pose(self, s_mm: float, lane: int) -> TrackPose:
        """Return center pose for 1-based lane number from config/track.yaml."""
        if lane < 1 or lane > self.cfg["lanes"]["count"]:
            raise ValueError(f"lane must be in the range 1~{self.cfg['lanes']['count']} : {lane}")
        pose = self.pose_with_right_offset(s_mm, (lane - 0.5) * self.lane_width)
        pose.lane = lane
        return pose

    def road_center_pose(self, s_mm: float) -> TrackPose:
        """Return pose at the center of the whole road width."""
        return self.pose_with_right_offset(s_mm, self.road_width / 2.0)

    # ---- validation ---------------------------------------------------
    def loop_closure_error(self) -> float:
        """Integrate geometry records and return distance back to the start."""
        geoms = self.geometries()
        if not geoms:
            return 0.0
        x, y, hdg = geoms[0].x, geoms[0].y, geoms[0].heading
        for geom in geoms:
            x, y, hdg = integrate(geom.x, geom.y, geom.heading, geom.length, geom.curvature)
        return math.hypot(x - geoms[0].x, y - geoms[0].y)


def catmull_rom(p0, p1, p2, p3, t: float, tension: float) -> tuple[float, float]:
    """Centripetal-style Catmull-Rom interpolation with configurable tangent scale."""
    t2 = t * t
    t3 = t2 * t
    m1x = tension * (p2[0] - p0[0])
    m1y = tension * (p2[1] - p0[1])
    m2x = tension * (p3[0] - p1[0])
    m2y = tension * (p3[1] - p1[1])
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    x = h00 * p1[0] + h10 * m1x + h01 * p2[0] + h11 * m2x
    y = h00 * p1[1] + h10 * m1y + h01 * p2[1] + h11 * m2y
    return x, y


def integrate(x: float, y: float, hdg: float, length: float, curvature: float):
    """OpenDRIVE geometry integration from (x, y, heading)."""
    if abs(curvature) < 1e-12:
        return (x + length * math.cos(hdg),
                y + length * math.sin(hdg),
                hdg)
    hdg2 = hdg + curvature * length
    x2 = x + (math.sin(hdg2) - math.sin(hdg)) / curvature
    y2 = y - (math.cos(hdg2) - math.cos(hdg)) / curvature
    return x2, y2, hdg2
