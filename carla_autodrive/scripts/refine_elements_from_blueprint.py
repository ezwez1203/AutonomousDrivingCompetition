#!/usr/bin/env python
"""Refine mission element s/lane positions from blueprint marker pixels.

The dense track polyline is already image-derived.  This helper uses the saved
``source_image_px`` hints under ``elements`` and snaps each hint to the nearest
visible green blueprint marker component before projecting it onto the closed
centerline by segment arclength.  The result is a more consistent ``s`` value
than hand-maintained YAML numbers or nearest-vertex projection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMAGE = ROOT / "circuit_blueprint.png"
DEFAULT_TRACK = ROOT / "carla_autodrive" / "config" / "track.yaml"
DEFAULT_REPORT = ROOT / "carla_autodrive" / "reports" / "element_refine_report.json"
DEFAULT_OVERLAY = ROOT / "carla_autodrive" / "reports" / "element_refine_overlay.png"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine track.yaml element s positions from blueprint marker pixels")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--track-yaml", type=Path, default=DEFAULT_TRACK)
    ap.add_argument("--out-yaml", type=Path, default=DEFAULT_TRACK)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    ap.add_argument("--apply", action="store_true", help="write refined values to --out-yaml")
    ap.add_argument("--snap-radius-px", type=float, default=80.0)
    ap.add_argument("--min-component-area", type=int, default=20)
    return ap.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=120)


def image_frame(data: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    cal = data.get("calibration", {})
    dims = data.get("dimensions", {})
    frame = cal.get("image_frame_px")
    field = dims.get("field_mm")
    if not frame or not field:
        raise ValueError("track.yaml needs calibration.image_frame_px and dimensions.field_mm")
    x0, y0, x1, y1 = [float(v) for v in frame]
    width_mm, height_mm = [float(v) for v in field]
    sx = width_mm / max(1e-9, x1 - x0)
    sy = height_mm / max(1e-9, y1 - y0)
    return x0, y0, sx, sy, width_mm, height_mm


def px_to_mm(px: tuple[float, float], frame: tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    x0, y0, sx, sy, _width, _height = frame
    return (float((px[0] - x0) * sx), float((px[1] - y0) * sy))


def mm_to_px(mm: tuple[float, float], frame: tuple[float, float, float, float, float, float]) -> tuple[int, int]:
    x0, y0, sx, sy, _width, _height = frame
    return int(round(x0 + mm[0] / sx)), int(round(y0 + mm[1] / sy))


def centerline_points(data: dict[str, Any]) -> np.ndarray:
    raw = data.get("centerline", {}).get("points")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("refinement currently expects centerline.points polyline")
    return np.asarray([(float(p["x"]), float(p["y"])) for p in raw], dtype=np.float64)


def green_components(bgr: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 80)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask * 255, 8)
    out: list[dict[str, Any]] = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = float(cents[idx][0]), float(cents[idx][1])
        out.append({"area": area, "bbox": [x, y, w, h], "centroid": [cx, cy]})
    return out


def snap_to_component(seed_px: tuple[float, float], comps: list[dict[str, Any]], radius_px: float) -> tuple[float, float, dict[str, Any] | None]:
    sx, sy = seed_px
    best = None
    best_dist = float("inf")
    for comp in comps:
        cx, cy = comp["centroid"]
        x, y, w, h = comp["bbox"]
        # Distance to component bbox is stable for long marker lines; centroid alone can be far from a numbered tick.
        dx = max(float(x) - sx, 0.0, sx - float(x + w - 1))
        dy = max(float(y) - sy, 0.0, sy - float(y + h - 1))
        bbox_dist = math.hypot(dx, dy)
        centroid_dist = math.hypot(cx - sx, cy - sy)
        dist = min(bbox_dist, centroid_dist)
        if dist < best_dist:
            best_dist = dist
            best = comp
    if best is None or best_dist > radius_px:
        return sx, sy, None
    cx, cy = best["centroid"]
    return float(cx), float(cy), best


def project_point_to_centerline(point_mm: tuple[float, float], points: np.ndarray) -> dict[str, float]:
    p = np.asarray(point_mm, dtype=np.float64)
    a = points
    b = np.roll(points, -1, axis=0)
    ab = b - a
    seg_len2 = np.sum(ab * ab, axis=1)
    ap = p[None, :] - a
    t = np.clip(np.sum(ap * ab, axis=1) / np.maximum(seg_len2, 1e-9), 0.0, 1.0)
    proj = a + ab * t[:, None]
    d2 = np.sum((proj - p[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    seg_lengths = np.sqrt(seg_len2)
    s = float(np.sum(seg_lengths[:idx]) + seg_lengths[idx] * t[idx])
    best = proj[idx]
    heading = math.atan2(float(ab[idx, 1]), float(ab[idx, 0]))
    # Positive right offset follows TrackSpec.pose_with_right_offset convention in image/mm coordinates.
    right = np.asarray([math.sin(heading), -math.cos(heading)], dtype=np.float64)
    right_offset = float(np.dot(p - best, right))
    return {
        "s": s,
        "distance_mm": float(math.sqrt(d2[idx])),
        "right_offset_mm": right_offset,
        "projected_x": float(best[0]),
        "projected_y": float(best[1]),
        "heading_rad": heading,
    }


def iter_element_items(elements: dict[str, Any]):
    for key, value in elements.items():
        if isinstance(value, dict):
            if "source_image_px" in value and "s" in value:
                yield (key,), value
            for subkey in ("obstacle_2_presets", "obstacle_3_presets"):
                pass
        if key in ("obstacle_2_presets", "obstacle_3_presets") and isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict) and "source_image_px" in item and "s" in item:
                    yield (key, idx), item


def update_element_item(item: dict[str, Any], projected: dict[str, float], snapped_px: tuple[float, float], comp: dict[str, Any] | None) -> dict[str, Any]:
    old_s = float(item["s"])
    item["s"] = int(round(projected["s"]))
    item["source_image_px_refined"] = [round(float(snapped_px[0]), 1), round(float(snapped_px[1]), 1)]
    item["projection_error_mm"] = round(projected["distance_mm"], 1)
    if comp is not None:
        item["source_component_bbox_px"] = list(comp["bbox"])
    return {
        "old_s": old_s,
        "new_s": float(item["s"]),
        "delta_s": float(item["s"] - old_s),
        "projection_error_mm": projected["distance_mm"],
        "right_offset_mm": projected["right_offset_mm"],
        "refined_px": item["source_image_px_refined"],
    }


def draw_overlay(bgr: np.ndarray, data: dict[str, Any], reports: list[dict[str, Any]], frame, out_path: Path) -> None:
    overlay = bgr.copy()
    pts = centerline_points(data)
    poly = np.asarray([mm_to_px((float(x), float(y)), frame) for x, y in pts], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [poly], True, (255, 0, 0), 2, cv2.LINE_AA)
    for rec in reports:
        sx, sy = [int(round(v)) for v in rec["seed_px"]]
        rx, ry = [int(round(v)) for v in rec["refined_px"]]
        cv2.circle(overlay, (sx, sy), 10, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (rx, ry), 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(overlay, (sx, sy), (rx, ry), (0, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, rec["name"], (rx + 8, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 90, 0), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main() -> int:
    args = parse_args()
    data = load_yaml(args.track_yaml)
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    frame = image_frame(data)
    points = centerline_points(data)
    comps = green_components(bgr, args.min_component_area)
    elements = data.get("elements", {})
    if not isinstance(elements, dict):
        raise ValueError("track.yaml elements must be a mapping")

    reports: list[dict[str, Any]] = []
    for path, item in iter_element_items(elements):
        seed_px_raw = item.get("source_image_px")
        if not isinstance(seed_px_raw, list) or len(seed_px_raw) < 2:
            continue
        seed_px = (float(seed_px_raw[0]), float(seed_px_raw[1]))
        rx, ry, comp = snap_to_component(seed_px, comps, args.snap_radius_px)
        point_mm = px_to_mm((rx, ry), frame)
        projected = project_point_to_centerline(point_mm, points)
        result = update_element_item(item, projected, (rx, ry), comp)
        name = path[0] if len(path) == 1 else f"{path[0]}[{path[1]}]"
        reports.append({"name": name, "seed_px": [seed_px[0], seed_px[1]], **result})

    data.setdefault("meta", {})["elements_refined_by"] = "refine_elements_from_blueprint.py"
    data["meta"]["elements_refine_source_image"] = str(args.image)
    payload = {
        "image": str(args.image),
        "track_yaml": str(args.track_yaml),
        "applied": bool(args.apply),
        "green_components": len(comps),
        "items": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_overlay(bgr, data, reports, frame, args.overlay)
    if args.apply:
        write_yaml(args.out_yaml, data)
    print(json.dumps({"applied": bool(args.apply), "items": len(reports), "report": str(args.report), "overlay": str(args.overlay)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
