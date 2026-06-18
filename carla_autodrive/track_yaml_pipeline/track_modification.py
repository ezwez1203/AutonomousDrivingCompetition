#!/usr/bin/env python3
"""Repair track_v11 point ordering while keeping center-angle diagnostics.

Center-based slope/angle sorting is only safe for star-shaped loops.  The SKKU
track has directions where a ray from the center intersects the track more than
once, so the default output preserves the source polyline order and records
center/slope/angle values only as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

Point = Dict[str, float]
PointTuple = Tuple[float, float]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "track_v11.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "track_v11_cleaned.yaml"
DEFAULT_OVERLAY = SCRIPT_DIR / "track_v11_cleaned_overlay.png"
DEFAULT_CSV = SCRIPT_DIR / "track_v11_cleaned_order.csv"
DEFAULT_TOP_LEFT_SPUR = (2198, 2288)
DEFAULT_BRIDGE_SPACING_MM = 10.0


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=120)


def read_centerline_points(data: Dict[str, Any]) -> List[Point]:
    centerline = data.get("centerline")
    if not isinstance(centerline, dict):
        raise ValueError("YAML does not contain centerline mapping")

    raw = centerline.get("points") or centerline.get("control_points")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("centerline.points/control_points must contain at least 3 points")

    points: List[Point] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            points.append({**item, "x": float(item["x"]), "y": float(item["y"])})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append({"x": float(item[0]), "y": float(item[1])})
        else:
            raise ValueError(f"Unsupported point format at index {idx}: {item!r}")
    return points


def center_of_points(points: Sequence[Point]) -> PointTuple:
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def point_order_records(points: Sequence[Point], center: PointTuple) -> List[Dict[str, Any]]:
    cx, cy = center
    records: List[Dict[str, Any]] = []
    for original_index, point in enumerate(points):
        dx = point["x"] - cx
        dy = point["y"] - cy
        angle = math.atan2(dy, dx)
        if angle < 0.0:
            angle += math.tau
        radius = math.hypot(dx, dy)
        slope = math.inf if abs(dx) < 1e-12 else dy / dx
        records.append(
            {
                "original_index": original_index,
                "point": point,
                "dx": dx,
                "dy": dy,
                "slope": slope,
                "angle_rad": angle,
                "angle_deg": math.degrees(angle),
                "radius": radius,
            }
        )
    return records


def rotate_to_nearest_start(records: Sequence[Dict[str, Any]], start: Point) -> List[Dict[str, Any]]:
    sx, sy = start["x"], start["y"]
    start_idx = min(
        range(len(records)),
        key=lambda i: (records[i]["point"]["x"] - sx) ** 2 + (records[i]["point"]["y"] - sy) ** 2,
    )
    return list(records[start_idx:]) + list(records[:start_idx])


def reorder_points_by_center_angle(
    points: Sequence[Point],
    clockwise: bool = False,
    keep_original_start: bool = True,
) -> Tuple[List[Point], PointTuple, List[Dict[str, Any]]]:
    center = center_of_points(points)
    records = point_order_records(points, center)
    records.sort(key=lambda r: (r["angle_rad"], r["radius"], r["original_index"]), reverse=clockwise)
    if keep_original_start:
        records = rotate_to_nearest_start(records, points[0])
    ordered_points = [dict(record["point"]) for record in records]
    return ordered_points, center, records


def reorder_points_by_nearest_neighbor(
    points: Sequence[Point],
    keep_original_start: bool = True,
    prefer_original_direction: bool = True,
) -> Tuple[List[Point], PointTuple, List[Dict[str, Any]]]:
    center = center_of_points(points)
    records_by_index = point_order_records(points, center)
    pts = np.asarray([(p["x"], p["y"]) for p in points], dtype=np.float64)
    if len(pts) < 3:
        return [dict(p) for p in points], center, records_by_index

    start_idx = 0 if keep_original_start else int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    ordered_indices = [start_idx]
    remaining = set(range(len(pts)))
    remaining.remove(start_idx)

    if prefer_original_direction and keep_original_start and len(pts) > 1 and 1 in remaining:
        ordered_indices.append(1)
        remaining.remove(1)

    while remaining:
        cur = pts[ordered_indices[-1]]
        next_idx = min(
            remaining,
            key=lambda i: (float(np.sum((pts[i] - cur) ** 2)), i),
        )
        ordered_indices.append(next_idx)
        remaining.remove(next_idx)

    records = []
    for order, idx in enumerate(ordered_indices):
        record = dict(records_by_index[idx])
        record["order"] = order
        records.append(record)
    ordered_points = [dict(points[idx]) for idx in ordered_indices]
    return ordered_points, center, records


def closed_segment_lengths(points: Sequence[Point]) -> np.ndarray:
    pts = np.asarray([(p["x"], p["y"]) for p in points], dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    return np.linalg.norm(nxt - pts, axis=1)


def interpolate_bridge(start: Point, end: Point, spacing_mm: float) -> List[Point]:
    distance = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
    if distance <= 1e-9:
        return []
    steps = max(1, int(math.ceil(distance / max(spacing_mm, 1e-9))))
    bridge: List[Point] = []
    for i in range(1, steps):
        t = i / steps
        bridge.append(
            {
                "x": start["x"] * (1.0 - t) + end["x"] * t,
                "y": start["y"] * (1.0 - t) + end["y"] * t,
            }
        )
    return bridge


def remove_point_range_with_bridge(
    points: Sequence[Point],
    start_idx: int,
    end_idx: int,
    spacing_mm: float,
) -> List[Point]:
    if start_idx <= 0 or end_idx >= len(points) - 1 or start_idx > end_idx:
        raise ValueError(f"invalid removal range: {start_idx}..{end_idx} for {len(points)} points")
    before = [dict(p) for p in points[:start_idx]]
    after = [dict(p) for p in points[end_idx + 1 :]]
    bridge = interpolate_bridge(points[start_idx - 1], points[end_idx + 1], spacing_mm)
    return before + bridge + after


def update_centerline_stats(
    data: Dict[str, Any],
    points: Sequence[Point],
    center: PointTuple,
    method: str,
    removed_ranges: Sequence[Tuple[int, int]] = (),
) -> None:
    centerline = data["centerline"]
    lengths = closed_segment_lengths(points)
    centerline["mode"] = "polyline"
    centerline["point_count"] = len(points)
    centerline["estimated_length_mm"] = round(float(lengths.sum()), 2)
    centerline["mean_spacing_mm"] = round(float(lengths.mean()), 3)
    centerline["max_spacing_mm"] = round(float(lengths.max()), 3)
    ordering = {
        "method": method,
        "center_mm": [round(center[0], 3), round(center[1], 3)],
        "connection_key": {
            "original_yaml_order": "source centerline point sequence",
            "original_yaml_order_top_left_spur_removed": "source sequence with top-left spur replaced by a linear bridge",
            "nearest_neighbor_continuity": "nearest unvisited Euclidean neighbor",
            "center_polar_angle_sort": "atan2(y - center_y, x - center_x)",
        }.get(method, method),
        "slope_note": "dy/dx and atan2 are diagnostics only for cleaned output; center-angle sorting can cross branches on non-star-shaped loops.",
    }
    if removed_ranges:
        ordering["removed_ranges_original_indices"] = [[int(a), int(b)] for a, b in removed_ranges]
        ordering["bridge_spacing_mm"] = DEFAULT_BRIDGE_SPACING_MM
    centerline["ordering"] = ordering
    centerline["points"] = [{"x": round(p["x"], 1), "y": round(p["y"], 1)} for p in points]


def update_meta(data: Dict[str, Any], input_path: Path, output_path: Path, method: str) -> None:
    meta = data.setdefault("meta", {})
    if not isinstance(meta, dict):
        data["meta"] = meta = {}
    meta["modified_by"] = "track_modification.py"
    meta["modified_at"] = dt.datetime.now().isoformat(timespec="seconds")
    meta["modification_method"] = method
    meta["source_yaml"] = str(input_path)
    meta["output_yaml"] = str(output_path)


def write_order_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "original_index", "x", "y", "dx", "dy", "slope", "angle_deg", "radius"])
        for order, record in enumerate(records):
            point = record["point"]
            slope = record["slope"]
            writer.writerow(
                [
                    order,
                    record["original_index"],
                    round(point["x"], 3),
                    round(point["y"], 3),
                    round(record["dx"], 6),
                    round(record["dy"], 6),
                    "inf" if math.isinf(slope) else round(float(slope), 9),
                    round(record["angle_deg"], 6),
                    round(record["radius"], 6),
                ]
            )


def image_frame_from_yaml(data: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    calibration = data.get("calibration", {})
    dimensions = data.get("dimensions", {})
    frame = calibration.get("image_frame_px")
    field = dimensions.get("field_mm")
    if not (
        isinstance(frame, list)
        and len(frame) == 4
        and isinstance(field, list)
        and len(field) == 2
    ):
        raise ValueError("YAML needs calibration.image_frame_px and dimensions.field_mm to draw an image overlay")
    return (
        float(frame[0]),
        float(frame[1]),
        float(frame[2]),
        float(frame[3]),
        float(field[0]),
        float(field[1]),
    )


def mm_to_px(point: Point, frame: Tuple[float, float, float, float, float, float]) -> Tuple[int, int]:
    x0, y0, x1, y1, width_mm, height_mm = frame
    sx = width_mm / max(1e-9, x1 - x0)
    sy = height_mm / max(1e-9, y1 - y0)
    x = x0 + point["x"] / sx
    y = y0 + point["y"] / sy
    return int(round(x)), int(round(y))


def draw_overlay(path: Path, data: Dict[str, Any], points: Sequence[Point], center: PointTuple) -> None:
    from PIL import Image, ImageDraw

    frame = image_frame_from_yaml(data)
    x0, y0, x1, y1, _, _ = frame
    width = max(int(math.ceil(x1 + 40)), 100)
    height = max(int(math.ceil(y1 + 40)), 100)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    poly = [mm_to_px(p, frame) for p in points]
    draw.rectangle((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))), outline=(180, 180, 180), width=2)
    if len(poly) >= 2:
        draw.line(poly + [poly[0]], fill=(0, 60, 255), width=2, joint="curve")

    center_point = {"x": center[0], "y": center[1]}
    cx, cy = mm_to_px(center_point, frame)
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 0, 0))

    for idx in np.linspace(0, len(points) - 1, min(36, len(points)), dtype=int):
        px, py = mm_to_px(points[int(idx)], frame)
        draw.line((cx, cy, px, py), fill=(210, 210, 210), width=1)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(0, 160, 0))

    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair track_v11 centerline ordering and write a corrected YAML."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input track YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output sorted track YAML")
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY, help="Output overlay PNG")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Output ordering diagnostics CSV")
    parser.add_argument("--method", choices=("original", "nearest", "angle"), default="original", help="Connection ordering method")
    parser.add_argument("--keep-top-left-spur", action="store_true", help="Do not remove the known track_v11 top-left spur")
    parser.add_argument("--spur-range", default=f"{DEFAULT_TOP_LEFT_SPUR[0]},{DEFAULT_TOP_LEFT_SPUR[1]}", help="Inclusive original point index range to remove, e.g. 2198,2288")
    parser.add_argument("--bridge-spacing-mm", type=float, default=DEFAULT_BRIDGE_SPACING_MM, help="Spacing for the replacement bridge")
    parser.add_argument("--clockwise", action="store_true", help="Reverse angular order when --method angle is used")
    parser.add_argument("--no-keep-start", action="store_true", help="Do not keep the original first point as the start")
    parser.add_argument("--no-prefer-original-direction", action="store_true", help="Do not force original point 1 as the second point for nearest ordering")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay PNG generation")
    parser.add_argument("--no-csv", action="store_true", help="Skip ordering CSV generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_yaml(args.input)
    original_points = read_centerline_points(data)
    removed_ranges: List[Tuple[int, int]] = []
    if args.method == "angle":
        method_name = "center_polar_angle_sort"
        ordered_points, center, records = reorder_points_by_center_angle(
            original_points,
            clockwise=args.clockwise,
            keep_original_start=not args.no_keep_start,
        )
    elif args.method == "original":
        method_name = "original_yaml_order"
        ordered_points = [dict(p) for p in original_points]
        if not args.keep_top_left_spur:
            start_idx, end_idx = [int(v.strip()) for v in args.spur_range.split(",")]
            ordered_points = remove_point_range_with_bridge(
                ordered_points,
                start_idx,
                end_idx,
                args.bridge_spacing_mm,
            )
            method_name = "original_yaml_order_top_left_spur_removed"
            removed_ranges.append((start_idx, end_idx))
        center = center_of_points(ordered_points)
        records = point_order_records(ordered_points, center)
    else:
        method_name = "nearest_neighbor_continuity"
        ordered_points, center, records = reorder_points_by_nearest_neighbor(
            original_points,
            keep_original_start=not args.no_keep_start,
            prefer_original_direction=not args.no_prefer_original_direction,
        )

    update_centerline_stats(data, ordered_points, center, method_name, removed_ranges)
    update_meta(data, args.input, args.output, method_name)
    write_yaml(args.output, data)

    if not args.no_csv:
        write_order_csv(args.csv, records)
    if not args.no_overlay:
        draw_overlay(args.overlay, data, ordered_points, center)

    print(f"method={method_name}")
    if removed_ranges:
        print(f"removed_ranges={removed_ranges}")
    print(f"center=({center[0]:.3f}, {center[1]:.3f})")
    print(f"points={len(ordered_points)}")
    print(f"wrote {args.output}")
    if not args.no_csv:
        print(f"wrote {args.csv}")
    if not args.no_overlay:
        print(f"wrote {args.overlay}")


if __name__ == "__main__":
    main()
