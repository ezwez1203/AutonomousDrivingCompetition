#!/usr/bin/env python
"""Measure generated track shape against a blueprint image.

The metric compares the current ``config/track.yaml`` centerline against colored
pixels extracted from ``circuit_blueprint.png``. It is an image-based estimate,
not a CAD-grade validation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from carla_autodrive.maps import TrackSpec
from carla_autodrive.utils.config import load_config
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMAGE = ROOT / "circuit_blueprint.png"
DEFAULT_OUT = ROOT / "carla_autodrive" / "reports"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Measure track.yaml geometry against the blueprint image")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--track-yaml", type=Path, default=None, help="Path to the track.yaml to evaluate. Defaults to config/track.yaml.")
    ap.add_argument("--mask", choices=("green", "red", "dark", "nonwhite"), default="green",
                    help="Color candidate to extract as the blueprint reference line.")
    ap.add_argument("--sample-step-mm", type=float, default=100.0,
                    help="track.yaml centerline sample spacing.")
    ap.add_argument("--tolerance-mm", type=float, default=None,
                    help="Allowed accuracy tolerance. Defaults to half the road width.")
    ap.add_argument("--component", choices=("all", "largest"), default="all",
                    help="Choose whether to use all mask pixels or only the largest connected component.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overlay-name", default="track_accuracy_overlay.png")
    ap.add_argument("--json-name", default="track_accuracy_report.json")
    return ap.parse_args()


def mask_pixels(rgb: np.ndarray, mode: str) -> np.ndarray:
    arr = rgb.astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if mode == "green":
        return (g > 120) & (g - r > 25) & (g - b > 15)
    if mode == "red":
        return (r > 120) & (r - g > 35) & (r - b > 35)
    if mode == "dark":
        return (r < 90) & (g < 90) & (b < 90)
    return (r < 245) | (g < 245) | (b < 245)


def filter_component(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "all":
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labels == largest


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No pixels were extracted from the selected mask.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def track_samples(spec: TrackSpec, step_mm: float) -> np.ndarray:
    total_mm = spec.total_length() / spec.scale * 1000.0
    count = max(32, int(math.ceil(total_mm / step_mm)))
    samples = []
    for i in range(count):
        s_mm = total_mm * i / count
        pose = spec.reference_pose_at(spec.mm(s_mm))
        x_mm = pose.x / spec.scale * 1000.0
        y_mm = pose.y / spec.scale * 1000.0
        samples.append((x_mm, y_mm))
    return np.asarray(samples, dtype=np.float64)


def project_mm_to_px(points_mm: np.ndarray, field_mm: tuple[float, float],
                     bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    field_w, field_h = field_mm
    sx = (x1 - x0) / field_w
    sy = (y1 - y0) / field_h
    px = np.empty_like(points_mm)
    px[:, 0] = x0 + points_mm[:, 0] * sx
    px[:, 1] = y0 + points_mm[:, 1] * sy
    return px, (sx, sy)


def distances(track_px: np.ndarray, mask: np.ndarray, max_mask_points: int = 20000):
    ys, xs = np.where(mask)
    mask_px = np.column_stack([xs, ys]).astype(np.float64)
    if len(mask_px) > max_mask_points:
        rng = np.random.default_rng(7)
        mask_px = mask_px[rng.choice(len(mask_px), size=max_mask_points, replace=False)]

    mask_tree = cKDTree(mask_px)
    track_to_mask, _ = mask_tree.query(track_px, k=1)

    track_tree = cKDTree(track_px)
    mask_to_track, _ = track_tree.query(mask_px, k=1)
    return track_to_mask, mask_to_track, mask_px


def summarize(values_px: np.ndarray, px_to_mm: float, tolerance_px: float) -> dict:
    return {
        "mean_px": float(np.mean(values_px)),
        "median_px": float(np.median(values_px)),
        "p95_px": float(np.percentile(values_px, 95)),
        "max_px": float(np.max(values_px)),
        "mean_mm": float(np.mean(values_px) * px_to_mm),
        "median_mm": float(np.median(values_px) * px_to_mm),
        "p95_mm": float(np.percentile(values_px, 95) * px_to_mm),
        "within_tolerance_percent": float(np.mean(values_px <= tolerance_px) * 100.0),
    }


def draw_overlay(image: Image.Image, mask: np.ndarray, track_px: np.ndarray,
                 bbox: tuple[int, int, int, int], out_path: Path) -> None:
    overlay = image.convert("RGBA")
    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_arr = np.asarray(mask_layer).copy()
    mask_arr[mask] = (0, 180, 0, 110)
    overlay = Image.alpha_composite(overlay, Image.fromarray(mask_arr))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(bbox, outline=(0, 80, 255, 255), width=4)
    pts = [(float(x), float(y)) for x, y in track_px]
    if len(pts) > 1:
        draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=6)
    for x, y in pts[::max(1, len(pts) // 80)]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(out_path)


def main() -> int:
    args = parse_args()
    if args.track_yaml is None:
        spec = TrackSpec()
    else:
        with args.track_yaml.open("r", encoding="utf-8") as f:
            spec = TrackSpec(yaml.safe_load(f))
    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image)
    raw_mask = mask_pixels(rgb, args.mask)
    mask = filter_component(raw_mask, args.component)
    bbox = bbox_from_mask(mask)

    field = spec.cfg["dimensions"]["field_mm"]
    field_mm = (float(field[0]), float(field[1]))
    samples_mm = track_samples(spec, args.sample_step_mm)
    track_px, scales = project_mm_to_px(samples_mm, field_mm, bbox)

    track_to_mask, mask_to_track, mask_px = distances(track_px, mask)
    px_to_mm = 1.0 / ((scales[0] + scales[1]) / 2.0)
    tolerance_mm = args.tolerance_mm if args.tolerance_mm is not None else spec.cfg["dimensions"]["road_width_mm"] / 2.0
    tolerance_px = tolerance_mm / px_to_mm

    track_summary = summarize(track_to_mask, px_to_mm, tolerance_px)
    mask_summary = summarize(mask_to_track, px_to_mm, tolerance_px)
    bidirectional_mean_mm = (track_summary["mean_mm"] + mask_summary["mean_mm"]) / 2.0
    bidirectional_within = (track_summary["within_tolerance_percent"] + mask_summary["within_tolerance_percent"]) / 2.0

    report = {
        "image": str(args.image),
        "track_yaml": str(args.track_yaml) if args.track_yaml is not None else "config/track.yaml",
        "mask": args.mask,
        "component": args.component,
        "image_size_px": list(image.size),
        "raw_mask_pixels": int(raw_mask.sum()),
        "mask_pixels": int(mask.sum()),
        "mask_bbox_px": list(bbox),
        "field_mm": list(field_mm),
        "track_total_length_m": spec.total_length(),
        "track_samples": int(len(track_px)),
        "px_per_mm": {"x": scales[0], "y": scales[1], "average": (scales[0] + scales[1]) / 2.0},
        "tolerance_mm": float(tolerance_mm),
        "tolerance_px": float(tolerance_px),
        "track_to_blueprint": track_summary,
        "blueprint_to_track": mask_summary,
        "estimated_shape_accuracy_percent": float(bidirectional_within),
        "bidirectional_mean_error_mm": float(bidirectional_mean_mm),
        "notes": [
            "Accuracy is based on nearest-pixel distance after fitting track.yaml field_mm to the extracted mask bbox.",
            "Use --mask green/red/dark/nonwhite if the selected blueprint layer is not the intended reference line.",
            "This is not CAD validation; final judgement should use official CAD/DXF coordinates when available.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / args.json_name
    overlay_path = args.out_dir / args.overlay_name
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_overlay(image, mask, track_px, bbox, overlay_path)

    print(f"image={args.image}")
    print(f"mask={args.mask} component={args.component} raw_pixels={report['raw_mask_pixels']} pixels={report['mask_pixels']} bbox={report['mask_bbox_px']}")
    print(f"track_samples={len(track_px)} tolerance={tolerance_mm:.1f}mm ({tolerance_px:.1f}px)")
    print("track->blueprint mean={mean_mm:.1f}mm median={median_mm:.1f}mm p95={p95_mm:.1f}mm within={within_tolerance_percent:.1f}%".format(**track_summary))
    print("blueprint->track mean={mean_mm:.1f}mm median={median_mm:.1f}mm p95={p95_mm:.1f}mm within={within_tolerance_percent:.1f}%".format(**mask_summary))
    print(f"estimated_shape_accuracy={bidirectional_within:.1f}%")
    print(f"bidirectional_mean_error={bidirectional_mean_mm:.1f}mm")
    print(f"wrote {json_path}")
    print(f"wrote {overlay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
