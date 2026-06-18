#!/usr/bin/env python3
"""
image_to_track_yaml.py

CAD/JPG/PNG track drawing -> dense track.yaml generator.

Core idea
---------
1. Segment the target boundary line from the image.  Default is the red/pink track line.
2. Build an initial coarse path at roughly `coarse_spacing_mm` spacing.
   - Preferred: read a seed YAML containing centerline.control_points or centerline.points.
   - Fallback: pick a large red contour from the image.
3. Recursively refine each segment:
   - For endpoints A,B, define a local oriented search rectangle centered at the Euclidean midpoint.
   - Search target-line pixels inside that rectangle.
   - Insert the target-line pixel closest to the half-distance midpoint.
   - Repeat until the segment length is <= target_spacing_mm.
4. Resample the refined polyline at target_spacing_mm and write YAML.

This is not a neural-network trainer.  It is a deterministic image-registration / active-contour style
pipeline that 'learns' the track geometry from one drawing.  If you later collect multiple corrected
masks, the `segment_target_line()` function is the part to replace with a trained segmentation model.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

Point = Tuple[float, float]


@dataclass
class ImageFrame:
    """Mapping between image pixels and field coordinates in millimeters."""

    x0: float
    y0: float
    x1: float
    y1: float
    width_mm: float
    height_mm: float

    @property
    def sx(self) -> float:
        return self.width_mm / max(1e-9, self.x1 - self.x0)

    @property
    def sy(self) -> float:
        return self.height_mm / max(1e-9, self.y1 - self.y0)

    @property
    def mean_mm_per_px(self) -> float:
        return (abs(self.sx) + abs(self.sy)) * 0.5

    def px_to_mm(self, p: Point) -> Point:
        x, y = p
        return ((x - self.x0) * self.sx, (y - self.y0) * self.sy)

    def mm_to_px(self, p: Point) -> Point:
        x, y = p
        return (self.x0 + x / self.sx, self.y0 + y / self.sy)


@dataclass
class PipelineConfig:
    image_path: str
    output_yaml: str
    output_overlay: str
    seed_yaml: Optional[str]
    field_width_mm: float
    field_height_mm: float
    road_width_mm: float
    target_spacing_mm: float = 10.0
    coarse_spacing_mm: float = 1000.0
    reference: str = "inner"  # inner | outer | largest | smallest
    frame_rect_px: Optional[Tuple[float, float, float, float]] = None
    hsv_sat_min: int = 6
    hsv_val_min: int = 80
    red_delta: int = 4
    mask_dilate_px: int = 3
    search_width_factor: float = 1.0
    snap_radius_mm: float = 80.0
    smoothing_passes: int = 1
    keep_elements_from_seed: bool = True
    output_precision: int = 1
    skip_refinement: bool = False


# -----------------------------------------------------------------------------
# Basic geometry
# -----------------------------------------------------------------------------


def _as_array(points: Sequence[Point]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def polyline_lengths(points: Sequence[Point], closed: bool = True) -> np.ndarray:
    pts = _as_array(points)
    if len(pts) < 2:
        return np.array([], dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0) if closed else pts[1:]
    cur = pts if closed else pts[:-1]
    return np.linalg.norm(nxt - cur, axis=1)


def polyline_length(points: Sequence[Point], closed: bool = True) -> float:
    return float(polyline_lengths(points, closed=closed).sum())


def order_points_by_nearest_neighbor(points: Sequence[Point]) -> List[Point]:
    """Order points by greedily connecting the nearest unvisited Euclidean neighbor.

    This is used for coarse overlay/control points when contour extraction or a
    seed file produced the right point set but the point order is tangled.  The
    first input point is kept as the start so generated YAML remains stable.
    """
    pts = _as_array(points)
    n = len(pts)
    if n < 3:
        return [(float(x), float(y)) for x, y in pts]

    ordered: List[int] = [0]
    remaining = set(range(1, n))
    while remaining:
        cur = pts[ordered[-1]]
        next_idx = min(
            remaining,
            key=lambda i: (float(np.sum((pts[i] - cur) ** 2)), i),
        )
        ordered.append(next_idx)
        remaining.remove(next_idx)

    return [(float(pts[i, 0]), float(pts[i, 1])) for i in ordered]


def resample_polyline(points: Sequence[Point], spacing: float, closed: bool = True) -> List[Point]:
    """Uniformly resample a polyline by arclength."""
    pts = _as_array(points)
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [(float(pts[0, 0]), float(pts[0, 1]))]

    if closed:
        work = np.vstack([pts, pts[0]])
    else:
        work = pts.copy()

    seg = np.linalg.norm(work[1:] - work[:-1], axis=1)
    good = seg > 1e-9
    work0 = work[:-1][good]
    work1 = work[1:][good]
    seg = seg[good]
    if len(seg) == 0:
        return [(float(pts[0, 0]), float(pts[0, 1]))]

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= spacing:
        return [(float(x), float(y)) for x, y in pts]

    targets = np.arange(0.0, total, spacing)
    out: List[Point] = []
    j = 0
    for s in targets:
        while j < len(seg) - 1 and cum[j + 1] < s:
            j += 1
        t = (s - cum[j]) / max(seg[j], 1e-9)
        p = (1.0 - t) * work0[j] + t * work1[j]
        out.append((float(p[0]), float(p[1])))
    return out


def chaikin_smooth(points: Sequence[Point], passes: int = 1, closed: bool = True) -> List[Point]:
    """Very light corner-cutting smoother.  Kept conservative because CAD lines should remain faithful."""
    pts = _as_array(points)
    if passes <= 0 or len(pts) < 4:
        return [(float(x), float(y)) for x, y in pts]
    for _ in range(passes):
        new: List[np.ndarray] = []
        n = len(pts)
        rng = range(n) if closed else range(n - 1)
        if not closed:
            new.append(pts[0])
        for i in rng:
            p = pts[i]
            q = pts[(i + 1) % n]
            new.append(0.75 * p + 0.25 * q)
            new.append(0.25 * p + 0.75 * q)
        if not closed:
            new.append(pts[-1])
        pts = np.asarray(new, dtype=np.float64)
    return [(float(x), float(y)) for x, y in pts]


def catmull_rom_closed(points: Sequence[Point], samples_per_segment: int = 24, alpha: float = 0.5) -> List[Point]:
    """Closed Catmull-Rom interpolation used only to densify sparse seed control points."""
    pts = _as_array(points)
    n = len(pts)
    if n < 4:
        return [(float(x), float(y)) for x, y in pts]

    def tj(ti: float, pi: np.ndarray, pj: np.ndarray) -> float:
        return ti + float(np.linalg.norm(pj - pi) ** alpha)

    out: List[Point] = []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        # avoid equal parameter values
        if min(t1 - t0, t2 - t1, t3 - t2) <= 1e-9:
            continue
        for t in np.linspace(t1, t2, samples_per_segment, endpoint=False):
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
            out.append((float(C[0]), float(C[1])))
    return out


# -----------------------------------------------------------------------------
# Image segmentation / calibration
# -----------------------------------------------------------------------------


def segment_target_line(bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Segment pale red/pink CAD track boundary lines."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # HSV red wraps around 0/180.  The CAD line is pale, so saturation threshold must be low.
    red_hue = (hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 165)
    red_rgb_bias = (r.astype(np.int16) > g.astype(np.int16) + cfg.red_delta) & (
        r.astype(np.int16) > b.astype(np.int16) + cfg.red_delta
    )
    mask = red_hue & (hsv[:, :, 1] >= cfg.hsv_sat_min) & (hsv[:, :, 2] >= cfg.hsv_val_min) & red_rgb_bias
    mask = mask.astype(np.uint8) * 255

    if cfg.mask_dilate_px > 0:
        k = int(max(1, cfg.mask_dilate_px))
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def auto_detect_frame(bgr: np.ndarray, cfg: PipelineConfig, target_mask: Optional[np.ndarray] = None) -> ImageFrame:
    """Detect a useful image-to-mm frame.  The CLI can override this with --frame-rect-px."""
    h, w = bgr.shape[:2]
    if cfg.frame_rect_px is not None:
        x0, y0, x1, y1 = cfg.frame_rect_px
        return ImageFrame(x0, y0, x1, y1, cfg.field_width_mm, cfg.field_height_mm)

    # Robust default: use the bounding box of all non-white drawing content, expanded slightly.
    # This works on exported CAD screenshots where margins are white.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    nonwhite = (gray < 247) | (hsv[:, :, 1] > 12)

    if target_mask is not None and cv2.countNonZero(target_mask) > 0:
        # Include target mask strongly, but keep frame based on full drawing so mm mapping is stable.
        nonwhite = nonwhite | (target_mask > 0)

    ys, xs = np.where(nonwhite)
    if len(xs) < 100:
        pad = 0
        return ImageFrame(pad, pad, w - 1 - pad, h - 1 - pad, cfg.field_width_mm, cfg.field_height_mm)

    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])

    # Try to prevent dimension text far outside the CAD frame from dominating.
    # Use red bbox if it is available and choose a padded box around it when it is more conservative.
    if target_mask is not None and cv2.countNonZero(target_mask) > 0:
        rys, rxs = np.where(target_mask > 0)
        rx0, rx1 = float(np.min(rxs)), float(np.max(rxs))
        ry0, ry1 = float(np.min(rys)), float(np.max(rys))
        pad_x = 0.18 * (rx1 - rx0)
        pad_y = 0.18 * (ry1 - ry0)
        bx0 = max(0.0, rx0 - pad_x)
        bx1 = min(float(w - 1), rx1 + pad_x)
        by0 = max(0.0, ry0 - pad_y)
        by1 = min(float(h - 1), ry1 + pad_y)
        # Use the tighter of the two boxes while ensuring target line remains inside.
        x0, x1 = max(0.0, min(x0, bx0)), min(float(w - 1), max(x1, bx1))
        y0, y1 = max(0.0, min(y0, by0)), min(float(h - 1), max(y1, by1))

    return ImageFrame(float(x0), float(y0), float(x1), float(y1), cfg.field_width_mm, cfg.field_height_mm)


# -----------------------------------------------------------------------------
# Seeds
# -----------------------------------------------------------------------------


def load_seed_yaml_points(path: str) -> Tuple[List[Point], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cl = data.get("centerline", {}) if isinstance(data, dict) else {}
    raw = cl.get("points") or cl.get("polyline") or cl.get("control_points") or []
    points: List[Point] = []
    for p in raw:
        if isinstance(p, dict):
            points.append((float(p["x"]), float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            points.append((float(p[0]), float(p[1])))
    return points, data


def choose_contour(mask: np.ndarray, reference: str = "inner") -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for c in contours:
        per = cv2.arcLength(c, True)
        area = abs(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        if per > 0.20 * min(mask.shape[:2]) and w > 0.10 * mask.shape[1] and h > 0.10 * mask.shape[0]:
            candidates.append((area, per, c))
    if not candidates:
        raise RuntimeError("No large red/pink contour found. Try lowering --hsv-sat-min or set --seed-yaml.")
    if reference in ("inner", "smallest"):
        candidates.sort(key=lambda t: t[0])
    elif reference in ("outer", "largest"):
        candidates.sort(key=lambda t: t[0], reverse=True)
    else:
        candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[0][2]


def auto_seed_from_contour(mask: np.ndarray, frame: ImageFrame, coarse_spacing_mm: float, reference: str) -> List[Point]:
    c = choose_contour(mask, reference=reference)
    pts_px = c.reshape(-1, 2).astype(np.float64)

    # IMPORTANT: a thick CAD line contour is the perimeter of the line band.
    # If we use the full contour, the generated path goes around both sides of the band
    # and the length becomes roughly doubled.  Splitting the ordered contour in half gives
    # one side of the red line band; the other half is the parallel side.  Either side is
    # only a few pixels away from the visible center, and the recursive snap stage pulls
    # it back to target-line pixels.
    if len(pts_px) >= 20:
        closed = np.vstack([pts_px, pts_px[0]])
        seg = np.linalg.norm(closed[1:] - closed[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        half_s = cum[-1] * 0.5
        half_idx = int(np.searchsorted(cum, half_s))
        half_idx = max(2, min(len(pts_px) - 2, half_idx))
        side_a = pts_px[:half_idx]
        side_b = pts_px[half_idx:][::-1]
        def path_len(arr: np.ndarray) -> float:
            return float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).sum()) if len(arr) > 1 else 0.0
        # Pick one side of the line band, not the whole perimeter.  If one side is clearly
        # shorter/less jagged, prefer it; otherwise side_a is deterministic.
        la, lb = path_len(side_a), path_len(side_b)
        pts_px = side_a if la <= lb * 1.10 else side_b

    pts_mm = [frame.px_to_mm((float(x), float(y))) for x, y in pts_px]
    pts_mm = resample_polyline(pts_mm, coarse_spacing_mm, closed=True)
    return pts_mm


def make_initial_coarse_points(cfg: PipelineConfig, mask: np.ndarray, frame: ImageFrame) -> Tuple[List[Point], Dict[str, Any]]:
    seed_data: Dict[str, Any] = {}
    if cfg.seed_yaml:
        seed_points, seed_data = load_seed_yaml_points(cfg.seed_yaml)
        if len(seed_points) >= 4:
            mode = (seed_data.get("centerline", {}) or {}).get("mode", "")
            if mode == "spline" or len(seed_points) < 200:
                dense = catmull_rom_closed(seed_points, samples_per_segment=32)
            else:
                dense = seed_points
            coarse = resample_polyline(dense, cfg.coarse_spacing_mm, closed=True)
            return order_points_by_nearest_neighbor(coarse), seed_data

    coarse = auto_seed_from_contour(mask, frame, cfg.coarse_spacing_mm, cfg.reference)
    return order_points_by_nearest_neighbor(coarse), seed_data


# -----------------------------------------------------------------------------
# Recursive midpoint refinement
# -----------------------------------------------------------------------------


def red_candidate_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])


def nearest_mask_point_in_oriented_box(
    candidates_px: np.ndarray,
    a_px: Point,
    b_px: Point,
    search_width_factor: float,
) -> Optional[Point]:
    """Find target-mask pixel closest to the midpoint inside a local oriented box.

    A and B define the local vector.  The box is centered at midpoint, length is |AB|,
    and width is search_width_factor * |AB|.  This is a practical interpretation of
    'using the vector as the rectangle diagonal / local search region'.
    """
    if len(candidates_px) == 0:
        return None
    a = np.asarray(a_px, dtype=np.float64)
    b = np.asarray(b_px, dtype=np.float64)
    v = b - a
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return (float(a[0]), float(a[1]))
    mid = 0.5 * (a + b)
    u = v / length
    n = np.array([-u[1], u[0]])

    # Coarse axis-aligned prefilter for speed.
    radius = 0.75 * length * max(1.0, search_width_factor)
    lo = mid - radius
    hi = mid + radius
    m = (
        (candidates_px[:, 0] >= lo[0])
        & (candidates_px[:, 0] <= hi[0])
        & (candidates_px[:, 1] >= lo[1])
        & (candidates_px[:, 1] <= hi[1])
    )
    cand = candidates_px[m]
    if len(cand) == 0:
        return None

    d = cand - mid
    along = np.abs(d @ u)
    across = np.abs(d @ n)
    half_len = 0.55 * length
    half_width = 0.5 * search_width_factor * length
    inside = (along <= half_len) & (across <= half_width)
    cand = cand[inside]
    if len(cand) == 0:
        return None

    # The selected insertion point is the visible line pixel closest to the Euclidean half-distance point.
    dist2 = np.sum((cand - mid) ** 2, axis=1)
    p = cand[int(np.argmin(dist2))]
    return (float(p[0]), float(p[1]))


def recursive_refine_segment(
    a_mm: Point,
    b_mm: Point,
    frame: ImageFrame,
    candidates_px: np.ndarray,
    target_spacing_mm: float,
    search_width_factor: float,
    max_depth: int = 20,
) -> List[Point]:
    """Return [a, inserted..., b]."""
    ax, ay = a_mm
    bx, by = b_mm
    seg_len = math.hypot(bx - ax, by - ay)
    if seg_len <= target_spacing_mm or max_depth <= 0:
        return [a_mm, b_mm]

    a_px = frame.mm_to_px(a_mm)
    b_px = frame.mm_to_px(b_mm)
    p_px = nearest_mask_point_in_oriented_box(candidates_px, a_px, b_px, search_width_factor)
    if p_px is None:
        # Fallback to geometric midpoint if the mask is interrupted.
        mid_mm = ((ax + bx) * 0.5, (ay + by) * 0.5)
    else:
        mid_mm = frame.px_to_mm(p_px)

    left = recursive_refine_segment(
        a_mm, mid_mm, frame, candidates_px, target_spacing_mm, search_width_factor, max_depth - 1
    )
    right = recursive_refine_segment(
        mid_mm, b_mm, frame, candidates_px, target_spacing_mm, search_width_factor, max_depth - 1
    )
    return left[:-1] + right


def recursive_refine_loop(
    coarse_mm: Sequence[Point],
    frame: ImageFrame,
    mask: np.ndarray,
    target_spacing_mm: float,
    search_width_factor: float,
) -> List[Point]:
    candidates = red_candidate_points(mask)
    pts = list(coarse_mm)
    if len(pts) < 2:
        return pts
    out: List[Point] = []
    n = len(pts)
    for i in range(n):
        seg_pts = recursive_refine_segment(
            pts[i], pts[(i + 1) % n], frame, candidates, target_spacing_mm, search_width_factor
        )
        out.extend(seg_pts[:-1])
    return out


def snap_points_to_mask(
    points_mm: Sequence[Point], frame: ImageFrame, mask: np.ndarray, radius_mm: float
) -> List[Point]:
    """Optionally snap final points to the nearest target mask pixel within radius."""
    candidates = red_candidate_points(mask)
    if len(candidates) == 0 or radius_mm <= 0:
        return list(points_mm)
    radius_px = radius_mm / max(frame.mean_mm_per_px, 1e-9)
    out: List[Point] = []
    for p_mm in points_mm:
        p_px = np.asarray(frame.mm_to_px(p_mm), dtype=np.float64)
        lo = p_px - radius_px
        hi = p_px + radius_px
        m = (
            (candidates[:, 0] >= lo[0])
            & (candidates[:, 0] <= hi[0])
            & (candidates[:, 1] >= lo[1])
            & (candidates[:, 1] <= hi[1])
        )
        cand = candidates[m]
        if len(cand) == 0:
            out.append(p_mm)
            continue
        d2 = np.sum((cand - p_px) ** 2, axis=1)
        nearest = cand[int(np.argmin(d2))]
        if float(np.min(d2)) <= radius_px * radius_px:
            out.append(frame.px_to_mm((float(nearest[0]), float(nearest[1]))))
        else:
            out.append(p_mm)
    return out


# -----------------------------------------------------------------------------
# YAML / overlay output
# -----------------------------------------------------------------------------


def _round_point(p: Point, precision: int) -> Dict[str, float]:
    return {"x": round(float(p[0]), precision), "y": round(float(p[1]), precision)}


def build_output_yaml(
    cfg: PipelineConfig,
    frame: ImageFrame,
    points_mm: Sequence[Point],
    seed_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed_data = seed_data or {}
    elements = seed_data.get("elements", {}) if cfg.keep_elements_from_seed else {}
    lanes = seed_data.get("lanes", {"count": 2})
    dims_seed = seed_data.get("dimensions", {}) if isinstance(seed_data, dict) else {}

    lens = polyline_lengths(points_mm, closed=True)
    total_len = float(lens.sum()) if len(lens) else 0.0
    stats = {
        "point_count": len(points_mm),
        "estimated_length_mm": round(total_len, 2),
        "mean_spacing_mm": round(float(np.mean(lens)), 3) if len(lens) else 0.0,
        "max_spacing_mm": round(float(np.max(lens)), 3) if len(lens) else 0.0,
    }

    return {
        "meta": {
            "name": seed_data.get("meta", {}).get("name", "SKKU_AD_Track"),
            "generated_by": "track_yaml_pipeline.py",
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "source_image": os.path.basename(cfg.image_path),
            "method": "coarse_seed_recursive_midpoint_refinement",
            "note": "Image-derived coordinates. Use CAD/DXF for final production-level geometry.",
        },
        "dimensions": {
            "field_mm": [float(cfg.field_width_mm), float(cfg.field_height_mm)],
            "road_width_mm": float(dims_seed.get("road_width_mm", cfg.road_width_mm)),
            "lane_mark_mm": float(dims_seed.get("lane_mark_mm", 50)),
            "start_line_mm": float(dims_seed.get("start_line_mm", 100)),
            "crosswalk_mm": dims_seed.get("crosswalk_mm", [1000, 100]),
            "parking_mm": dims_seed.get("parking_mm", [950, 1500]),
        },
        "calibration": {
            "image_frame_px": [round(frame.x0, 2), round(frame.y0, 2), round(frame.x1, 2), round(frame.y1, 2)],
            "mm_per_px": [round(frame.sx, 6), round(frame.sy, 6)],
        },
        "lanes": lanes,
        "centerline": {
            "mode": "polyline",
            "coordinate_system": "origin=field_top_left, x=right, y=down, unit=mm",
            "reference": cfg.reference,
            "coarse_spacing_mm": float(cfg.coarse_spacing_mm),
            "target_spacing_mm": float(cfg.target_spacing_mm),
            **stats,
            "points": [_round_point(p, cfg.output_precision) for p in points_mm],
        },
        "elements": elements,
    }


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=120)


def draw_overlay(
    bgr: np.ndarray,
    frame: ImageFrame,
    mask: np.ndarray,
    coarse_mm: Sequence[Point],
    final_mm: Sequence[Point],
    out_path: str,
) -> None:
    overlay = bgr.copy()
    # mask overlay in red-ish
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay = np.where((mask > 0)[:, :, None], (0.65 * overlay + 0.35 * red).astype(np.uint8), overlay)

    def to_int(p: Point) -> Tuple[int, int]:
        x, y = frame.mm_to_px(p)
        return int(round(x)), int(round(y))

    # calibration frame
    cv2.rectangle(
        overlay,
        (int(round(frame.x0)), int(round(frame.y0))),
        (int(round(frame.x1)), int(round(frame.y1))),
        (80, 80, 80),
        2,
    )

    # final dense polyline in blue
    if len(final_mm) >= 2:
        pts = np.array([to_int(p) for p in final_mm], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # coarse seed points/lines in green
    if len(coarse_mm) >= 2:
        pts = np.array([to_int(p) for p in coarse_mm], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 180, 0), thickness=1, lineType=cv2.LINE_AA)
        for p in coarse_mm:
            cv2.circle(overlay, to_int(p), 4, (0, 200, 0), -1, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, overlay)


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    bgr = cv2.imread(cfg.image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(cfg.image_path)

    mask = segment_target_line(bgr, cfg)
    frame = auto_detect_frame(bgr, cfg, mask)
    coarse_mm, seed_data = make_initial_coarse_points(cfg, mask, frame)
    if len(coarse_mm) < 4:
        raise RuntimeError("Could not create enough seed points. Provide --seed-yaml with coarse control_points.")

    if cfg.skip_refinement:
        # 1. Interpolate seed points to be moderately dense
        refined = resample_polyline(coarse_mm, 50.0, closed=True)
        # 2. Pull them precisely onto the red mask (Snap)
        refined = snap_points_to_mask(refined, frame, mask, cfg.snap_radius_mm)
    else:
        refined = recursive_refine_loop(
            coarse_mm,
            frame,
            mask,
            target_spacing_mm=cfg.target_spacing_mm,
            search_width_factor=cfg.search_width_factor,
        )
        # Standardize exact spacing after recursive insertion.
        refined = resample_polyline(refined, cfg.target_spacing_mm, closed=True)
        refined = snap_points_to_mask(refined, frame, mask, cfg.snap_radius_mm)

    if cfg.smoothing_passes > 0:
        refined = chaikin_smooth(refined, passes=cfg.smoothing_passes, closed=True)
        refined = resample_polyline(refined, cfg.target_spacing_mm, closed=True)

    data = build_output_yaml(cfg, frame, refined, seed_data)
    write_yaml(cfg.output_yaml, data)
    draw_overlay(bgr, frame, mask, coarse_mm, refined, cfg.output_overlay)
    return data


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_frame_rect(s: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not s:
        return None
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("--frame-rect-px must be 'x0,y0,x1,y1'")
    return tuple(vals)  # type: ignore[return-value]


def parse_args() -> PipelineConfig:
    ap = argparse.ArgumentParser(description="Generate dense track.yaml from a CAD/JPG/PNG track image.")
    ap.add_argument("--image", required=True, help="Input JPG/PNG image path")
    ap.add_argument("--out-yaml", required=True, help="Output track.yaml path")
    ap.add_argument("--out-overlay", required=True, help="Output overlay preview PNG path")
    ap.add_argument("--seed-yaml", default=None, help="Optional YAML with centerline.control_points/points")
    ap.add_argument("--field-mm", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), default=[12000.0, 16000.0])
    ap.add_argument("--road-width-mm", type=float, default=850.0)
    ap.add_argument("--target-spacing-mm", type=float, default=10.0)
    ap.add_argument("--coarse-spacing-mm", type=float, default=1000.0)
    ap.add_argument("--reference", choices=["inner", "outer", "largest", "smallest"], default="inner")
    ap.add_argument("--frame-rect-px", type=parse_frame_rect, default=None, help="Manual calibration crop: x0,y0,x1,y1")
    ap.add_argument("--hsv-sat-min", type=int, default=6)
    ap.add_argument("--hsv-val-min", type=int, default=80)
    ap.add_argument("--red-delta", type=int, default=4)
    ap.add_argument("--mask-dilate-px", type=int, default=3)
    ap.add_argument("--search-width-factor", type=float, default=1.0)
    ap.add_argument("--snap-radius-mm", type=float, default=80.0)
    ap.add_argument("--smoothing-passes", type=int, default=1)
    ap.add_argument("--precision", type=int, default=1)
    ap.add_argument("--skip-refinement", action="store_true", help="Skip recursive search and just snap to mask")
    ns = ap.parse_args()
    return PipelineConfig(
        image_path=ns.image,
        output_yaml=ns.out_yaml,
        output_overlay=ns.out_overlay,
        seed_yaml=ns.seed_yaml,
        field_width_mm=ns.field_mm[0],
        field_height_mm=ns.field_mm[1],
        road_width_mm=ns.road_width_mm,
        target_spacing_mm=ns.target_spacing_mm,
        coarse_spacing_mm=ns.coarse_spacing_mm,
        reference=ns.reference,
        frame_rect_px=ns.frame_rect_px,
        hsv_sat_min=ns.hsv_sat_min,
        hsv_val_min=ns.hsv_val_min,
        red_delta=ns.red_delta,
        mask_dilate_px=ns.mask_dilate_px,
        search_width_factor=ns.search_width_factor,
        snap_radius_mm=ns.snap_radius_mm,
        smoothing_passes=ns.smoothing_passes,
        output_precision=ns.precision,
        skip_refinement=ns.skip_refinement,
    )


def main() -> None:
    cfg = parse_args()
    data = run_pipeline(cfg)
    cl = data["centerline"]
    print(f"Wrote: {cfg.output_yaml}")
    print(f"Overlay: {cfg.output_overlay}")
    print(
        f"points={cl['point_count']}, length={cl['estimated_length_mm']}mm, "
        f"mean_spacing={cl['mean_spacing_mm']}mm, max_spacing={cl['max_spacing_mm']}mm"
    )


if __name__ == "__main__":
    main()
