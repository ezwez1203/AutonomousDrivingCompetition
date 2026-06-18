"""Camera-based lane/track alignment for Phase 3."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import LaneObservation, ParkingLineObservation


@dataclass(slots=True)
class LaneDetectorConfig:
    roi_y_start: float = 0.45
    min_pixels: int = 80
    min_confidence: float = 0.05
    white_threshold: int = 185
    saturation_max: int = 70
    yellow_red_min: int = 120
    yellow_green_min: int = 110
    yellow_blue_max: int = 120


class LaneDetector:
    """Estimate lateral and heading error from visible lane/track markings.

    The detector intentionally uses only NumPy so it works in the CARLA conda
    environment even before OpenCV is installed. It looks for bright white and
    yellow-ish markings in the lower camera ROI and fits a center line through
    row-wise mask centroids.
    """

    def __init__(self, cfg: LaneDetectorConfig | None = None):
        self.cfg = cfg or LaneDetectorConfig()

    def detect(self, camera_bgra: np.ndarray | None) -> LaneObservation:
        if camera_bgra is None:
            return LaneObservation(False, message="camera missing")
        if camera_bgra.ndim != 3 or camera_bgra.shape[2] < 3:
            return LaneObservation(False, message=f"bad camera shape {camera_bgra.shape}")

        height, width = camera_bgra.shape[:2]
        y0 = int(max(0, min(height - 1, round(height * self.cfg.roi_y_start))))
        bgr = camera_bgra[y0:, :, :3].astype(np.int16, copy=False)
        blue = bgr[:, :, 0]
        green = bgr[:, :, 1]
        red = bgr[:, :, 2]

        maxc = np.maximum.reduce([red, green, blue])
        minc = np.minimum.reduce([red, green, blue])
        saturation = maxc - minc
        white = (maxc >= self.cfg.white_threshold) & (saturation <= self.cfg.saturation_max)
        yellow = (
            (red >= self.cfg.yellow_red_min)
            & (green >= self.cfg.yellow_green_min)
            & (blue <= self.cfg.yellow_blue_max)
            & ((red - blue) >= 35)
        )
        mask = white | yellow
        mask_pixels = int(mask.sum())
        image_center = width * 0.5
        if mask_pixels < self.cfg.min_pixels:
            return LaneObservation(
                False,
                image_center_px=image_center,
                mask_pixels=mask_pixels,
                message="not enough lane pixels",
            )

        rows, cols = np.nonzero(mask)
        band_count = 12
        band_edges = np.linspace(0, mask.shape[0], band_count + 1, dtype=int)
        centers: list[tuple[float, float, int]] = []
        for lo, hi in zip(band_edges[:-1], band_edges[1:]):
            band_cols = cols[(rows >= lo) & (rows < hi)]
            if len(band_cols) < max(5, self.cfg.min_pixels // band_count):
                continue
            y_center = y0 + (lo + hi - 1) * 0.5
            centers.append((float(y_center), float(np.mean(band_cols)), int(len(band_cols))))

        confidence = min(1.0, mask_pixels / max(1.0, width * (height - y0) * 0.08))
        if len(centers) < 2:
            lane_center = float(np.mean(cols))
            error_px = lane_center - image_center
            detected = confidence >= self.cfg.min_confidence
            return LaneObservation(
                detected,
                center_error_px=float(error_px),
                center_error_norm=float(error_px / image_center),
                heading_error_rad=0.0,
                lane_center_px=lane_center,
                image_center_px=image_center,
                confidence=float(confidence),
                mask_pixels=mask_pixels,
                message="" if detected else "low confidence",
            )

        ys = np.asarray([row for row, _center, _count in centers], dtype=np.float64)
        xs = np.asarray([center for _row, center, _count in centers], dtype=np.float64)
        weights = np.asarray([count for _row, _center, count in centers], dtype=np.float64)
        fit = np.polyfit(ys, xs, deg=1, w=np.sqrt(weights))
        slope_dx_per_y = float(fit[0])
        lane_center = float(fit[0] * (height - 1) + fit[1])
        error_px = lane_center - image_center
        heading_error = math.atan(slope_dx_per_y)
        detected = confidence >= self.cfg.min_confidence

        return LaneObservation(
            detected,
            center_error_px=float(error_px),
            center_error_norm=float(error_px / image_center),
            heading_error_rad=float(heading_error),
            lane_center_px=lane_center,
            image_center_px=image_center,
            confidence=float(confidence),
            mask_pixels=mask_pixels,
            message="" if detected else "low confidence",
        )


@dataclass(slots=True)
class ParkingLineDetectorConfig:
    roi_y_start: float = 0.58
    roi_y_end: float = 0.95
    min_pixels: int = 60
    min_confidence: float = 0.03
    white_threshold: int = 185
    saturation_max: int = 75


class ParkingLineDetector:
    """Estimate parking slot center error from bright markings in the lower ROI."""

    def __init__(self, cfg: ParkingLineDetectorConfig | None = None):
        self.cfg = cfg or ParkingLineDetectorConfig()

    def detect(self, camera_bgra: np.ndarray | None) -> ParkingLineObservation:
        if camera_bgra is None:
            return ParkingLineObservation(False, message="camera missing")
        if camera_bgra.ndim != 3 or camera_bgra.shape[2] < 3:
            return ParkingLineObservation(False, message=f"bad camera shape {camera_bgra.shape}")

        height, width = camera_bgra.shape[:2]
        y0 = int(max(0, min(height - 1, round(height * self.cfg.roi_y_start))))
        y1 = int(max(y0 + 1, min(height, round(height * self.cfg.roi_y_end))))
        bgr = camera_bgra[y0:y1, :, :3].astype(np.int16, copy=False)
        blue = bgr[:, :, 0]
        green = bgr[:, :, 1]
        red = bgr[:, :, 2]
        maxc = np.maximum.reduce([red, green, blue])
        minc = np.minimum.reduce([red, green, blue])
        mask = (maxc >= self.cfg.white_threshold) & ((maxc - minc) <= self.cfg.saturation_max)
        rows, cols = np.nonzero(mask)
        mask_pixels = int(mask.sum())
        if mask_pixels < self.cfg.min_pixels:
            return ParkingLineObservation(False, mask_pixels=mask_pixels, message="not enough parking pixels")

        image_center = width * 0.5
        center_px = float(np.mean(cols))
        error_px = center_px - image_center
        confidence = min(1.0, mask_pixels / max(1.0, width * (y1 - y0) * 0.06))
        detected = confidence >= self.cfg.min_confidence
        return ParkingLineObservation(
            detected,
            center_error_px=float(error_px),
            center_error_norm=float(error_px / image_center),
            confidence=float(confidence),
            mask_pixels=mask_pixels,
            message="" if detected else "low confidence",
        )
