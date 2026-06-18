#!/usr/bin/env python
"""Train a weakly-supervised pixel model for blueprint track extraction.

There is only one blueprint image available, so this trains from pseudo-labels:
the largest non-white connected component is treated as positive track/road
evidence, and far-white background pixels are treated as negative. The trained
model outputs a probability mask that can be used by ``fit_track_from_blueprint``
instead of a hand-written color threshold.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from carla_autodrive.scripts.measure_track_accuracy import filter_component, mask_pixels

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMAGE = ROOT / "circuit_blueprint.png"
DEFAULT_OUT = ROOT / "carla_autodrive" / "reports" / "ml_track_model"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="train the weakly supervised blueprint track extraction model")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", choices=("histgb", "forest"), default="histgb")
    ap.add_argument("--positive-mask", choices=("nonwhite", "green", "red", "dark"), default="nonwhite")
    ap.add_argument("--positive-component", choices=("largest", "all"), default="largest")
    ap.add_argument("--negative-margin-px", type=float, default=18.0,
                    help="use only background farther than this distance from positives as negative pseudo-labels")
    ap.add_argument("--samples-per-class", type=int, default=50000)
    ap.add_argument("--predict-chunk-rows", type=int, default=256)
    ap.add_argument("--prob-threshold", type=float, default=0.45)
    ap.add_argument("--random-seed", type=int, default=17)
    return ap.parse_args()


def rgb_to_hsv_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=2)
    mn = np.min(arr, axis=2)
    diff = mx - mn
    hue = np.zeros_like(mx)
    mask = diff > 1e-6
    idx = (mx == r) & mask
    hue[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = (mx == g) & mask
    hue[idx] = (b[idx] - r[idx]) / diff[idx] + 2
    idx = (mx == b) & mask
    hue[idx] = (r[idx] - g[idx]) / diff[idx] + 4
    hue /= 6.0
    sat = np.where(mx > 1e-6, diff / mx, 0.0)
    return hue, sat, mx


def build_feature_planes(rgb: np.ndarray) -> list[np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    hue, sat, val = rgb_to_hsv_features(rgb)
    h, w = gray.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    xx /= max(w - 1, 1)
    yy /= max(h - 1, 1)
    gray_blur = ndimage.gaussian_filter(gray, sigma=2.0)
    sat_blur = ndimage.gaussian_filter(sat, sigma=2.0)
    grad = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
    return [r, g, b, gray, hue, sat, val, gray_blur, sat_blur, grad, xx, yy]


def features_at(planes: list[np.ndarray], ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.column_stack([plane[ys, xs] for plane in planes]).astype(np.float32)


def features_for_rows(planes: list[np.ndarray], y0: int, y1: int) -> np.ndarray:
    return np.column_stack([plane[y0:y1].reshape(-1) for plane in planes]).astype(np.float32)


def pseudo_labels(rgb: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    positive_raw = mask_pixels(rgb, args.positive_mask)
    positive = filter_component(positive_raw, args.positive_component)
    nonwhite = mask_pixels(rgb, "nonwhite")
    dist_from_positive = ndimage.distance_transform_edt(~positive)
    negative = (~nonwhite) & (dist_from_positive >= args.negative_margin_px)
    return positive, negative


def sample_training_points(positive: np.ndarray, negative: np.ndarray, args: argparse.Namespace):
    rng = np.random.default_rng(args.random_seed)
    pos_y, pos_x = np.where(positive)
    neg_y, neg_x = np.where(negative)
    if len(pos_x) == 0 or len(neg_x) == 0:
        raise ValueError("pseudo-label positive/negative samples are empty.")
    n_pos = min(args.samples_per_class, len(pos_x))
    n_neg = min(args.samples_per_class, len(neg_x))
    pos_idx = rng.choice(len(pos_x), size=n_pos, replace=False)
    neg_idx = rng.choice(len(neg_x), size=n_neg, replace=False)
    ys = np.concatenate([pos_y[pos_idx], neg_y[neg_idx]])
    xs = np.concatenate([pos_x[pos_idx], neg_x[neg_idx]])
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)])
    return ys, xs, y


def make_model(kind: str, seed: int):
    if kind == "forest":
        return RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=4,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed,
        )
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=seed,
    )


def predict_probability_image(model, planes: list[np.ndarray], chunk_rows: int) -> np.ndarray:
    h, w = planes[0].shape
    out = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        x = features_for_rows(planes, y0, y1)
        out[y0:y1] = model.predict_proba(x)[:, 1].reshape(y1 - y0, w)
    return out


def save_overlay(image: Image.Image, positive: np.ndarray, probability: np.ndarray,
                 threshold: float, out_path: Path) -> None:
    overlay = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    arr = np.asarray(layer).copy()
    pred = probability >= threshold
    arr[positive] = (0, 180, 0, 95)
    arr[pred] = (255, 0, 0, 95)
    overlay = Image.alpha_composite(overlay, Image.fromarray(arr))
    draw = ImageDraw.Draw(overlay)
    draw.text((24, 24), "green=pseudo-label, red=model prediction", fill=(0, 0, 0, 255))
    overlay.convert("RGB").save(out_path)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image)
    planes = build_feature_planes(rgb)
    positive, negative = pseudo_labels(rgb, args)
    ys, xs, y = sample_training_points(positive, negative, args)
    x = features_at(planes, ys, xs)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, random_state=args.random_seed, stratify=y
    )
    model = make_model(args.model, args.random_seed)
    model.fit(x_train, y_train)
    val_prob = model.predict_proba(x_val)[:, 1]
    val_pred = val_prob >= args.prob_threshold

    probability = predict_probability_image(model, planes, args.predict_chunk_rows)
    prob_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    prob_path = args.out_dir / "track_probability.png"
    mask_path = args.out_dir / "track_prediction_mask.png"
    overlay_path = args.out_dir / "track_model_overlay.png"
    model_path = args.out_dir / "track_pixel_model.pkl"
    report_path = args.out_dir / "track_model_report.json"

    Image.fromarray(prob_u8).save(prob_path)
    Image.fromarray((probability >= args.prob_threshold).astype(np.uint8) * 255).save(mask_path)
    save_overlay(image, positive, probability, args.prob_threshold, overlay_path)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    report = {
        "image": str(args.image),
        "model": args.model,
        "positive_mask": args.positive_mask,
        "positive_component": args.positive_component,
        "positive_pixels": int(positive.sum()),
        "negative_pixels": int(negative.sum()),
        "samples_per_class": args.samples_per_class,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "validation_auc": float(roc_auc_score(y_val, val_prob)),
        "validation_report": classification_report(y_val, val_pred, output_dict=True),
        "prob_threshold": args.prob_threshold,
        "prediction_positive_pixels": int((probability >= args.prob_threshold).sum()),
        "outputs": {
            "model": str(model_path),
            "probability": str(prob_path),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
        },
        "notes": [
            "This is weak supervision from pseudo-labels, not ground-truth CAD labels.",
            "Use track_probability.png with fit_track_from_blueprint --probability-mask.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"validation_auc={report['validation_auc']:.4f}")
    print(f"positive_pixels={report['positive_pixels']} negative_pixels={report['negative_pixels']}")
    print(f"prediction_positive_pixels={report['prediction_positive_pixels']}")
    print(f"wrote {model_path}")
    print(f"wrote {prob_path}")
    print(f"wrote {mask_path}")
    print(f"wrote {overlay_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
