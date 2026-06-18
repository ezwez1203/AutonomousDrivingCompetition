# Track YAML Pipeline

This pipeline extracts the red track boundary from a CAD-exported JPG/PNG image and generates a `track.yaml` candidate. It is deterministic image geometry extraction, not really a trained ML model. Still, it is useful when a clean DXF/CAD source is not available yet.

## Flow

1. Segment the red/pink CAD boundary line with HSV/RGB thresholds.
2. Build a coarse 1000 mm path.
3. If `--seed-yaml` is provided, use `centerline.control_points` or `centerline.points` as a starting shape.
4. If no seed is provided, choose the inner or outer red contour automatically.
5. Refine midpoints recursively until every segment is shorter than `--target-spacing-mm`.
6. Save the final polyline under `centerline.points` and write an overlay PNG for review.

## Basic Run

```bash
python -m carla_autodrive.track_yaml_pipeline.track_yaml_pipeline \
  --image circuit_blueprint.png \
  --output carla_autodrive/track_yaml_pipeline/track_candidate.yaml \
  --overlay carla_autodrive/track_yaml_pipeline/track_candidate_overlay.png
```

## Run With An Existing YAML Seed

```bash
python -m carla_autodrive.track_yaml_pipeline.track_yaml_pipeline \
  --image circuit_blueprint.png \
  --seed-yaml carla_autodrive/config/track.yaml \
  --output carla_autodrive/track_yaml_pipeline/track_candidate.yaml \
  --overlay carla_autodrive/track_yaml_pipeline/track_candidate_overlay.png
```

## Main Tuning Parameters

- `--field-mm WIDTH HEIGHT`: real field size used to convert image coordinates to millimeters.
- `--frame-rect-px x0,y0,x1,y1`: manual calibration frame override.
- `--reference inner|outer`: which red boundary to treat as the route reference.
- `--hsv-sat-min`, `--red-delta`: red-line segmentation thresholds.
- `--mask-dilate-px`: dilation size for reconnecting broken CAD strokes.
- `--search-width-factor`: local search width during midpoint refinement.
- `--snap-radius-mm`: maximum radius for snapping final points back to the red mask.
- `--smoothing-passes`: light smoothing passes to reduce CAD export noise.

## Limits

If the original DXF/CAD becomes available, prefer parsing that for the final OpenDRIVE build. The image route is helpful, but it is still a guess from raster pixels.
