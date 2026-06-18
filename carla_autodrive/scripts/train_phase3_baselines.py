#!/usr/bin/env python
"""Train baseline models for Phase 3 synthetic perception labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from carla_autodrive.learning import TrainConfig, train_phase3_baselines
from carla_autodrive.utils import get_logger

log = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train lane/traffic/obstacle baseline models from Phase 3 synthetic dataset"
    )
    parser.add_argument("--dataset-dir", required=True, help="Directory containing labels.jsonl and snapshots/")
    parser.add_argument("--out-dir", default="carla_autodrive/reports/phase3_baselines")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated tasks: lane,traffic,obstacle,all",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = tuple(item.strip().lower() for item in args.tasks.split(",") if item.strip())
    valid = {"all", "lane", "traffic", "obstacle"}
    unknown = sorted(set(tasks) - valid)
    if unknown:
        log.error("unknown task(s): %s", unknown)
        return 2

    cfg = TrainConfig(
        dataset_dir=Path(args.dataset_dir),
        out_dir=Path(args.out_dir),
        tasks=tasks or ("all",),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_width=args.image_width,
        image_height=args.image_height,
        max_samples=args.max_samples,
        device=args.device,
        num_workers=args.num_workers,
    )
    try:
        results = train_phase3_baselines(cfg)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("training failed: %s", exc)
        return 1

    log.info("Phase 3 baseline training report:")
    log.info("%s", json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
