#!/usr/bin/env python
"""Phase 6 parameter test runner for Phase 5 mission runs."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from carla_autodrive.state_machine import MissionMode
from carla_autodrive.utils import get_logger

log = get_logger()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: run Phase 5 missions over a small parameter grid")
    parser.add_argument("--mission", choices=[mode.value for mode in MissionMode], default=MissionMode.TIME_TRIAL.value)
    parser.add_argument("--out-dir", default="carla_autodrive/reports/phase6")
    parser.add_argument("--ticks", type=int, default=7000)
    parser.add_argument("--target-speeds", default="2.0")
    parser.add_argument("--curve-max-lat-accs", default="0.45")
    parser.add_argument("--curve-lookaheads", default="8.0")
    parser.add_argument("--steer-speed-gains", default="2.2")
    parser.add_argument("--route-lane", type=int, default=2)
    parser.add_argument("--obstacle2", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--obstacle3", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--parking-zone", type=int, default=2, choices=(1, 2))
    parser.add_argument("--green-after-sec", type=float, default=3.0)
    parser.add_argument("--spawn-preset-obstacles", action="store_true")
    parser.add_argument("--no-perception", action="store_true", default=True)
    parser.add_argument("--with-perception", dest="no_perception", action="store_false")
    parser.add_argument("--no-collision-sensor", action="store_true")
    parser.add_argument("--cte-warning", type=float, default=0.75)
    parser.add_argument("--lane-intrusion-cte", type=float, default=0.45)
    parser.add_argument("--lane-departure-cte", type=float, default=0.85)
    parser.set_defaults(lane_corridor_scoring=True)
    parser.add_argument("--lane-corridor-scoring", dest="lane_corridor_scoring", action="store_true",
                        help="Score lane events against the lane corridor. This is the default.")
    parser.add_argument("--no-lane-corridor-scoring", dest="lane_corridor_scoring", action="store_false",
                        help="Use legacy raw route-CTE thresholds for lane events.")
    parser.add_argument("--lane-boundary-margin", type=float, default=0.0)
    parser.add_argument("--rank-objective", choices=("score", "time"), default="score",
                        help="Rank by score first, or by fastest safe completion first.")
    parser.add_argument("--stop-violation-speed", type=float, default=0.35)
    parser.add_argument("--no-auto-load-track-map", action="store_true")
    parser.add_argument("--record-best-video", action="store_true",
                        help="Record front/rear videos during each run but retain only the current best run videos")
    parser.add_argument("--best-video-dir", default=None,
                        help="Directory for retained best-run videos. Defaults to <out-dir>/best_run_video")
    parser.add_argument("--record-video-every", type=int, default=1,
                        help="Record one video frame every N simulator ticks when --record-best-video is enabled")
    parser.add_argument("--record-video-fps", type=float, default=20.0,
                        help="FPS metadata for best-run video output")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    grid = list(itertools.product(
        parse_float_list(args.target_speeds),
        parse_float_list(args.curve_max_lat_accs),
        parse_float_list(args.curve_lookaheads),
        parse_float_list(args.steer_speed_gains),
    ))
    video_root = Path(args.best_video_dir) if args.best_video_dir else out_dir / "best_run_video"
    video_candidates_dir = out_dir / ".video_candidates"
    best_video_dir: Path | None = None
    best_video_row: dict[str, object] | None = None
    log.info("Phase 6 test runner: mission=%s runs=%d out=%s", args.mission, len(grid), out_dir)

    for idx, (target_speed, curve_acc, curve_lookahead, steer_gain) in enumerate(grid, start=1):
        stem = (
            f"run{idx:03d}_{args.mission}_v{target_speed:g}_"
            f"acc{curve_acc:g}_look{curve_lookahead:g}_steer{steer_gain:g}"
        ).replace(".", "p")
        report_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"{stem}.ticks.csv"
        candidate_video_dir = video_candidates_dir / stem
        cmd = [
            sys.executable,
            "-m",
            "carla_autodrive.scripts.phase5_mission_runner",
            "--mission", args.mission,
            "--ticks", str(args.ticks),
            "--target-speed", str(target_speed),
            "--curve-max-lat-acc", str(curve_acc),
            "--curve-lookahead", str(curve_lookahead),
            "--steer-speed-gain", str(steer_gain),
            "--route-lane", str(args.route_lane),
            "--cte-warning", str(args.cte_warning),
            "--lane-intrusion-cte", str(args.lane_intrusion_cte),
            "--lane-departure-cte", str(args.lane_departure_cte),
            "--lane-boundary-margin", str(args.lane_boundary_margin),
            "--stop-violation-speed", str(args.stop_violation_speed),
            "--report-path", str(report_path),
            "--csv-path", str(csv_path),
        ]
        if args.record_best_video:
            cmd += [
                "--monitor-cameras",
                "--record-monitor-video-dir", str(candidate_video_dir),
                "--record-monitor-video-every", str(args.record_video_every),
                "--record-monitor-video-fps", str(args.record_video_fps),
            ]
        if args.lane_corridor_scoring:
            cmd.append("--lane-corridor-scoring")
        else:
            cmd.append("--no-lane-corridor-scoring")
        if args.no_perception:
            cmd.append("--no-perception")
        if args.no_collision_sensor:
            cmd.append("--no-collision-sensor")
        if args.no_auto_load_track_map:
            cmd.append("--no-auto-load-track-map")
        if args.mission == MissionMode.OBSTACLE_SIGNAL.value:
            cmd += ["--obstacle2", str(args.obstacle2), "--obstacle3", str(args.obstacle3), "--green-after-sec", str(args.green_after_sec)]
            if args.spawn_preset_obstacles:
                cmd.append("--spawn-preset-obstacles")
        if args.mission == MissionMode.PARKING.value:
            cmd += ["--parking-zone", str(args.parking_zone), "--reverse-parking"]

        log.info("run %d/%d: %s", idx, len(grid), " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=Path.cwd(), check=False)
        if result.returncode != 0:
            if args.record_best_video:
                _delete_tree(candidate_video_dir)
            rows.append({
                "run": idx,
                "status": "failed_process",
                "returncode": result.returncode,
                "report": str(report_path),
                "target_speed_mps": target_speed,
                "curve_max_lat_acc": curve_acc,
                "curve_lookahead": curve_lookahead,
                "steer_speed_gain": steer_gain,
            })
            continue
        row = _summary_row(idx, report_path, target_speed, curve_acc, curve_lookahead, steer_gain)
        if args.record_best_video:
            if _is_better(row, best_video_row, objective=args.rank_objective):
                if best_video_dir is not None:
                    _delete_tree(best_video_dir)
                video_root.parent.mkdir(parents=True, exist_ok=True)
                _delete_tree(video_root)
                if candidate_video_dir.exists():
                    shutil.move(str(candidate_video_dir), str(video_root))
                    row["video_dir"] = str(video_root)
                best_video_dir = video_root
                if best_video_row is not None:
                    best_video_row.pop("video_dir", None)
                    best_video_row["video_dir_deleted"] = True
                best_video_row = row
                log.info("new best video retained: run=%d dir=%s", idx, video_root)
            else:
                _delete_tree(candidate_video_dir)
                row["video_dir_deleted"] = True
        rows.append(row)

    summary_path = out_dir / "summary.csv"
    if rows:
        _annotate_rank(rows, objective=args.rank_objective)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Phase 6 summary written: %s", summary_path)
        best_rows = [row for row in rows if row.get("rank") == 1]
        if best_rows:
            best_path = out_dir / "best_run.json"
            best_path.write_text(json.dumps(best_rows[0], ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Phase 6 best run written: %s", best_path)
            if args.record_best_video:
                log.info("Phase 6 retained best-run video dir: %s", best_rows[0].get("video_dir", video_root))
    elif args.dry_run:
        log.info("dry-run complete; no summary written")
    return 0



def _delete_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _is_better(row: dict[str, object], best_row: dict[str, object] | None, *, objective: str) -> bool:
    if row.get("status") != "ok":
        return False
    if best_row is None:
        return True
    return _rank_key(row, objective=objective) < _rank_key(best_row, objective=objective)


def _summary_row(idx: int, report_path: Path, target_speed: float, curve_acc: float, curve_lookahead: float, steer_gain: float) -> dict[str, object]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    penalties = summary.get("penalties", {})
    events = summary.get("events", {})
    return {
        "run": idx,
        "status": "ok",
        "report": str(report_path),
        "mission": summary.get("mission"),
        "completed": summary.get("completed"),
        "finish_reason": summary.get("finish_reason"),
        "sim_time_s": summary.get("sim_time_s"),
        "distance_m": summary.get("distance_m"),
        "avg_speed_mps": summary.get("avg_speed_mps"),
        "max_speed_mps": summary.get("max_speed_mps"),
        "mean_abs_cte_m": summary.get("mean_abs_cte_m"),
        "max_abs_cte_m": summary.get("max_abs_cte_m"),
        "score": summary.get("score"),
        "collision_count": events.get("collision_count", 0.0),
        "collision_max_impulse": events.get("collision_max_impulse", 0.0),
        "lane_intrusion_ticks": events.get("lane_intrusion_ticks", 0.0),
        "lane_departure_ticks": events.get("lane_departure_ticks", 0.0),
        "stop_violation_ticks": events.get("stop_violation_ticks", 0.0),
        "parking_hold_ticks": events.get("parking_hold_ticks", 0.0),
        "time_limit_excess_s": penalties.get("time_limit_excess_s", 0.0),
        "cte_warning_excess_m": penalties.get("cte_warning_excess_m", 0.0),
        "collision_penalty": penalties.get("collision_events", 0.0),
        "lane_departure_penalty": penalties.get("lane_departure_ticks", 0.0),
        "lane_intrusion_penalty": penalties.get("lane_intrusion_ticks", 0.0),
        "stop_violation_penalty": penalties.get("stop_violation_ticks", 0.0),
        "incomplete_penalty": penalties.get("incomplete", 0.0),
        "target_speed_mps": target_speed,
        "curve_max_lat_acc": curve_acc,
        "curve_lookahead": curve_lookahead,
        "steer_speed_gain": steer_gain,
    }


def _annotate_rank(rows: list[dict[str, object]], *, objective: str = "score") -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ranked = sorted(ok_rows, key=lambda row: _rank_key(row, objective=objective))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["rank_objective"] = objective
        row["rank_key"] = "|".join(str(item) for item in _rank_key(row, objective=objective))


def _rank_key(row: dict[str, object], *, objective: str = "score") -> tuple[object, ...]:
    completed = bool(row.get("completed"))
    collision_count = float(row.get("collision_count") or 0.0)
    lane_departure_ticks = float(row.get("lane_departure_ticks") or 0.0)
    stop_violation_ticks = float(row.get("stop_violation_ticks") or 0.0)
    score = float(row.get("score") or 0.0)
    sim_time_s = float(row.get("sim_time_s") or 1e9)
    mean_abs_cte = float(row.get("mean_abs_cte_m") or 1e9)
    if objective == "time":
        return (
            not completed,
            collision_count > 0.0,
            lane_departure_ticks > 0.0,
            stop_violation_ticks > 0.0,
            sim_time_s,
            score,
            mean_abs_cte,
        )
    return (
        not completed,
        score,
        sim_time_s,
        mean_abs_cte,
    )


if __name__ == "__main__":
    sys.exit(main())
