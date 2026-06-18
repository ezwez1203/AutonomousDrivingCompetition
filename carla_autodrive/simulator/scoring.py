"""Phase 6 scoring and run report utilities."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TickRecord:
    tick: int
    sim_time_s: float
    state: str
    decision_reason: str
    control_reason: str
    speed_mps: float
    desired_speed_mps: float
    target_speed_mps: float
    throttle: float
    brake: float
    steer: float
    reverse: bool
    cross_track_error_m: float
    heading_error_rad: float
    collision_count: int = 0
    collision_impulse: float = 0.0
    lane_intrusion: bool = False
    lane_departure: bool = False
    stop_required: bool = False
    stop_violation: bool = False
    parking_hold: bool = False
    route_index: int | None = None
    route_s_m: float | None = None


@dataclass(slots=True)
class RunSummary:
    mission: str
    completed: bool
    finish_reason: str
    ticks: int
    sim_time_s: float
    distance_m: float
    avg_speed_mps: float
    max_speed_mps: float
    mean_abs_cte_m: float
    max_abs_cte_m: float
    max_abs_heading_rad: float
    states: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    events: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    score: float = 0.0


@dataclass(slots=True)
class ScoringThresholds:
    cte_warning_m: float = 0.75
    lane_intrusion_cte_m: float = 0.45
    lane_departure_cte_m: float = 0.85
    stop_violation_speed_mps: float = 0.35


class CompetitionScorer:
    """Collect run telemetry and compute lightweight competition-oriented metrics."""

    def __init__(
        self,
        mission: str,
        *,
        time_limit_s: float | None = None,
        cte_warning_m: float = 0.75,
        lane_intrusion_cte_m: float = 0.45,
        lane_departure_cte_m: float = 0.85,
        stop_violation_speed_mps: float = 0.35,
    ):
        self.mission = mission
        self.time_limit_s = time_limit_s
        self.thresholds = ScoringThresholds(
            cte_warning_m=float(cte_warning_m),
            lane_intrusion_cte_m=float(lane_intrusion_cte_m),
            lane_departure_cte_m=float(lane_departure_cte_m),
            stop_violation_speed_mps=float(stop_violation_speed_mps),
        )
        self.records: list[TickRecord] = []
        self.states: dict[str, int] = {}
        self.reasons: dict[str, int] = {}
        self.distance_m = 0.0
        self.finish_reason = "not_finished"
        self.completed = False
        self.finish_time_s: float | None = None

    def add_tick(self, record: TickRecord) -> None:
        self.records.append(record)
        self.states[record.state] = self.states.get(record.state, 0) + 1
        reason = f"{record.decision_reason}|{record.control_reason}"
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def set_distance(self, distance_m: float) -> None:
        self.distance_m = float(distance_m)

    def finish(self, *, completed: bool, reason: str, sim_time_s: float | None = None) -> None:
        self.completed = bool(completed)
        self.finish_reason = reason
        if sim_time_s is not None:
            self.finish_time_s = float(sim_time_s)

    def summary(self) -> RunSummary:
        speeds = [record.speed_mps for record in self.records]
        abs_ctes = [abs(record.cross_track_error_m) for record in self.records]
        abs_headings = [abs(record.heading_error_rad) for record in self.records]
        sim_time_s = self.finish_time_s if self.finish_time_s is not None else (self.records[-1].sim_time_s if self.records else 0.0)
        events = self._events()
        penalties = self._penalties(sim_time_s, abs_ctes, events)
        return RunSummary(
            mission=self.mission,
            completed=self.completed,
            finish_reason=self.finish_reason,
            ticks=len(self.records),
            sim_time_s=sim_time_s,
            distance_m=self.distance_m,
            avg_speed_mps=sum(speeds) / len(speeds) if speeds else 0.0,
            max_speed_mps=max(speeds) if speeds else 0.0,
            mean_abs_cte_m=sum(abs_ctes) / len(abs_ctes) if abs_ctes else 0.0,
            max_abs_cte_m=max(abs_ctes) if abs_ctes else 0.0,
            max_abs_heading_rad=max(abs_headings) if abs_headings else 0.0,
            states=dict(self.states),
            reasons=dict(self.reasons),
            events=events,
            penalties=penalties,
            score=sum(penalties.values()),
        )

    def write_json(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "summary": asdict(self.summary()),
            "ticks": [asdict(record) for record in self.records],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(record) for record in self.records]
        with out.open("w", newline="", encoding="utf-8") as fh:
            if not rows:
                fh.write("")
                return
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _events(self) -> dict[str, float]:
        if not self.records:
            return {}

        stop_speeds = [record.speed_mps for record in self.records if record.stop_required]
        return {
            "collision_count": float(max(record.collision_count for record in self.records)),
            "collision_max_impulse": float(max(record.collision_impulse for record in self.records)),
            "lane_intrusion_ticks": float(sum(1 for record in self.records if record.lane_intrusion)),
            "lane_departure_ticks": float(sum(1 for record in self.records if record.lane_departure)),
            "stop_required_ticks": float(sum(1 for record in self.records if record.stop_required)),
            "stop_violation_ticks": float(sum(1 for record in self.records if record.stop_violation)),
            "parking_hold_ticks": float(sum(1 for record in self.records if record.parking_hold)),
            "min_stop_speed_mps": min(stop_speeds) if stop_speeds else 0.0,
        }

    def _penalties(self, sim_time_s: float, abs_ctes: list[float], events: dict[str, float]) -> dict[str, float]:
        penalties: dict[str, float] = {}
        if self.time_limit_s is not None and sim_time_s > self.time_limit_s:
            penalties["time_limit_excess_s"] = sim_time_s - self.time_limit_s
        if abs_ctes:
            cte_excess = max(abs_ctes) - self.thresholds.cte_warning_m
            if cte_excess > 0.0:
                penalties["cte_warning_excess_m"] = cte_excess
        collision_count = events.get("collision_count", 0.0)
        if collision_count > 0.0:
            penalties["collision_events"] = collision_count * 100.0
        lane_departure_ticks = events.get("lane_departure_ticks", 0.0)
        if lane_departure_ticks > 0.0:
            penalties["lane_departure_ticks"] = lane_departure_ticks
        lane_intrusion_ticks = events.get("lane_intrusion_ticks", 0.0)
        if lane_intrusion_ticks > 0.0:
            penalties["lane_intrusion_ticks"] = lane_intrusion_ticks * 0.1
        stop_violation_ticks = events.get("stop_violation_ticks", 0.0)
        if stop_violation_ticks > 0.0:
            penalties["stop_violation_ticks"] = stop_violation_ticks * 0.2
        if not self.completed:
            penalties["incomplete"] = 1000.0
        return penalties
