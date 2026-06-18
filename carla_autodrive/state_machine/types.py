"""Mission state and decision types for Phase 5."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MissionMode(str, Enum):
    TIME_TRIAL = "time_trial"
    OBSTACLE_SIGNAL = "obstacle_signal"
    PARKING = "parking"


class MissionState(str, Enum):
    IDLE = "IDLE"
    TIME_TRIAL = "TIME_TRIAL"
    LANE_FOLLOW = "LANE_FOLLOW"
    OBSTACLE_AVOID = "OBSTACLE_AVOID"
    TRAFFIC_STOP = "TRAFFIC_STOP"
    PARKING = "PARKING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


@dataclass(slots=True)
class MissionDecision:
    state: MissionState
    target_speed_mps: float
    reason: str
    finished: bool = False
    force_stop: bool = False

