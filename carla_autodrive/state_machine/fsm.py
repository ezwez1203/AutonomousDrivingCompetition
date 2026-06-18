"""Mission state machine for Phase 5 competition flows."""
from __future__ import annotations

from dataclasses import dataclass

from carla_autodrive.maps import TrackSpec

from .types import MissionDecision, MissionMode, MissionState


@dataclass(slots=True)
class MissionFSMConfig:
    mode: MissionMode
    target_speed_mps: float = 2.0
    total_laps: int = 2
    crosswalk_stop_before_mm: float = 900.0
    crosswalk_finish_after_mm: float = 1200.0
    green_after_sec: float = 3.0
    parking_hold_sec: float = 3.2
    parking_stop_speed_mps: float = 0.25


@dataclass(slots=True)
class MissionContext:
    tick: int
    sim_time_s: float
    speed_mps: float
    route_index: int | None = None
    route_length: int | None = None
    route_s_m: float | None = None
    route_finished: bool = False
    green_signal: bool = False


class MissionFSM:
    """Small deterministic FSM that selects mission-level speed/stop behavior."""

    def __init__(self, spec: TrackSpec, cfg: MissionFSMConfig):
        self.spec = spec
        self.cfg = cfg
        self.state = MissionState.IDLE
        self.lap_count = 0
        self._last_route_index: int | None = None
        self._stopped_since_s: float | None = None
        self._parking_hold_start_s: float | None = None
        self._crosswalk_s_m = self._element_s_m("crosswalk", default_mm=17000.0)
        self._stop_s_m = self._crosswalk_s_m - spec.mm(cfg.crosswalk_stop_before_mm)
        self._finish_s_m = self._crosswalk_s_m + spec.mm(cfg.crosswalk_finish_after_mm)

    def start(self) -> MissionDecision:
        if self.cfg.mode == MissionMode.TIME_TRIAL:
            self.state = MissionState.TIME_TRIAL
            return self._decision("time_trial_start")
        if self.cfg.mode == MissionMode.OBSTACLE_SIGNAL:
            self.state = MissionState.OBSTACLE_AVOID
            return self._decision("obstacle_signal_start")
        if self.cfg.mode == MissionMode.PARKING:
            self.state = MissionState.PARKING
            return self._decision("parking_start")
        self.state = MissionState.FAILED
        return self._decision("unsupported_mode", target_speed=0.0, finished=True)

    def update(self, ctx: MissionContext) -> MissionDecision:
        if self.state == MissionState.IDLE:
            self.start()
        if self.state in {MissionState.FINISHED, MissionState.FAILED}:
            return self._decision(self.state.value.lower(), target_speed=0.0, finished=True)

        self._update_lap_count(ctx)

        if self.cfg.mode == MissionMode.TIME_TRIAL:
            return self._update_time_trial()
        if self.cfg.mode == MissionMode.OBSTACLE_SIGNAL:
            return self._update_obstacle_signal(ctx)
        if self.cfg.mode == MissionMode.PARKING:
            return self._update_parking(ctx)
        self.state = MissionState.FAILED
        return self._decision("unsupported_mode", target_speed=0.0, finished=True)

    def summary(self) -> str:
        return f"fsm state={self.state.value} laps={self.lap_count}"

    def _update_time_trial(self) -> MissionDecision:
        if self.lap_count >= self.cfg.total_laps:
            self.state = MissionState.FINISHED
            return self._decision("time_trial_complete", target_speed=0.0, finished=True)
        return self._decision(f"time_trial_lap_{self.lap_count + 1}")

    def _update_obstacle_signal(self, ctx: MissionContext) -> MissionDecision:
        route_s_m = ctx.route_s_m
        if self.state == MissionState.OBSTACLE_AVOID and route_s_m is not None and route_s_m >= self._stop_s_m:
            self.state = MissionState.TRAFFIC_STOP
            self._stopped_since_s = None

        if self.state == MissionState.TRAFFIC_STOP:
            if ctx.speed_mps <= 0.25:
                if self._stopped_since_s is None:
                    self._stopped_since_s = ctx.sim_time_s
                stopped_for = ctx.sim_time_s - self._stopped_since_s
                green = ctx.green_signal or stopped_for >= self.cfg.green_after_sec
                if green:
                    self.state = MissionState.LANE_FOLLOW
                    return self._decision("traffic_green")
                return self._decision("traffic_red_stop", target_speed=0.0, force_stop=True)
            return self._decision("traffic_red_decelerate", target_speed=0.0, force_stop=True)

        if self.state == MissionState.LANE_FOLLOW and route_s_m is not None and route_s_m >= self._finish_s_m:
            self.state = MissionState.FINISHED
            return self._decision("obstacle_signal_complete", target_speed=0.0, finished=True)

        return self._decision(self.state.value.lower())

    def _update_parking(self, ctx: MissionContext) -> MissionDecision:
        if ctx.route_finished and ctx.speed_mps <= self.cfg.parking_stop_speed_mps:
            if self._parking_hold_start_s is None:
                self._parking_hold_start_s = ctx.sim_time_s
            held = ctx.sim_time_s - self._parking_hold_start_s
            if held >= self.cfg.parking_hold_sec:
                self.state = MissionState.FINISHED
                return self._decision("parking_hold_complete", target_speed=0.0, finished=True)
            return self._decision("parking_hold", target_speed=0.0, force_stop=True)
        if ctx.route_finished:
            return self._decision("parking_endpoint_stop", target_speed=0.0, force_stop=True)
        self._parking_hold_start_s = None
        return self._decision("parking_maneuver")

    def _update_lap_count(self, ctx: MissionContext) -> None:
        if ctx.route_index is None or ctx.route_length is None or ctx.route_length <= 0:
            return
        if self._last_route_index is not None:
            wrap_threshold = max(4, int(ctx.route_length * 0.5))
            if self._last_route_index - ctx.route_index > wrap_threshold:
                self.lap_count += 1
        self._last_route_index = ctx.route_index

    def _decision(
        self,
        reason: str,
        *,
        target_speed: float | None = None,
        finished: bool = False,
        force_stop: bool = False,
    ) -> MissionDecision:
        speed = self.cfg.target_speed_mps if target_speed is None else target_speed
        return MissionDecision(
            state=self.state,
            target_speed_mps=float(speed),
            reason=reason,
            finished=finished,
            force_stop=force_stop,
        )

    def _element_s_m(self, key: str, *, default_mm: float) -> float:
        elements = self.spec.cfg.get("elements")
        if isinstance(elements, dict):
            item = elements.get(key)
            if isinstance(item, dict) and "s" in item:
                return self.spec.mm(float(item["s"]))
        return self.spec.mm(default_mm)
