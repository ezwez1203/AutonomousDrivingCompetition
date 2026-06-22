"""Live camera monitoring and signal-compliance recording helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .ultrasonic import UltrasonicReading


@dataclass(slots=True)
class CameraMonitor:
    """Optional OpenCV front/rear camera display."""

    display: bool = False
    window_prefix: str = "carla"
    _cv2: object | None = field(default=None, init=False, repr=False)
    _disabled_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.display:
            return
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local GUI deps
            self._disabled_reason = f"OpenCV display unavailable: {exc}"
            self.display = False
            return
        self._cv2 = cv2

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def step(self, camera_frames: dict[str, tuple[int, np.ndarray | None]]) -> None:
        if not self.display or self._cv2 is None:
            return
        try:
            for name, (_frame, image) in camera_frames.items():
                if image is None:
                    continue
                bgr = image[:, :, :3]
                self._cv2.imshow(f"{self.window_prefix}:{name}", bgr)
            self._cv2.waitKey(1)
        except Exception as exc:  # pragma: no cover - depends on local GUI backend
            self._disabled_reason = f"OpenCV display disabled after runtime error: {exc}"
            self.display = False

    def close(self) -> None:
        if self._cv2 is None:
            return
        try:
            self._cv2.destroyAllWindows()
        except Exception:
            pass


@dataclass(slots=True)
class SignalComplianceRecorder:
    """Record frames and metadata while the FSM requires a traffic-light stop."""

    out_dir: str | Path
    sample_every_ticks: int = 5
    log_events: bool = False
    _records: list[dict[str, object]] = field(default_factory=list, init=False)
    _saved_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def step(
        self,
        *,
        tick: int,
        sim_time_s: float,
        state: str,
        decision_reason: str,
        stop_required: bool,
        stop_violation: bool,
        speed_mps: float,
        camera_frames: dict[str, tuple[int, np.ndarray | None]],
        ultrasonic: UltrasonicReading | None,
    ) -> dict[str, object] | None:
        should_record = stop_required or state == "TRAFFIC_STOP" or decision_reason.startswith("traffic_")
        if not should_record:
            return None
        every = max(1, int(self.sample_every_ticks))
        if tick % every != 0 and not stop_violation:
            return None

        payload: dict[str, object] = {
            "tick": int(tick),
            "sim_time_s": float(sim_time_s),
            "state": state,
            "decision_reason": decision_reason,
            "stop_required": bool(stop_required),
            "stop_violation": bool(stop_violation),
            "speed_mps": float(speed_mps),
            "camera_frames": {name: int(frame) for name, (frame, _img) in camera_frames.items()},
            "ultrasonic_front": None if ultrasonic is None else {
                "frame": ultrasonic.frame,
                "distance_m": ultrasonic.distance_m,
                "actor_id": ultrasonic.actor_id,
                "actor_type": ultrasonic.actor_type,
            },
        }

        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(json.dumps(payload, ensure_ascii=False)),
        }
        for name, (_frame, image) in camera_frames.items():
            if image is not None:
                arrays[f"{name}_bgra"] = image
        frame_path = self.out_dir / f"signal_stop_{self._saved_count:04d}_tick{tick:06d}.npz"
        np.savez_compressed(frame_path, **arrays)
        payload["frame_npz"] = str(frame_path)
        self._records.append(payload)
        self._saved_count += 1
        if self.log_events:
            print(
                "signal_record "
                f"tick={tick} t={sim_time_s:.2f}s state={state} reason={decision_reason} "
                f"speed={speed_mps:.2f}m/s violation={stop_violation} file={frame_path}"
            )
        return payload

    def write_manifest(self) -> Path:
        manifest = self.out_dir / "signal_compliance_log.json"
        manifest.write_text(json.dumps(self._records, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @property
    def saved_count(self) -> int:
        return self._saved_count


@dataclass(slots=True)
class RunVideoRecorder:
    """Record full-run front/rear monitoring camera videos."""

    out_dir: str | Path
    sample_every_ticks: int = 1
    fps: float = 20.0
    codec: str = "mp4v"
    _cv2: object | None = field(default=None, init=False, repr=False)
    _writers: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _paths: dict[str, str] = field(default_factory=dict, init=False)
    _frames_written: dict[str, int] = field(default_factory=dict, init=False)
    _records: list[dict[str, object]] = field(default_factory=list, init=False)
    _disabled_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local cv2 install
            self._disabled_reason = f"OpenCV video recording unavailable: {exc}"
            return
        self._cv2 = cv2

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def step(
        self,
        *,
        tick: int,
        sim_time_s: float,
        camera_frames: dict[str, tuple[int, np.ndarray | None]],
    ) -> None:
        if self._cv2 is None or self._disabled_reason is not None:
            return
        every = max(1, int(self.sample_every_ticks))
        if int(tick) % every != 0:
            return
        for name, (sensor_frame, image) in camera_frames.items():
            if image is None:
                continue
            writer = self._writer_for(name, image)
            if writer is None:
                continue
            bgr = image[:, :, :3]
            writer.write(bgr)
            self._frames_written[name] = self._frames_written.get(name, 0) + 1
            self._records.append({
                "camera": name,
                "tick": int(tick),
                "sim_time_s": float(sim_time_s),
                "sensor_frame": int(sensor_frame),
                "video_path": self._paths.get(name),
            })

    def _writer_for(self, name: str, image: np.ndarray):
        if name in self._writers:
            return self._writers[name]
        if self._cv2 is None:
            return None
        height, width = image.shape[:2]
        fourcc = self._cv2.VideoWriter_fourcc(*self.codec)
        safe_name = name.replace("/", "_").replace(" ", "_")
        path = Path(self.out_dir) / f"{safe_name}.mp4"
        writer = self._cv2.VideoWriter(str(path), fourcc, float(self.fps), (int(width), int(height)))
        if not writer.isOpened():
            self._disabled_reason = f"OpenCV VideoWriter failed to open: {path}"
            return None
        self._writers[name] = writer
        self._paths[name] = str(path)
        self._frames_written[name] = 0
        return writer

    def close(self) -> Path:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        manifest = Path(self.out_dir) / "monitor_video_manifest.json"
        payload = {
            "disabled_reason": self._disabled_reason,
            "videos": self._paths,
            "frames_written": self._frames_written,
            "sample_every_ticks": int(self.sample_every_ticks),
            "fps": float(self.fps),
            "records": self._records,
        }
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
