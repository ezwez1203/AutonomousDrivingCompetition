"""RGB camera wrapper."""
from __future__ import annotations

import numpy as np

from .base import SensorBase


class RGBCamera(SensorBase):
    """Convert sensor.camera.rgb frames to BGRA numpy arrays with shape (H, W, 4)."""

    def __init__(self, cfg: dict, name: str = "rgb_camera"):
        super().__init__(name, cfg)
        self.width = int(cfg["image_size_x"])
        self.height = int(cfg["image_size_y"])

    def _configure_blueprint(self, bp) -> None:
        bp.set_attribute("image_size_x", str(self.cfg["image_size_x"]))
        bp.set_attribute("image_size_y", str(self.cfg["image_size_y"]))
        bp.set_attribute("fov", str(self.cfg["fov"]))
        if "sensor_tick" in self.cfg:
            bp.set_attribute("sensor_tick", str(self.cfg["sensor_tick"]))

    def _on_data(self, image) -> None:
        # carla.Image.raw_data → BGRA
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self._store(image.frame, array)

    def summary(self) -> str:
        """one-line summary for console output."""
        frame, img = self.get_latest()
        if img is None:
            return "RGB  : (waiting for data)"
        bgr = img[:, :, :3]
        return (f"{self.name}: frame={frame} shape={img.shape} "
                f"mean_BGR=({bgr[...,0].mean():.0f},"
                f"{bgr[...,1].mean():.0f},{bgr[...,2].mean():.0f})")
