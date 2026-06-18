"""PyTorch baselines for Phase 3 synthetic perception datasets."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


TRAFFIC_CLASSES = ("red", "yellow", "green", "off")


@dataclass(slots=True)
class TrainConfig:
    dataset_dir: Path
    out_dir: Path
    tasks: tuple[str, ...]
    epochs: int = 8
    batch_size: int = 16
    lr: float = 1e-3
    val_ratio: float = 0.2
    seed: int = 7
    image_width: int = 160
    image_height: int = 120
    max_samples: int | None = None
    device: str = "auto"
    num_workers: int = 0


class Phase3DatasetIndex:
    """Index labels.jsonl records and resolve snapshot paths."""

    def __init__(self, dataset_dir: str | Path, *, max_samples: int | None = None):
        self.dataset_dir = Path(dataset_dir)
        label_path = self.dataset_dir / "labels.jsonl"
        if not label_path.exists():
            raise FileNotFoundError(f"labels.jsonl not found: {label_path}")
        rows: list[dict[str, Any]] = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
        if not rows:
            raise ValueError(f"no labels found in {label_path}")
        self.rows = rows

    def snapshot_path(self, row: dict[str, Any]) -> Path:
        return self.dataset_dir / row["snapshot_path"]


class LaneDataset(Dataset):
    def __init__(self, index: Phase3DatasetIndex, image_size: tuple[int, int]):
        self.index = index
        self.image_size = image_size
        self.rows = [row for row in index.rows if row.get("lane", {}).get("detected")]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = _load_camera_tensor(self.index.snapshot_path(row), self.image_size)
        lane = row["lane"]
        target = torch.tensor(
            [
                float(lane.get("center_error_norm", 0.0)),
                float(lane.get("heading_error_rad", 0.0)),
            ],
            dtype=torch.float32,
        )
        return image, target


class TrafficDataset(Dataset):
    def __init__(self, index: Phase3DatasetIndex, image_size: tuple[int, int]):
        self.index = index
        self.image_size = image_size
        self.class_to_idx = {name: idx for idx, name in enumerate(TRAFFIC_CLASSES)}
        self.rows = [
            row
            for row in index.rows
            if row.get("traffic_light", {}).get("detected")
            and row.get("traffic_light", {}).get("state") in self.class_to_idx
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = _load_camera_tensor(self.index.snapshot_path(row), self.image_size)
        label = self.class_to_idx[row["traffic_light"]["state"]]
        return image, torch.tensor(label, dtype=torch.long)


class ObstacleDataset(Dataset):
    def __init__(self, index: Phase3DatasetIndex, grid_size: tuple[int, int] = (64, 64)):
        self.index = index
        self.grid_size = grid_size
        self.rows = list(index.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        grid = _load_obstacle_grid(self.index.snapshot_path(row), self.grid_size)
        obstacles = row.get("obstacles") or []
        if obstacles:
            nearest = min(obstacles, key=lambda item: float(item.get("distance_m", 1e9)))
            local = nearest.get("vehicle_local", {})
            target = torch.tensor(
                [
                    1.0,
                    _normalize(float(local.get("x", 0.0)), -1.0, 8.0),
                    _normalize(float(local.get("y", 0.0)), -3.0, 3.0),
                    min(float(nearest.get("distance_m", 0.0)) / 8.0, 1.0),
                ],
                dtype=torch.float32,
            )
        else:
            target = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        return grid, target


class ImageBackbone(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LaneBaselineNet(nn.Module):
    """Small CNN that regresses lane center error and heading error from RGB."""

    def __init__(self):
        super().__init__()
        self.backbone = ImageBackbone(64)
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class TrafficBaselineNet(nn.Module):
    """Small CNN traffic-light color classifier from RGB."""

    def __init__(self, num_classes: int = len(TRAFFIC_CLASSES)):
        super().__init__()
        self.backbone = ImageBackbone(64)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ObstacleBaselineNet(nn.Module):
    """2D occupancy-grid CNN for obstacle presence and nearest-obstacle pose."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_phase3_baselines(cfg: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    index = Phase3DatasetIndex(cfg.dataset_dir, max_samples=cfg.max_samples)
    device = _resolve_device(cfg.device)
    image_size = (cfg.image_width, cfg.image_height)

    results: dict[str, Any] = {
        "dataset_dir": str(cfg.dataset_dir),
        "out_dir": str(cfg.out_dir),
        "device": str(device),
        "total_records": len(index.rows),
        "tasks": {},
    }
    selected = _expand_tasks(cfg.tasks)
    if "lane" in selected:
        results["tasks"]["lane"] = _train_lane(index, cfg, image_size, device)
    if "traffic" in selected:
        results["tasks"]["traffic"] = _train_traffic(index, cfg, image_size, device)
    if "obstacle" in selected:
        results["tasks"]["obstacle"] = _train_obstacle(index, cfg, device)

    report_path = cfg.out_dir / "phase3_baseline_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return results


def _train_lane(index: Phase3DatasetIndex, cfg: TrainConfig, image_size: tuple[int, int], device: torch.device) -> dict[str, Any]:
    dataset = LaneDataset(index, image_size)
    if len(dataset) < 2:
        return {"status": "skipped", "reason": "need at least 2 lane-labeled samples", "samples": len(dataset)}
    train_loader, val_loader, split = _make_loaders(dataset, cfg)
    model = LaneBaselineNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss()
    history = []
    best = math.inf
    best_path = cfg.out_dir / "lane_baseline.pt"
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_epoch_regression(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mae = _eval_lane(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae})
        if val_loss <= best:
            best = val_loss
            _save_checkpoint(best_path, model, cfg, "lane", {"val_loss": val_loss, "val_mae": val_mae})
    return {"status": "trained", "samples": len(dataset), "split": split, "best_val_loss": best, "checkpoint": str(best_path), "history": history}


def _train_traffic(index: Phase3DatasetIndex, cfg: TrainConfig, image_size: tuple[int, int], device: torch.device) -> dict[str, Any]:
    dataset = TrafficDataset(index, image_size)
    classes = sorted({row["traffic_light"]["state"] for row in dataset.rows})
    if len(dataset) < 2 or len(classes) < 2:
        return {
            "status": "skipped",
            "reason": "need at least 2 samples across at least 2 traffic-light classes",
            "samples": len(dataset),
            "classes": classes,
        }
    train_loader, val_loader, split = _make_loaders(dataset, cfg)
    model = TrafficBaselineNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best = math.inf
    best_path = cfg.out_dir / "traffic_baseline.pt"
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_epoch_classification(model, train_loader, optimizer, loss_fn, device)
        val_loss, accuracy = _eval_traffic(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": accuracy})
        if val_loss <= best:
            best = val_loss
            _save_checkpoint(
                best_path,
                model,
                cfg,
                "traffic",
                {"val_loss": val_loss, "val_accuracy": accuracy, "classes": TRAFFIC_CLASSES},
            )
    return {"status": "trained", "samples": len(dataset), "split": split, "best_val_loss": best, "checkpoint": str(best_path), "classes": TRAFFIC_CLASSES, "history": history}


def _train_obstacle(index: Phase3DatasetIndex, cfg: TrainConfig, device: torch.device) -> dict[str, Any]:
    dataset = ObstacleDataset(index)
    positives = sum(1 for row in dataset.rows if row.get("obstacles"))
    if len(dataset) < 2:
        return {"status": "skipped", "reason": "need at least 2 obstacle samples", "samples": len(dataset), "positives": positives}
    if positives < 1:
        return {"status": "skipped", "reason": "need at least 1 positive obstacle sample", "samples": len(dataset), "positives": positives}
    train_loader, val_loader, split = _make_loaders(dataset, cfg)
    model = ObstacleBaselineNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []
    best = math.inf
    best_path = cfg.out_dir / "obstacle_baseline.pt"
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_epoch_obstacle(model, train_loader, optimizer, device)
        val_loss, metrics = _eval_obstacle(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        if val_loss <= best:
            best = val_loss
            _save_checkpoint(best_path, model, cfg, "obstacle", {"val_loss": val_loss, **metrics})
    return {"status": "trained", "samples": len(dataset), "positives": positives, "split": split, "best_val_loss": best, "checkpoint": str(best_path), "history": history}


def _make_loaders(dataset: Dataset, cfg: TrainConfig):
    n = len(dataset)
    indices = list(range(n))
    random.Random(cfg.seed).shuffle(indices)
    val_count = max(1, int(round(n * cfg.val_ratio))) if n > 1 else 0
    val_count = min(val_count, n - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, {"train": len(train_indices), "val": len(val_indices)}


def _train_epoch_regression(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(x)
        count += len(x)
    return total / max(1, count)


def _train_epoch_classification(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(x)
        count += len(x)
    return total / max(1, count)


def _train_epoch_obstacle(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = _obstacle_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(x)
        count += len(x)
    return total / max(1, count)


@torch.no_grad()
def _eval_lane(model, loader, device) -> tuple[float, dict[str, float]]:
    model.eval()
    loss_fn = nn.SmoothL1Loss(reduction="sum")
    total_loss = 0.0
    total_abs = torch.zeros(2, device=device)
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total_loss += float(loss_fn(pred, y).item())
        total_abs += torch.abs(pred - y).sum(dim=0)
        count += len(x)
    denom = max(1, count)
    mae = (total_abs / denom).detach().cpu().tolist()
    return total_loss / denom, {"center_error_norm": float(mae[0]), "heading_error_rad": float(mae[1])}


@torch.no_grad()
def _eval_traffic(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * len(x)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        count += len(x)
    return total_loss / max(1, count), correct / max(1, count)


@torch.no_grad()
def _eval_obstacle(model, loader, device) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    count = 0
    correct = 0
    positives = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total_loss += float(_obstacle_loss(pred, y).item()) * len(x)
        pred_present = torch.sigmoid(pred[:, 0]) >= 0.5
        true_present = y[:, 0] >= 0.5
        correct += int((pred_present == true_present).sum().item())
        positives += int(true_present.sum().item())
        count += len(x)
    return total_loss / max(1, count), {"presence_accuracy": correct / max(1, count), "val_positives": positives}


def _obstacle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    presence_loss = nn.functional.binary_cross_entropy_with_logits(pred[:, 0], target[:, 0])
    mask = target[:, 0:1]
    if float(mask.sum().item()) > 0.0:
        pose_loss = nn.functional.smooth_l1_loss(torch.sigmoid(pred[:, 1:4]) * mask, target[:, 1:4] * mask)
    else:
        pose_loss = pred[:, 1:4].sum() * 0.0
    return presence_loss + pose_loss


def _load_camera_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    with np.load(path) as data:
        if "camera_bgra" not in data:
            raise KeyError(f"camera_bgra missing in {path}")
        bgra = data["camera_bgra"]
    bgr = bgra[:, :, :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, image_size, interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _load_obstacle_grid(path: Path, grid_size: tuple[int, int]) -> torch.Tensor:
    height, width = grid_size
    grid = np.zeros((2, height, width), dtype=np.float32)
    with np.load(path) as data:
        lidar = data["lidar_points"] if "lidar_points" in data else np.empty((0, 4), dtype=np.float32)
        radar = data["radar_points"] if "radar_points" in data else np.empty((0, 8), dtype=np.float32)
    _points_to_grid(grid[0], lidar[:, :2] if len(lidar) else np.empty((0, 2), dtype=np.float32), x_range=(-1.0, 8.0), y_range=(-3.0, 3.0))
    _points_to_grid(grid[1], radar[:, :2] if len(radar) else np.empty((0, 2), dtype=np.float32), x_range=(-1.0, 8.0), y_range=(-3.0, 3.0))
    return torch.from_numpy(grid)


def _points_to_grid(channel: np.ndarray, points_xy: np.ndarray, *, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
    if len(points_xy) == 0:
        return
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
    if not np.any(mask):
        return
    x = x[mask]
    y = y[mask]
    rows = ((x - x_range[0]) / (x_range[1] - x_range[0]) * (channel.shape[0] - 1)).astype(np.int32)
    cols = ((y - y_range[0]) / (y_range[1] - y_range[0]) * (channel.shape[1] - 1)).astype(np.int32)
    np.add.at(channel, (rows, cols), 1.0)
    np.clip(channel, 0.0, 5.0, out=channel)
    channel /= 5.0


def _normalize(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _expand_tasks(tasks: tuple[str, ...]) -> set[str]:
    selected = set()
    for task in tasks:
        if task == "all":
            selected.update({"lane", "traffic", "obstacle"})
        else:
            selected.add(task)
    return selected


def _save_checkpoint(path: Path, model: nn.Module, cfg: TrainConfig, task: str, metrics: dict[str, Any]) -> None:
    torch.save(
        {
            "task": task,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": {
                "image_width": cfg.image_width,
                "image_height": cfg.image_height,
                "traffic_classes": TRAFFIC_CLASSES,
            },
        },
        path,
    )
