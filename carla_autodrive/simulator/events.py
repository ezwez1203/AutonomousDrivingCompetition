"""Runtime event sensors used by Phase 6 scoring."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla


@dataclass(slots=True)
class CollisionSnapshot:
    count: int = 0
    max_impulse: float = 0.0
    last_actor_type: str | None = None


class CollisionMonitor:
    """Collect cumulative collision events from CARLA's collision sensor."""

    def __init__(self):
        self.count = 0
        self.max_impulse = 0.0
        self.last_actor_type: str | None = None

    def spawn(self, world, parent) -> "carla.Actor":
        import carla

        bp = world.get_blueprint_library().find("sensor.other.collision")
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=parent)
        actor.listen(self._on_collision)
        return actor

    def snapshot(self) -> CollisionSnapshot:
        return CollisionSnapshot(
            count=int(self.count),
            max_impulse=float(self.max_impulse),
            last_actor_type=self.last_actor_type,
        )

    def _on_collision(self, event) -> None:
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x * impulse.x + impulse.y * impulse.y + impulse.z * impulse.z)
        self.count += 1
        self.max_impulse = max(self.max_impulse, float(intensity))
        other = getattr(event, "other_actor", None)
        self.last_actor_type = getattr(other, "type_id", None)
