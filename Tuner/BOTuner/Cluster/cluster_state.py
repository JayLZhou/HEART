from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ClusterState:
    cluster_id: int
    shared_plan: Dict[str, Any] | None = None
    shared_reasoning: str | None = None
    last_updated_round: int = 0
    completed_trial_numbers: list[int] = field(default_factory=list)


class LGBOClusterStateStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._states: Dict[int, ClusterState] = {}
        if self.path and self.path.exists():
            self._load()

    def get(self, cluster_id: int) -> ClusterState:
        if cluster_id not in self._states:
            self._states[cluster_id] = ClusterState(cluster_id=cluster_id)
        return self._states[cluster_id]

    def update(
        self,
        cluster_id: int,
        *,
        shared_plan: Dict[str, Any] | None,
        shared_reasoning: str | None,
        round_id: int,
        trial_numbers: list[int],
    ) -> ClusterState:
        state = self.get(cluster_id)
        state.shared_plan = shared_plan
        state.shared_reasoning = shared_reasoning
        state.last_updated_round = round_id
        state.completed_trial_numbers = list(trial_numbers)
        self.flush()
        return state

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(cluster_id): asdict(state)
            for cluster_id, state in sorted(self._states.items(), key=lambda item: item[0])
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _load(self) -> None:
        payload = json.loads(self.path.read_text())
        for raw_cluster_id, item in payload.items():
            self._states[int(raw_cluster_id)] = ClusterState(**item)
