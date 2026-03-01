"""
Innovation tracking for SYNE v5.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class InnovationTracker:
    """Tracks innovation numbers for nodes and connections."""

    node_innovation: int = 0
    connection_innovation: int = 0
    connection_cache: Dict[Tuple[int, int], int] = field(default_factory=dict)
    fusion_history: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    fusion_counter: int = 0

    def get_node_innovation(self) -> int:
        self.node_innovation += 1
        return self.node_innovation

    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self.connection_cache:
            self.connection_innovation += 1
            self.connection_cache[key] = self.connection_innovation
        return self.connection_cache[key]

    def clear_generation_cache(self) -> None:
        self.connection_cache.clear()

    def record_fusion(self, parent1_id: int, parent2_id: int) -> int:
        self.fusion_counter += 1
        self.fusion_history[self.fusion_counter] = (parent1_id, parent2_id)
        return self.fusion_counter

    def get_fusion_ancestry(self, fusion_id: int) -> Optional[Tuple[int, int]]:
        return self.fusion_history.get(fusion_id)

    def reset(self) -> None:
        self.node_innovation = 0
        self.connection_innovation = 0
        self.connection_cache.clear()
        self.fusion_history.clear()
        self.fusion_counter = 0


_global_tracker: Optional[InnovationTracker] = None


def get_innovation_tracker() -> InnovationTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = InnovationTracker()
    return _global_tracker


def reset_innovation_tracker() -> None:
    global _global_tracker
    _global_tracker = InnovationTracker()
