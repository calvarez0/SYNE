"""
Innovation tracking for SYNE.

Tracks historical origins of genes for crossover alignment,
following NEAT's innovation number scheme but extended for symbiogenesis.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class InnovationTracker:
    """
    Tracks innovation numbers for nodes and connections.

    In SYNE, innovation numbers serve two purposes:
    1. Enable crossover alignment (like NEAT)
    2. Track lineage through fusion events for analysis

    Each node and connection gets a unique innovation number when first created.
    During fusion, original innovation numbers are preserved, maintaining
    the historical identity of genes from merged genomes.
    """

    # Current innovation counters
    node_innovation: int = 0
    connection_innovation: int = 0

    # Cache for connection innovations: (in_node, out_node) -> innovation number
    # This ensures the same structural innovation gets the same number
    connection_cache: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Track fusion events for lineage analysis
    # fusion_id -> (parent1_id, parent2_id)
    fusion_history: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    fusion_counter: int = 0

    def get_node_innovation(self) -> int:
        """Get a new innovation number for a node."""
        self.node_innovation += 1
        return self.node_innovation

    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        """
        Get innovation number for a connection.

        If this connection structure has been seen before in this generation,
        return the cached innovation number. Otherwise, create a new one.
        """
        key = (in_node, out_node)
        if key not in self.connection_cache:
            self.connection_innovation += 1
            self.connection_cache[key] = self.connection_innovation
        return self.connection_cache[key]

    def clear_generation_cache(self) -> None:
        """
        Clear the connection cache at the end of a generation.

        This allows the same structural innovation in different generations
        to get different innovation numbers, which is important for
        distinguishing convergent evolution from shared ancestry.
        """
        self.connection_cache.clear()

    def record_fusion(self, parent1_id: int, parent2_id: int) -> int:
        """
        Record a fusion event and return a fusion ID.

        This allows tracking the symbiogenetic history of genomes.
        """
        self.fusion_counter += 1
        self.fusion_history[self.fusion_counter] = (parent1_id, parent2_id)
        return self.fusion_counter

    def get_fusion_ancestry(self, fusion_id: int) -> Optional[Tuple[int, int]]:
        """Get the parent IDs for a fusion event."""
        return self.fusion_history.get(fusion_id)

    def reset(self) -> None:
        """Reset all innovation tracking (for new runs)."""
        self.node_innovation = 0
        self.connection_innovation = 0
        self.connection_cache.clear()
        self.fusion_history.clear()
        self.fusion_counter = 0


# Global innovation tracker instance
_global_tracker: Optional[InnovationTracker] = None


def get_innovation_tracker() -> InnovationTracker:
    """Get the global innovation tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = InnovationTracker()
    return _global_tracker


def reset_innovation_tracker() -> None:
    """Reset the global innovation tracker."""
    global _global_tracker
    _global_tracker = InnovationTracker()
