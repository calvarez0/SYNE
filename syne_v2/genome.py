"""
Genome representation for SYNE.

A genome encodes a neural network topology and weights using node genes
and connection genes, similar to NEAT. However, SYNE genomes do not
undergo mutation - they only change through fusion (symbiogenesis)
and crossover.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from copy import deepcopy
import random
import math

from syne_v2.config import Config, GenomeConfig, ACTIVATION_FUNCTIONS
from syne_v2.innovation import get_innovation_tracker, InnovationTracker


@dataclass
class NodeGene:
    """
    Represents a single node (neuron) in the network.

    Attributes:
        key: Unique identifier for this node
        node_type: 'input', 'output', or 'hidden'
        bias: Bias value for the node
        activation: Activation function name
        response: Response multiplier (scaling factor)
    """
    key: int
    node_type: str  # 'input', 'output', 'hidden'
    bias: float = 0.0
    activation: str = 'sigmoid'
    response: float = 1.0

    # Track which original genome this node came from (for fusion analysis)
    origin_genome_id: Optional[int] = None

    def copy(self) -> 'NodeGene':
        """Create a deep copy of this node gene."""
        return NodeGene(
            key=self.key,
            node_type=self.node_type,
            bias=self.bias,
            activation=self.activation,
            response=self.response,
            origin_genome_id=self.origin_genome_id,
        )

    def crossover(self, other: 'NodeGene') -> 'NodeGene':
        """
        Create a child node by combining this node with another.

        For matching genes (same key), randomly inherit attributes from
        either parent.
        """
        assert self.key == other.key
        assert self.node_type == other.node_type

        return NodeGene(
            key=self.key,
            node_type=self.node_type,
            bias=random.choice([self.bias, other.bias]),
            activation=random.choice([self.activation, other.activation]),
            response=random.choice([self.response, other.response]),
            origin_genome_id=None,  # Child has new identity
        )


@dataclass
class ConnectionGene:
    """
    Represents a connection between two nodes.

    Attributes:
        key: Tuple of (input_node, output_node) keys
        weight: Connection weight
        enabled: Whether the connection is active
        innovation: Innovation number for crossover alignment
    """
    key: Tuple[int, int]  # (in_node, out_node)
    weight: float = 0.0
    enabled: bool = True
    innovation: int = 0

    # Track origin for fusion analysis
    origin_genome_id: Optional[int] = None

    @property
    def in_node(self) -> int:
        return self.key[0]

    @property
    def out_node(self) -> int:
        return self.key[1]

    def copy(self) -> 'ConnectionGene':
        """Create a deep copy of this connection gene."""
        return ConnectionGene(
            key=self.key,
            weight=self.weight,
            enabled=self.enabled,
            innovation=self.innovation,
            origin_genome_id=self.origin_genome_id,
        )

    def crossover(self, other: 'ConnectionGene', blend: bool = True) -> 'ConnectionGene':
        """
        Create a child connection by combining this connection with another.

        For matching genes (same innovation number), either randomly inherit
        the weight from one parent, or blend the weights.

        Args:
            blend: If True, sometimes blend weights instead of choosing one.
                   This creates variation without mutation.
        """
        assert self.innovation == other.innovation

        # Weight inheritance with optional blending
        if blend and random.random() < 0.5:
            # Blend weights (like BLX-alpha crossover)
            alpha = random.uniform(-0.1, 1.1)
            weight = alpha * self.weight + (1 - alpha) * other.weight
        else:
            # Standard: inherit from one parent
            weight = random.choice([self.weight, other.weight])

        return ConnectionGene(
            key=self.key,
            weight=weight,
            enabled=self.enabled if self.enabled == other.enabled else random.choice([True, False]),
            innovation=self.innovation,
            origin_genome_id=None,
        )


class Genome:
    """
    A genome representing a neural network.

    Contains node genes and connection genes that define the network
    topology and weights. Genomes evolve through fusion (symbiogenesis)
    and crossover, but NOT through mutation.
    """

    # Class-level genome ID counter
    _next_id: int = 0

    def __init__(self, genome_id: Optional[int] = None):
        """Initialize an empty genome."""
        if genome_id is None:
            genome_id = Genome._next_id
            Genome._next_id += 1

        self.id: int = genome_id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[Tuple[int, int], ConnectionGene] = {}
        self.fitness: Optional[float] = None

        # Track fusion history
        self.parent_ids: Optional[Tuple[int, int]] = None
        self.fusion_generation: Optional[int] = None
        self.is_fused: bool = False

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the genome ID counter."""
        cls._next_id = 0

    def configure_new(self, config: GenomeConfig) -> None:
        """
        Initialize a new genome with the configured structure.

        Creates input and output nodes, and optionally hidden nodes,
        with connections based on the connectivity setting.
        """
        tracker = get_innovation_tracker()

        # Create input nodes (negative keys by convention)
        for i in range(config.num_inputs):
            node_key = -(i + 1)
            self.nodes[node_key] = NodeGene(
                key=node_key,
                node_type='input',
                bias=0.0,  # Input nodes don't use bias
                activation='identity',
                origin_genome_id=self.id,
            )

        # Create output nodes (keys starting at 0)
        for i in range(config.num_outputs):
            node_key = i
            self.nodes[node_key] = NodeGene(
                key=node_key,
                node_type='output',
                bias=random.gauss(config.bias_init_mean, config.bias_init_std),
                activation=config.activation_default,
                origin_genome_id=self.id,
            )

        # Create initial connections based on connectivity setting
        input_keys = [k for k in self.nodes if self.nodes[k].node_type == 'input']
        output_keys = [k for k in self.nodes if self.nodes[k].node_type == 'output']

        if config.initial_connectivity == 'full':
            # Connect all inputs to all outputs
            for in_key in input_keys:
                for out_key in output_keys:
                    self._add_connection(in_key, out_key, config, tracker)

        elif config.initial_connectivity == 'partial':
            # Randomly connect some inputs to outputs
            for in_key in input_keys:
                for out_key in output_keys:
                    if random.random() < config.partial_connectivity_prob:
                        self._add_connection(in_key, out_key, config, tracker)

        # 'none' leaves no initial connections

    def _add_connection(
        self,
        in_node: int,
        out_node: int,
        config: GenomeConfig,
        tracker: InnovationTracker
    ) -> None:
        """Add a connection between two nodes."""
        key = (in_node, out_node)
        if key in self.connections:
            return

        weight = random.gauss(config.weight_init_mean, config.weight_init_std)
        weight = max(config.weight_min, min(config.weight_max, weight))

        innovation = tracker.get_connection_innovation(in_node, out_node)

        self.connections[key] = ConnectionGene(
            key=key,
            weight=weight,
            enabled=True,
            innovation=innovation,
            origin_genome_id=self.id,
        )

    def copy(self) -> 'Genome':
        """Create a deep copy of this genome."""
        new_genome = Genome()
        new_genome.nodes = {k: v.copy() for k, v in self.nodes.items()}
        new_genome.connections = {k: v.copy() for k, v in self.connections.items()}
        new_genome.fitness = self.fitness
        new_genome.parent_ids = self.parent_ids
        new_genome.fusion_generation = self.fusion_generation
        new_genome.is_fused = self.is_fused
        return new_genome

    def size(self) -> Tuple[int, int]:
        """Return (num_nodes, num_enabled_connections)."""
        num_nodes = len(self.nodes)
        num_connections = sum(1 for c in self.connections.values() if c.enabled)
        return num_nodes, num_connections

    def complexity(self) -> int:
        """Return total complexity score (nodes + connections)."""
        nodes, conns = self.size()
        return nodes + conns

    def get_input_nodes(self) -> List[int]:
        """Get list of input node keys."""
        return [k for k, n in self.nodes.items() if n.node_type == 'input']

    def get_output_nodes(self) -> List[int]:
        """Get list of output node keys."""
        return [k for k, n in self.nodes.items() if n.node_type == 'output']

    def get_hidden_nodes(self) -> List[int]:
        """Get list of hidden node keys."""
        return [k for k, n in self.nodes.items() if n.node_type == 'hidden']

    def distance(self, other: 'Genome', config) -> float:
        """
        Compute genetic distance to another genome.

        Uses the NEAT compatibility distance formula:
        d = c1*E/N + c2*D/N + c3*W

        Where:
        - E = number of excess genes
        - D = number of disjoint genes
        - W = average weight difference of matching genes
        - N = number of genes in larger genome (normalization)
        - c1, c2, c3 = coefficients from config
        """
        spec_config = config.speciation

        # Get all connection innovations
        innov1 = {c.innovation: c for c in self.connections.values()}
        innov2 = {c.innovation: c for c in other.connections.values()}

        all_innovations = set(innov1.keys()) | set(innov2.keys())

        if not all_innovations:
            return 0.0

        max_innov1 = max(innov1.keys()) if innov1 else 0
        max_innov2 = max(innov2.keys()) if innov2 else 0

        excess = 0
        disjoint = 0
        matching = 0
        weight_diff_sum = 0.0

        for innov in all_innovations:
            in1 = innov in innov1
            in2 = innov in innov2

            if in1 and in2:
                # Matching gene
                matching += 1
                weight_diff_sum += abs(innov1[innov].weight - innov2[innov].weight)
            elif in1 and not in2:
                # Gene only in genome 1
                if innov > max_innov2:
                    excess += 1
                else:
                    disjoint += 1
            else:
                # Gene only in genome 2
                if innov > max_innov1:
                    excess += 1
                else:
                    disjoint += 1

        # Normalize by size of larger genome
        n = max(len(self.connections), len(other.connections), 1)

        # Average weight difference
        avg_weight_diff = weight_diff_sum / matching if matching > 0 else 0.0

        distance = (
            spec_config.excess_coefficient * excess / n +
            spec_config.disjoint_coefficient * disjoint / n +
            spec_config.weight_coefficient * avg_weight_diff
        )

        return distance

    def __str__(self) -> str:
        nodes, conns = self.size()
        return f"Genome(id={self.id}, nodes={nodes}, connections={conns}, fitness={self.fitness})"

    def __repr__(self) -> str:
        return self.__str__()


def create_initial_genome(config: Config, with_hidden: bool = False) -> Genome:
    """
    Create a new genome with initial structure.

    Args:
        config: Configuration
        with_hidden: If True, add a random hidden node for diversity
    """
    genome = Genome()
    genome.configure_new(config.genome)

    if with_hidden:
        # Add a hidden node to create structural diversity
        # This is essential for symbiogenesis to work - we need diverse
        # initial structures that can combine usefully
        tracker = get_innovation_tracker()
        genome_config = config.genome

        # Create a hidden node
        hidden_key = max(genome.nodes.keys()) + 1
        genome.nodes[hidden_key] = NodeGene(
            key=hidden_key,
            node_type='hidden',
            bias=random.gauss(genome_config.bias_init_mean, genome_config.bias_init_std),
            activation=random.choice(genome_config.activation_options) if genome_config.activation_options else genome_config.activation_default,
            response=1.0,
            origin_genome_id=genome.id,
        )

        # Connect random input to hidden
        input_keys = [k for k, n in genome.nodes.items() if n.node_type == 'input']
        output_keys = [k for k, n in genome.nodes.items() if n.node_type == 'output']

        in_key = random.choice(input_keys)
        out_key = random.choice(output_keys)

        # Input -> Hidden
        weight1 = random.gauss(genome_config.weight_init_mean, genome_config.weight_init_std)
        weight1 = max(genome_config.weight_min, min(genome_config.weight_max, weight1))
        genome.connections[(in_key, hidden_key)] = ConnectionGene(
            key=(in_key, hidden_key),
            weight=weight1,
            enabled=True,
            innovation=tracker.get_connection_innovation(in_key, hidden_key),
            origin_genome_id=genome.id,
        )

        # Hidden -> Output
        weight2 = random.gauss(genome_config.weight_init_mean, genome_config.weight_init_std)
        weight2 = max(genome_config.weight_min, min(genome_config.weight_max, weight2))
        genome.connections[(hidden_key, out_key)] = ConnectionGene(
            key=(hidden_key, out_key),
            weight=weight2,
            enabled=True,
            innovation=tracker.get_connection_innovation(hidden_key, out_key),
            origin_genome_id=genome.id,
        )

    return genome


def crossover(parent1: Genome, parent2: Genome, config: Config) -> Genome:
    """
    Create a child genome through crossover of two parents.

    Following NEAT crossover rules:
    - Matching genes (same innovation number) are inherited randomly
    - Disjoint and excess genes are inherited from the fitter parent
    """
    # Determine which parent is fitter
    if parent1.fitness is None:
        parent1.fitness = float('-inf')
    if parent2.fitness is None:
        parent2.fitness = float('-inf')

    if parent1.fitness > parent2.fitness:
        fit_parent, other_parent = parent1, parent2
    elif parent2.fitness > parent1.fitness:
        fit_parent, other_parent = parent2, parent1
    else:
        # Equal fitness - randomly choose
        fit_parent, other_parent = random.choice([(parent1, parent2), (parent2, parent1)])

    child = Genome()
    child.parent_ids = (parent1.id, parent2.id)

    # Inherit nodes from fitter parent (they define structure)
    for key, node in fit_parent.nodes.items():
        if key in other_parent.nodes:
            # Both parents have this node - crossover
            child.nodes[key] = node.crossover(other_parent.nodes[key])
        else:
            # Only fitter parent has this node
            child.nodes[key] = node.copy()

    # Also inherit any unique nodes from other parent that connect to existing structure
    # (This supports fusion-derived diversity)
    for key, node in other_parent.nodes.items():
        if key not in child.nodes:
            child.nodes[key] = node.copy()

    # Inherit connections
    innov1 = {c.innovation: c for c in fit_parent.connections.values()}
    innov2 = {c.innovation: c for c in other_parent.connections.values()}

    for innov, conn in innov1.items():
        if innov in innov2:
            # Matching gene - crossover
            new_conn = conn.crossover(innov2[innov])
        else:
            # Only in fitter parent
            new_conn = conn.copy()

        # Only add if both nodes exist in child
        if new_conn.in_node in child.nodes and new_conn.out_node in child.nodes:
            child.connections[new_conn.key] = new_conn

    # Also add connections from other parent that connect existing nodes
    for innov, conn in innov2.items():
        if innov not in innov1:
            new_conn = conn.copy()
            if (new_conn.in_node in child.nodes and
                new_conn.out_node in child.nodes and
                new_conn.key not in child.connections):
                child.connections[new_conn.key] = new_conn

    return child


def fuse(genome1: Genome, genome2: Genome, config: Config, generation: int = 0) -> Genome:
    """
    Create a new genome by fusing two genomes (symbiogenesis).

    This is the core operation of SYNE. Unlike crossover, fusion creates
    a genome that contains BOTH parent genomes as subsystems, connected
    by new inter-network connections.

    The fused genome is genuinely more complex than either parent -
    it contains all nodes and connections from both, plus new connections
    that wire the two sub-networks together.
    """
    tracker = get_innovation_tracker()
    fusion_config = config.fusion
    genome_config = config.genome

    child = Genome()
    child.parent_ids = (genome1.id, genome2.id)
    child.is_fused = True
    child.fusion_generation = generation

    # Record fusion event for lineage tracking
    tracker.record_fusion(genome1.id, genome2.id)

    # Step 1: Identify overlapping nodes and remap genome2's nodes
    # to avoid key collisions with genome1

    # Find the maximum node key in genome1
    max_key_g1 = max(genome1.nodes.keys()) if genome1.nodes else 0

    # Create node remapping for genome2
    # Input and output nodes keep their keys (they're structural)
    # Hidden nodes get remapped to avoid collisions
    node_remap: Dict[int, int] = {}
    next_hidden_key = max_key_g1 + 1

    for key, node in genome2.nodes.items():
        if node.node_type == 'hidden':
            node_remap[key] = next_hidden_key
            next_hidden_key += 1
        else:
            # Input/output nodes - check for collision
            if key in genome1.nodes:
                # Same structural position - keep same key
                node_remap[key] = key
            else:
                node_remap[key] = key

    # Step 2: Copy all nodes from genome1
    for key, node in genome1.nodes.items():
        child.nodes[key] = node.copy()
        child.nodes[key].origin_genome_id = genome1.id

    # Step 3: Copy hidden nodes from genome2 (with remapping)
    for key, node in genome2.nodes.items():
        if node.node_type == 'hidden':
            new_key = node_remap[key]
            new_node = node.copy()
            new_node.key = new_key
            new_node.origin_genome_id = genome2.id
            child.nodes[new_key] = new_node

    # Step 4: Copy all connections from genome1
    for key, conn in genome1.connections.items():
        child.connections[key] = conn.copy()
        child.connections[key].origin_genome_id = genome1.id

    # Step 5: Copy connections from genome2 (with node remapping for hidden nodes)
    for key, conn in genome2.connections.items():
        in_node = node_remap.get(conn.in_node, conn.in_node)
        out_node = node_remap.get(conn.out_node, conn.out_node)
        new_key = (in_node, out_node)

        # Skip if this connection already exists (from genome1)
        if new_key in child.connections:
            continue

        new_conn = ConnectionGene(
            key=new_key,
            weight=conn.weight,
            enabled=conn.enabled,
            innovation=tracker.get_connection_innovation(in_node, out_node),
            origin_genome_id=genome2.id,
        )
        child.connections[new_key] = new_conn

    # Step 6: Create inter-network connections (the "symbiotic wiring")
    # Get hidden nodes from each original genome
    g1_hidden = [k for k, n in child.nodes.items()
                 if n.node_type == 'hidden' and n.origin_genome_id == genome1.id]
    g2_hidden = [k for k, n in child.nodes.items()
                 if n.node_type == 'hidden' and n.origin_genome_id == genome2.id]

    # If no hidden nodes, use output nodes as connection targets
    if not g1_hidden:
        g1_hidden = [k for k, n in child.nodes.items()
                     if n.node_type == 'output']
    if not g2_hidden:
        g2_hidden = [k for k, n in child.nodes.items()
                     if n.node_type == 'output']

    # Determine number of inter-network connections
    if fusion_config.inter_network_connectivity == 'sparse':
        conn_prob = 0.05
    elif fusion_config.inter_network_connectivity == 'moderate':
        conn_prob = 0.15
    else:  # dense
        conn_prob = 0.30

    # Create bidirectional inter-network connections
    possible_connections = []

    if fusion_config.fusion_topology in ('parallel', 'mixed'):
        # Cross-connect hidden layers
        for n1 in g1_hidden:
            for n2 in g2_hidden:
                possible_connections.append((n1, n2))
                possible_connections.append((n2, n1))

    if fusion_config.fusion_topology in ('sequential', 'mixed'):
        # Connect outputs of one to hidden of other
        g1_outputs = [k for k, n in child.nodes.items()
                      if n.origin_genome_id == genome1.id and n.node_type == 'output']
        g2_outputs = [k for k, n in child.nodes.items()
                      if n.origin_genome_id == genome2.id and n.node_type == 'output']

        for out_node in g1_outputs:
            for hidden in g2_hidden:
                if hidden != out_node:  # Avoid self-connections
                    possible_connections.append((out_node, hidden))

        for out_node in g2_outputs:
            for hidden in g1_hidden:
                if hidden != out_node:
                    possible_connections.append((out_node, hidden))

    # Add connections with probability
    for in_node, out_node in possible_connections:
        if random.random() < conn_prob:
            key = (in_node, out_node)
            if key not in child.connections and in_node != out_node:
                # Ensure we're not creating a cycle to input nodes
                if child.nodes.get(out_node, NodeGene(0, 'input')).node_type != 'input':
                    weight = random.gauss(0.0, fusion_config.inter_weight_init_std)
                    weight = max(genome_config.weight_min, min(genome_config.weight_max, weight))

                    child.connections[key] = ConnectionGene(
                        key=key,
                        weight=weight,
                        enabled=True,
                        innovation=tracker.get_connection_innovation(in_node, out_node),
                        origin_genome_id=None,  # Inter-network connection
                    )

    return child
