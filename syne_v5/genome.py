"""
Genome representation for SYNE v5.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random

from syne_v5.config import Config, GenomeConfig, ACTIVATION_FUNCTIONS
from syne_v5.innovation import get_innovation_tracker, InnovationTracker


@dataclass
class NodeGene:
    """Represents a single node (neuron) in the network."""
    key: int
    node_type: str  # 'input', 'output', 'hidden'
    bias: float = 0.0
    activation: str = 'sigmoid'
    response: float = 1.0
    origin_genome_id: Optional[int] = None

    def copy(self) -> 'NodeGene':
        return NodeGene(
            key=self.key,
            node_type=self.node_type,
            bias=self.bias,
            activation=self.activation,
            response=self.response,
            origin_genome_id=self.origin_genome_id,
        )

    def crossover(self, other: 'NodeGene') -> 'NodeGene':
        assert self.key == other.key
        assert self.node_type == other.node_type
        return NodeGene(
            key=self.key,
            node_type=self.node_type,
            bias=random.choice([self.bias, other.bias]),
            activation=random.choice([self.activation, other.activation]),
            response=random.choice([self.response, other.response]),
            origin_genome_id=None,
        )


@dataclass
class ConnectionGene:
    """Represents a connection between two nodes."""
    key: Tuple[int, int]  # (in_node, out_node)
    weight: float = 0.0
    enabled: bool = True
    innovation: int = 0
    origin_genome_id: Optional[int] = None

    @property
    def in_node(self) -> int:
        return self.key[0]

    @property
    def out_node(self) -> int:
        return self.key[1]

    def copy(self) -> 'ConnectionGene':
        return ConnectionGene(
            key=self.key,
            weight=self.weight,
            enabled=self.enabled,
            innovation=self.innovation,
            origin_genome_id=self.origin_genome_id,
        )

    def crossover(self, other: 'ConnectionGene', blend: bool = True) -> 'ConnectionGene':
        assert self.innovation == other.innovation
        if blend and random.random() < 0.5:
            alpha = random.uniform(-0.1, 1.1)
            weight = alpha * self.weight + (1 - alpha) * other.weight
        else:
            weight = random.choice([self.weight, other.weight])

        return ConnectionGene(
            key=self.key,
            weight=weight,
            enabled=self.enabled if self.enabled == other.enabled else random.choice([True, False]),
            innovation=self.innovation,
            origin_genome_id=None,
        )


class Genome:
    """A genome representing a neural network."""
    _next_id: int = 0

    def __init__(self, genome_id: Optional[int] = None):
        if genome_id is None:
            genome_id = Genome._next_id
            Genome._next_id += 1

        self.id: int = genome_id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[Tuple[int, int], ConnectionGene] = {}
        self.fitness: Optional[float] = None
        self.parent_ids: Optional[Tuple[int, int]] = None
        self.fusion_generation: Optional[int] = None
        self.is_fused: bool = False

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_id = 0

    def configure_new(self, config: GenomeConfig) -> None:
        tracker = get_innovation_tracker()

        # Create input nodes (negative keys)
        for i in range(config.num_inputs):
            node_key = -(i + 1)
            self.nodes[node_key] = NodeGene(
                key=node_key,
                node_type='input',
                bias=0.0,
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

        # Create initial connections
        input_keys = [k for k in self.nodes if self.nodes[k].node_type == 'input']
        output_keys = [k for k in self.nodes if self.nodes[k].node_type == 'output']

        if config.initial_connectivity == 'full':
            for in_key in input_keys:
                for out_key in output_keys:
                    self._add_connection(in_key, out_key, config, tracker)
        elif config.initial_connectivity == 'partial':
            for in_key in input_keys:
                for out_key in output_keys:
                    if random.random() < config.partial_connectivity_prob:
                        self._add_connection(in_key, out_key, config, tracker)

    def _add_connection(self, in_node: int, out_node: int, config: GenomeConfig, tracker: InnovationTracker) -> None:
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
        new_genome = Genome()
        new_genome.nodes = {k: v.copy() for k, v in self.nodes.items()}
        new_genome.connections = {k: v.copy() for k, v in self.connections.items()}
        new_genome.fitness = self.fitness
        new_genome.parent_ids = self.parent_ids
        new_genome.fusion_generation = self.fusion_generation
        new_genome.is_fused = self.is_fused
        return new_genome

    def size(self) -> Tuple[int, int]:
        num_nodes = len(self.nodes)
        num_connections = sum(1 for c in self.connections.values() if c.enabled)
        return num_nodes, num_connections

    def complexity(self) -> int:
        nodes, conns = self.size()
        return nodes + conns

    def get_input_nodes(self) -> List[int]:
        return [k for k, n in self.nodes.items() if n.node_type == 'input']

    def get_output_nodes(self) -> List[int]:
        return [k for k, n in self.nodes.items() if n.node_type == 'output']

    def get_hidden_nodes(self) -> List[int]:
        return [k for k, n in self.nodes.items() if n.node_type == 'hidden']

    def distance(self, other: 'Genome', config) -> float:
        """Compute NEAT-style genetic distance."""
        spec_config = config.speciation
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
                matching += 1
                weight_diff_sum += abs(innov1[innov].weight - innov2[innov].weight)
            elif in1:
                if innov > max_innov2:
                    excess += 1
                else:
                    disjoint += 1
            else:
                if innov > max_innov1:
                    excess += 1
                else:
                    disjoint += 1

        n = max(len(self.connections), len(other.connections), 1)
        avg_weight_diff = weight_diff_sum / matching if matching > 0 else 0.0

        return (
            spec_config.excess_coefficient * excess / n +
            spec_config.disjoint_coefficient * disjoint / n +
            spec_config.weight_coefficient * avg_weight_diff
        )

    def __str__(self) -> str:
        nodes, conns = self.size()
        return f"Genome(id={self.id}, nodes={nodes}, connections={conns}, fitness={self.fitness})"

    def __repr__(self) -> str:
        return self.__str__()


def create_initial_genome(config: Config, with_hidden: bool = False) -> Genome:
    """Create a new genome with initial structure."""
    genome = Genome()
    genome.configure_new(config.genome)

    if with_hidden:
        tracker = get_innovation_tracker()
        genome_config = config.genome

        hidden_key = max(genome.nodes.keys()) + 1
        genome.nodes[hidden_key] = NodeGene(
            key=hidden_key,
            node_type='hidden',
            bias=random.gauss(genome_config.bias_init_mean, genome_config.bias_init_std),
            activation=random.choice(genome_config.activation_options) if genome_config.activation_options else genome_config.activation_default,
            response=1.0,
            origin_genome_id=genome.id,
        )

        input_keys = [k for k, n in genome.nodes.items() if n.node_type == 'input']
        output_keys = [k for k, n in genome.nodes.items() if n.node_type == 'output']

        in_key = random.choice(input_keys)
        out_key = random.choice(output_keys)

        weight1 = random.gauss(genome_config.weight_init_mean, genome_config.weight_init_std)
        weight1 = max(genome_config.weight_min, min(genome_config.weight_max, weight1))
        genome.connections[(in_key, hidden_key)] = ConnectionGene(
            key=(in_key, hidden_key),
            weight=weight1,
            enabled=True,
            innovation=tracker.get_connection_innovation(in_key, hidden_key),
            origin_genome_id=genome.id,
        )

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
    """Create child genome through crossover of two parents."""
    if parent1.fitness is None:
        parent1.fitness = float('-inf')
    if parent2.fitness is None:
        parent2.fitness = float('-inf')

    if parent1.fitness > parent2.fitness:
        fit_parent, other_parent = parent1, parent2
    elif parent2.fitness > parent1.fitness:
        fit_parent, other_parent = parent2, parent1
    else:
        fit_parent, other_parent = random.choice([(parent1, parent2), (parent2, parent1)])

    child = Genome()
    child.parent_ids = (parent1.id, parent2.id)

    for key, node in fit_parent.nodes.items():
        if key in other_parent.nodes:
            child.nodes[key] = node.crossover(other_parent.nodes[key])
        else:
            child.nodes[key] = node.copy()

    for key, node in other_parent.nodes.items():
        if key not in child.nodes:
            child.nodes[key] = node.copy()

    innov1 = {c.innovation: c for c in fit_parent.connections.values()}
    innov2 = {c.innovation: c for c in other_parent.connections.values()}

    for innov, conn in innov1.items():
        if innov in innov2:
            new_conn = conn.crossover(innov2[innov])
        else:
            new_conn = conn.copy()
        if new_conn.in_node in child.nodes and new_conn.out_node in child.nodes:
            child.connections[new_conn.key] = new_conn

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

    This is the core SYNE operation. Unlike crossover, fusion creates
    a genome containing BOTH parent genomes as subsystems, connected
    by new inter-network connections.
    """
    tracker = get_innovation_tracker()
    fusion_config = config.fusion
    genome_config = config.genome

    child = Genome()
    child.parent_ids = (genome1.id, genome2.id)
    child.is_fused = True
    child.fusion_generation = generation

    tracker.record_fusion(genome1.id, genome2.id)

    # Find max node key in genome1
    max_key_g1 = max(genome1.nodes.keys()) if genome1.nodes else 0

    # Create node remapping for genome2
    node_remap: Dict[int, int] = {}
    next_hidden_key = max_key_g1 + 1

    for key, node in genome2.nodes.items():
        if node.node_type == 'hidden':
            node_remap[key] = next_hidden_key
            next_hidden_key += 1
        else:
            node_remap[key] = key

    # Copy all nodes from genome1
    for key, node in genome1.nodes.items():
        child.nodes[key] = node.copy()
        child.nodes[key].origin_genome_id = genome1.id

    # Copy hidden nodes from genome2 (with remapping)
    for key, node in genome2.nodes.items():
        if node.node_type == 'hidden':
            new_key = node_remap[key]
            new_node = node.copy()
            new_node.key = new_key
            new_node.origin_genome_id = genome2.id
            child.nodes[new_key] = new_node

    # Copy connections from genome1
    for key, conn in genome1.connections.items():
        child.connections[key] = conn.copy()
        child.connections[key].origin_genome_id = genome1.id

    # Copy connections from genome2 (with remapping)
    for key, conn in genome2.connections.items():
        in_node = node_remap.get(conn.in_node, conn.in_node)
        out_node = node_remap.get(conn.out_node, conn.out_node)
        new_key = (in_node, out_node)

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

    # Create inter-network connections
    g1_hidden = [k for k, n in child.nodes.items()
                 if n.node_type == 'hidden' and n.origin_genome_id == genome1.id]
    g2_hidden = [k for k, n in child.nodes.items()
                 if n.node_type == 'hidden' and n.origin_genome_id == genome2.id]

    if not g1_hidden:
        g1_hidden = [k for k, n in child.nodes.items() if n.node_type == 'output']
    if not g2_hidden:
        g2_hidden = [k for k, n in child.nodes.items() if n.node_type == 'output']

    if fusion_config.inter_network_connectivity == 'sparse':
        conn_prob = 0.05
    elif fusion_config.inter_network_connectivity == 'moderate':
        conn_prob = 0.15
    else:
        conn_prob = 0.30

    possible_connections = []

    if fusion_config.fusion_topology in ('parallel', 'mixed'):
        for n1 in g1_hidden:
            for n2 in g2_hidden:
                possible_connections.append((n1, n2))
                possible_connections.append((n2, n1))

    if fusion_config.fusion_topology in ('sequential', 'mixed'):
        g1_outputs = [k for k, n in child.nodes.items()
                      if n.origin_genome_id == genome1.id and n.node_type == 'output']
        g2_outputs = [k for k, n in child.nodes.items()
                      if n.origin_genome_id == genome2.id and n.node_type == 'output']

        for out_node in g1_outputs:
            for hidden in g2_hidden:
                if hidden != out_node:
                    possible_connections.append((out_node, hidden))

        for out_node in g2_outputs:
            for hidden in g1_hidden:
                if hidden != out_node:
                    possible_connections.append((out_node, hidden))

    for in_node, out_node in possible_connections:
        if random.random() < conn_prob:
            key = (in_node, out_node)
            if key not in child.connections and in_node != out_node:
                if child.nodes.get(out_node, NodeGene(0, 'input')).node_type != 'input':
                    weight = random.gauss(0.0, fusion_config.inter_weight_init_std)
                    weight = max(genome_config.weight_min, min(genome_config.weight_max, weight))

                    child.connections[key] = ConnectionGene(
                        key=key,
                        weight=weight,
                        enabled=True,
                        innovation=tracker.get_connection_innovation(in_node, out_node),
                        origin_genome_id=None,
                    )

    return child
