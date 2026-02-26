"""
Neural network phenotype for SYNE.

Builds and evaluates feed-forward neural networks from genome genotypes.
"""

from typing import Dict, List, Tuple, Set, Optional, Callable
from collections import defaultdict

from syne.genome import Genome, NodeGene, ConnectionGene
from syne.config import ACTIVATION_FUNCTIONS, AGGREGATION_FUNCTIONS


class FeedForwardNetwork:
    """
    A feed-forward neural network built from a SYNE genome.

    This is the phenotype - the actual computational entity that
    processes inputs and produces outputs. It's built from the
    genome (genotype) which encodes the network structure.
    """

    def __init__(
        self,
        inputs: List[int],
        outputs: List[int],
        node_evals: List[Tuple[int, Callable, Callable, float, float, List[Tuple[int, float]]]],
    ):
        """
        Initialize the network.

        Args:
            inputs: List of input node keys
            outputs: List of output node keys
            node_evals: List of (node_key, activation_fn, aggregation_fn, bias, response, [(input_key, weight), ...])
        """
        self.inputs = inputs
        self.outputs = outputs
        self.node_evals = node_evals
        self.values: Dict[int, float] = {}

    def activate(self, inputs: List[float]) -> List[float]:
        """
        Activate the network with the given inputs.

        Args:
            inputs: List of input values (must match number of input nodes)

        Returns:
            List of output values
        """
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(inputs)}")

        # Set input values
        for key, value in zip(self.inputs, inputs):
            self.values[key] = value

        # Evaluate nodes in topological order
        for node_key, activation, aggregation, bias, response, links in self.node_evals:
            # Gather inputs to this node
            node_inputs = []
            for input_key, weight in links:
                node_inputs.append(self.values.get(input_key, 0.0) * weight)

            # Aggregate, add bias, apply response and activation
            if node_inputs:
                s = aggregation(node_inputs)
            else:
                s = 0.0

            self.values[node_key] = activation(bias + response * s)

        # Return output values
        return [self.values[key] for key in self.outputs]

    def reset(self) -> None:
        """Reset network state (clear cached values)."""
        self.values.clear()

    @staticmethod
    def create(genome: Genome) -> 'FeedForwardNetwork':
        """
        Create a neural network from a genome.

        Builds the phenotype (network) from the genotype (genome).
        """
        # Get input and output node keys
        inputs = sorted([k for k, n in genome.nodes.items() if n.node_type == 'input'])
        outputs = sorted([k for k, n in genome.nodes.items() if n.node_type == 'output'])

        # Build connection lookup: node -> list of (input_node, weight)
        connections: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for conn in genome.connections.values():
            if conn.enabled:
                connections[conn.out_node].append((conn.in_node, conn.weight))

        # Find all nodes required to compute outputs (recursive dependency trace)
        required_nodes: Set[int] = set()

        def add_required(node: int) -> None:
            if node in required_nodes or node in inputs:
                return
            required_nodes.add(node)
            for in_node, _ in connections.get(node, []):
                add_required(in_node)

        for output in outputs:
            add_required(output)

        # Always include outputs even if they have no incoming connections
        required_nodes.update(outputs)

        # Topological sort using Kahn's algorithm
        # Build in-degree count for required nodes
        in_degree: Dict[int, int] = {n: 0 for n in required_nodes}
        for node in required_nodes:
            for in_node, _ in connections.get(node, []):
                if in_node in required_nodes or in_node in inputs:
                    in_degree[node] += 1

        # Start with nodes that have no dependencies (or only depend on inputs)
        eval_order: List[int] = []
        processed = set(inputs)

        # Initially ready: nodes with in_degree 0 or only input dependencies
        ready = []
        for node in required_nodes:
            deps = [in_node for in_node, _ in connections.get(node, [])]
            if not deps or all(d in inputs for d in deps):
                ready.append(node)

        while ready:
            node = ready.pop(0)
            if node in processed:
                continue

            eval_order.append(node)
            processed.add(node)

            # Check which nodes are now ready
            for next_node in required_nodes:
                if next_node in processed:
                    continue
                deps = [in_node for in_node, _ in connections.get(next_node, [])]
                if all(d in processed for d in deps):
                    if next_node not in ready:
                        ready.append(next_node)

        # Handle any remaining nodes (shouldn't happen in valid feed-forward)
        for node in required_nodes:
            if node not in processed:
                eval_order.append(node)

        # Build node evaluation list
        node_evals = []
        for node_key in eval_order:
            node = genome.nodes[node_key]

            # Get activation and aggregation functions
            activation = ACTIVATION_FUNCTIONS.get(node.activation, ACTIVATION_FUNCTIONS['sigmoid'])
            aggregation = AGGREGATION_FUNCTIONS.get('sum')

            # Get incoming connections
            links = connections.get(node_key, [])

            node_evals.append((
                node_key,
                activation,
                aggregation,
                node.bias,
                node.response,
                links,
            ))

        return FeedForwardNetwork(inputs, outputs, node_evals)


class RecurrentNetwork:
    """
    A recurrent neural network built from a SYNE genome.

    Unlike FeedForwardNetwork, this can handle cycles in the genome.
    Uses iterative activation with a fixed number of steps.
    """

    def __init__(
        self,
        inputs: List[int],
        outputs: List[int],
        nodes: Dict[int, Tuple[Callable, float, float]],  # key -> (activation, bias, response)
        connections: List[Tuple[int, int, float]],  # (in, out, weight)
        activation_steps: int = 5,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self.connections = connections
        self.activation_steps = activation_steps
        self.values: Dict[int, float] = {k: 0.0 for k in nodes}
        self.prev_values: Dict[int, float] = {k: 0.0 for k in nodes}

    def activate(self, inputs: List[float]) -> List[float]:
        """Activate the network with multiple iterations."""
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(inputs)}")

        # Set input values
        for key, value in zip(self.inputs, inputs):
            self.values[key] = value
            self.prev_values[key] = value

        # Iterate activation
        for _ in range(self.activation_steps):
            # Compute new values
            for key, (activation, bias, response) in self.nodes.items():
                if key in self.inputs:
                    continue

                # Sum weighted inputs
                s = 0.0
                for in_node, out_node, weight in self.connections:
                    if out_node == key:
                        s += self.prev_values.get(in_node, 0.0) * weight

                self.values[key] = activation(bias + response * s)

            # Copy current to previous
            self.prev_values = dict(self.values)

        return [self.values[key] for key in self.outputs]

    def reset(self) -> None:
        """Reset network state."""
        self.values = {k: 0.0 for k in self.nodes}
        self.prev_values = {k: 0.0 for k in self.nodes}

    @staticmethod
    def create(genome: Genome, activation_steps: int = 5) -> 'RecurrentNetwork':
        """Create a recurrent network from a genome."""
        inputs = sorted([k for k, n in genome.nodes.items() if n.node_type == 'input'])
        outputs = sorted([k for k, n in genome.nodes.items() if n.node_type == 'output'])

        nodes = {}
        for key, node in genome.nodes.items():
            activation = ACTIVATION_FUNCTIONS.get(node.activation, ACTIVATION_FUNCTIONS['sigmoid'])
            nodes[key] = (activation, node.bias, node.response)

        connections = []
        for conn in genome.connections.values():
            if conn.enabled:
                connections.append((conn.in_node, conn.out_node, conn.weight))

        return RecurrentNetwork(inputs, outputs, nodes, connections, activation_steps)


def evaluate_genome(genome: Genome, inputs: List[List[float]]) -> List[List[float]]:
    """
    Convenience function to evaluate a genome on a set of inputs.

    Args:
        genome: The genome to evaluate
        inputs: List of input vectors

    Returns:
        List of output vectors
    """
    network = FeedForwardNetwork.create(genome)
    outputs = []
    for input_vector in inputs:
        output = network.activate(input_vector)
        outputs.append(output)
    return outputs
