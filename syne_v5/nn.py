"""
Neural network phenotype for SYNE v5.
"""

from typing import Dict, List, Tuple, Set, Callable
from collections import defaultdict

from syne_v5.genome import Genome, NodeGene
from syne_v5.config import ACTIVATION_FUNCTIONS, AGGREGATION_FUNCTIONS


class FeedForwardNetwork:
    """A feed-forward neural network built from a SYNE genome."""

    def __init__(
        self,
        inputs: List[int],
        outputs: List[int],
        node_evals: List[Tuple[int, Callable, Callable, float, float, List[Tuple[int, float]]]],
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.node_evals = node_evals
        self.values: Dict[int, float] = {}

    def activate(self, inputs: List[float]) -> List[float]:
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(inputs)}")

        for key, value in zip(self.inputs, inputs):
            self.values[key] = value

        for node_key, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = []
            for input_key, weight in links:
                node_inputs.append(self.values.get(input_key, 0.0) * weight)

            if node_inputs:
                s = aggregation(node_inputs)
            else:
                s = 0.0

            self.values[node_key] = activation(bias + response * s)

        return [self.values[key] for key in self.outputs]

    def reset(self) -> None:
        self.values.clear()

    @staticmethod
    def create(genome: Genome) -> 'FeedForwardNetwork':
        inputs = sorted([k for k, n in genome.nodes.items() if n.node_type == 'input'])
        outputs = sorted([k for k, n in genome.nodes.items() if n.node_type == 'output'])

        connections: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for conn in genome.connections.values():
            if conn.enabled:
                connections[conn.out_node].append((conn.in_node, conn.weight))

        required_nodes: Set[int] = set()

        def add_required(node: int) -> None:
            if node in required_nodes or node in inputs:
                return
            required_nodes.add(node)
            for in_node, _ in connections.get(node, []):
                add_required(in_node)

        for output in outputs:
            add_required(output)

        required_nodes.update(outputs)

        in_degree: Dict[int, int] = {n: 0 for n in required_nodes}
        for node in required_nodes:
            for in_node, _ in connections.get(node, []):
                if in_node in required_nodes or in_node in inputs:
                    in_degree[node] += 1

        eval_order: List[int] = []
        processed = set(inputs)

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

            for next_node in required_nodes:
                if next_node in processed:
                    continue
                deps = [in_node for in_node, _ in connections.get(next_node, [])]
                if all(d in processed for d in deps):
                    if next_node not in ready:
                        ready.append(next_node)

        for node in required_nodes:
            if node not in processed:
                eval_order.append(node)

        node_evals = []
        for node_key in eval_order:
            node = genome.nodes[node_key]
            activation = ACTIVATION_FUNCTIONS.get(node.activation, ACTIVATION_FUNCTIONS['sigmoid'])
            aggregation = AGGREGATION_FUNCTIONS.get('sum')
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
    """A recurrent neural network built from a SYNE genome."""

    def __init__(
        self,
        inputs: List[int],
        outputs: List[int],
        nodes: Dict[int, Tuple[Callable, float, float]],
        connections: List[Tuple[int, int, float]],
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
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(inputs)}")

        for key, value in zip(self.inputs, inputs):
            self.values[key] = value
            self.prev_values[key] = value

        for _ in range(self.activation_steps):
            for key, (activation, bias, response) in self.nodes.items():
                if key in self.inputs:
                    continue

                s = 0.0
                for in_node, out_node, weight in self.connections:
                    if out_node == key:
                        s += self.prev_values.get(in_node, 0.0) * weight

                self.values[key] = activation(bias + response * s)

            self.prev_values = dict(self.values)

        return [self.values[key] for key in self.outputs]

    def reset(self) -> None:
        self.values = {k: 0.0 for k in self.nodes}
        self.prev_values = {k: 0.0 for k in self.nodes}

    @staticmethod
    def create(genome: Genome, activation_steps: int = 5) -> 'RecurrentNetwork':
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
    """Evaluate a genome on a set of inputs."""
    network = FeedForwardNetwork.create(genome)
    outputs = []
    for input_vector in inputs:
        output = network.activate(input_vector)
        outputs.append(output)
    return outputs
