"""
Visualization utilities for SYNE.

Provides tools to visualize genomes, networks, and evolutionary progress.
"""

from typing import Dict, List, Optional, Tuple
import math

from syne_v2.genome import Genome, NodeGene, ConnectionGene
from syne_v2.population import GenerationStats


def genome_to_dot(genome: Genome, show_weights: bool = True) -> str:
    """
    Convert a genome to GraphViz DOT format.

    Args:
        genome: The genome to visualize
        show_weights: Whether to show connection weights

    Returns:
        DOT format string
    """
    lines = ['digraph genome {']
    lines.append('    rankdir=LR;')
    lines.append('    node [shape=circle];')
    lines.append('')

    # Group nodes by type
    input_nodes = [k for k, n in genome.nodes.items() if n.node_type == 'input']
    output_nodes = [k for k, n in genome.nodes.items() if n.node_type == 'output']
    hidden_nodes = [k for k, n in genome.nodes.items() if n.node_type == 'hidden']

    # Style input nodes
    lines.append('    subgraph cluster_inputs {')
    lines.append('        label="Inputs";')
    lines.append('        style=dashed;')
    for node_key in sorted(input_nodes):
        node = genome.nodes[node_key]
        color = _get_origin_color(node.origin_genome_id)
        lines.append(f'        {_node_name(node_key)} [label="I{abs(node_key)}", '
                    f'style=filled, fillcolor="{color}"];')
    lines.append('    }')
    lines.append('')

    # Style output nodes
    lines.append('    subgraph cluster_outputs {')
    lines.append('        label="Outputs";')
    lines.append('        style=dashed;')
    for node_key in sorted(output_nodes):
        node = genome.nodes[node_key]
        color = _get_origin_color(node.origin_genome_id)
        lines.append(f'        {_node_name(node_key)} [label="O{node_key}", '
                    f'style=filled, fillcolor="{color}", shape=doublecircle];')
    lines.append('    }')
    lines.append('')

    # Style hidden nodes
    if hidden_nodes:
        lines.append('    subgraph cluster_hidden {')
        lines.append('        label="Hidden";')
        lines.append('        style=dashed;')
        for node_key in sorted(hidden_nodes):
            node = genome.nodes[node_key]
            color = _get_origin_color(node.origin_genome_id)
            lines.append(f'        {_node_name(node_key)} [label="H{node_key}", '
                        f'style=filled, fillcolor="{color}"];')
        lines.append('    }')
        lines.append('')

    # Add connections
    for conn in genome.connections.values():
        if not conn.enabled:
            continue

        style = 'solid'
        color = 'black'

        # Color inter-network connections differently
        if conn.origin_genome_id is None:
            color = 'purple'
            style = 'bold'

        weight_label = f'{conn.weight:.2f}' if show_weights else ''
        width = max(0.5, min(3.0, abs(conn.weight)))

        lines.append(f'    {_node_name(conn.in_node)} -> {_node_name(conn.out_node)} '
                    f'[label="{weight_label}", penwidth={width}, '
                    f'color="{color}", style="{style}"];')

    lines.append('}')
    return '\n'.join(lines)


def _node_name(key: int) -> str:
    """Convert node key to valid DOT identifier."""
    if key < 0:
        return f'n_neg{abs(key)}'
    return f'n{key}'


def _get_origin_color(origin_id: Optional[int]) -> str:
    """Get a color based on origin genome ID."""
    if origin_id is None:
        return 'white'

    # Use a simple color cycling scheme
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink',
              'lightcoral', 'lightcyan', 'lavender', 'wheat']
    return colors[origin_id % len(colors)]


def print_genome_structure(genome: Genome) -> None:
    """Print a text representation of genome structure."""
    print(f"\n{'='*60}")
    print(f"Genome {genome.id}")
    print(f"{'='*60}")

    print(f"\nNodes ({len(genome.nodes)}):")
    for key in sorted(genome.nodes.keys()):
        node = genome.nodes[key]
        origin = f" (from genome {node.origin_genome_id})" if node.origin_genome_id else ""
        print(f"  [{node.node_type:6}] {key:4d}: bias={node.bias:.3f}, "
              f"act={node.activation}{origin}")

    print(f"\nConnections ({len(genome.connections)}):")
    for key in sorted(genome.connections.keys()):
        conn = genome.connections[key]
        enabled = "✓" if conn.enabled else "✗"
        origin = f" (from genome {conn.origin_genome_id})" if conn.origin_genome_id else " (inter-network)"
        print(f"  {enabled} {conn.in_node:4d} -> {conn.out_node:4d}: "
              f"w={conn.weight:7.3f}, innov={conn.innovation}{origin}")

    if genome.is_fused and genome.parent_ids:
        print(f"\nFusion info:")
        print(f"  Parents: {genome.parent_ids[0]} + {genome.parent_ids[1]}")
        print(f"  Fused at generation: {genome.fusion_generation}")

    print(f"\nSize: {genome.size()[0]} nodes, {genome.size()[1]} connections")
    print(f"Fitness: {genome.fitness}")
    print('='*60)


def plot_history_text(history: List[GenerationStats], width: int = 60) -> str:
    """
    Create a text-based plot of evolution history.

    Args:
        history: List of generation statistics
        width: Width of the plot in characters

    Returns:
        ASCII art plot
    """
    if not history:
        return "No history to display"

    lines = []
    lines.append("\n" + "="*width)
    lines.append("Evolution History")
    lines.append("="*width)

    # Get fitness range
    fitnesses = [s.best_fitness for s in history if s.best_fitness is not None]
    if not fitnesses:
        return "No fitness data to display"

    min_fit = min(fitnesses)
    max_fit = max(fitnesses)
    fit_range = max_fit - min_fit if max_fit > min_fit else 1.0

    # Plot fitness
    plot_width = width - 20
    lines.append(f"\nBest Fitness (max: {max_fit:.4f}, min: {min_fit:.4f}):")
    lines.append("-" * width)

    for stats in history:
        if stats.best_fitness is not None:
            # Normalize to plot width
            pos = int((stats.best_fitness - min_fit) / fit_range * plot_width)
            pos = max(0, min(plot_width - 1, pos))
            bar = " " * pos + "█"
            lines.append(f"G{stats.generation:4d} |{bar}")

    # Summary statistics
    lines.append("\n" + "-"*width)
    lines.append("Summary:")
    lines.append(f"  Generations: {len(history)}")
    lines.append(f"  Best fitness: {max_fit:.6f}")
    lines.append(f"  Final species: {history[-1].num_species}")
    lines.append(f"  Final genome size: {history[-1].best_genome_size}")

    # Fusion statistics
    total_fusions = sum(s.fusion_count for s in history)
    total_crossovers = sum(s.crossover_count for s in history)
    total = total_fusions + total_crossovers
    if total > 0:
        fusion_rate = total_fusions / total * 100
        lines.append(f"  Total fusions: {total_fusions} ({fusion_rate:.1f}%)")
        lines.append(f"  Total crossovers: {total_crossovers}")

    lines.append("="*width)

    return "\n".join(lines)


def analyze_complexity_growth(history: List[GenerationStats]) -> Dict:
    """
    Analyze how complexity grows over evolution.

    Returns statistics about complexity growth patterns.
    """
    if not history:
        return {}

    sizes = [(s.generation, s.best_genome_size[0], s.best_genome_size[1])
             for s in history]

    nodes = [s[1] for s in sizes]
    conns = [s[2] for s in sizes]

    # Calculate growth rates
    node_growth = (nodes[-1] - nodes[0]) / len(history) if len(history) > 1 else 0
    conn_growth = (conns[-1] - conns[0]) / len(history) if len(history) > 1 else 0

    # Find fusion-driven growth episodes
    fusion_episodes = []
    for i, stats in enumerate(history):
        if stats.fusion_count > 0 and i > 0:
            prev_size = history[i-1].best_genome_size
            curr_size = stats.best_genome_size
            if curr_size[0] > prev_size[0] + 2:  # Significant jump
                fusion_episodes.append({
                    'generation': stats.generation,
                    'node_increase': curr_size[0] - prev_size[0],
                    'conn_increase': curr_size[1] - prev_size[1],
                })

    return {
        'initial_nodes': nodes[0],
        'final_nodes': nodes[-1],
        'initial_connections': conns[0],
        'final_connections': conns[-1],
        'node_growth_rate': node_growth,
        'conn_growth_rate': conn_growth,
        'complexity_multiplier': (nodes[-1] + conns[-1]) / (nodes[0] + conns[0]) if (nodes[0] + conns[0]) > 0 else 1,
        'fusion_growth_episodes': fusion_episodes,
    }
