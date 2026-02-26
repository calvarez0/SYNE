"""
XOR Example for SYNE.

Demonstrates the symbiogenetic neuroevolution algorithm on the classic
XOR problem - a simple non-linearly separable classification task that
requires at least one hidden node to solve.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syne import Config, Population
from syne.nn import FeedForwardNetwork
from syne.genome import Genome
from syne.visualization import print_genome_structure, plot_history_text, analyze_complexity_growth


# XOR truth table
XOR_INPUTS = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

XOR_OUTPUTS = [
    [0.0],
    [1.0],
    [1.0],
    [0.0],
]


def xor_fitness(genome: Genome) -> float:
    """
    Evaluate genome fitness on XOR task.

    Fitness is the negative mean squared error, so higher is better.
    Perfect fitness is 0.0.
    """
    try:
        network = FeedForwardNetwork.create(genome)
    except Exception as e:
        # Invalid network (e.g., cycles or disconnected)
        return -4.0

    error = 0.0
    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = network.activate(inputs)
        error += (output[0] - expected[0]) ** 2

    # Return negative MSE (higher = better)
    return 4.0 - error  # Max fitness is 4.0 when error is 0


def run_xor_experiment():
    """Run the XOR experiment."""
    print("="*60)
    print("SYNE: Symbiogenetic Neuro-Evolution")
    print("XOR Benchmark")
    print("="*60)
    print()

    # Create configuration
    config = Config()
    config.genome.num_inputs = 2
    config.genome.num_outputs = 1
    config.genome.activation_default = 'sigmoid'
    config.genome.initial_connectivity = 'full'

    # Symbiogenesis settings
    config.fusion.fusion_prob = 0.5  # 50% chance of fusion vs crossover
    config.fusion.inter_network_connectivity = 'moderate'
    config.fusion.max_genome_nodes = 50
    config.fusion.fusion_fitness_threshold = 0.3  # Top 30% can fuse

    # Population settings
    config.population.population_size = 150
    config.population.fitness_threshold = 3.9  # Near-perfect XOR

    # Speciation - lower threshold to maintain more species for fusion
    config.speciation.compatibility_threshold = 1.0
    config.speciation.stagnation_limit = 20
    config.speciation.species_elitism = 3  # Protect top 3 species

    # Create and run population
    pop = Population(config)
    pop.initialize()

    print(f"Initial population: {len(pop.population)} genomes")
    print(f"Running evolution...")
    print()

    # Run evolution
    best, stats = pop.run(
        fitness_function=xor_fitness,
        max_generations=300,
        fitness_threshold=3.9,
    )

    # Print results
    print()
    print(plot_history_text(pop.history))
    print()

    # Analyze complexity growth
    complexity = analyze_complexity_growth(pop.history)
    print("\nComplexity Growth Analysis:")
    print(f"  Initial: {complexity['initial_nodes']} nodes, {complexity['initial_connections']} connections")
    print(f"  Final: {complexity['final_nodes']} nodes, {complexity['final_connections']} connections")
    print(f"  Complexity multiplier: {complexity['complexity_multiplier']:.2f}x")

    if complexity['fusion_growth_episodes']:
        print(f"\n  Fusion-driven growth episodes:")
        for ep in complexity['fusion_growth_episodes']:
            print(f"    Gen {ep['generation']}: +{ep['node_increase']} nodes, +{ep['conn_increase']} connections")

    # Show best network structure
    print_genome_structure(best)

    # Test the best network
    print("\nBest Network Evaluation:")
    print("-"*40)
    network = FeedForwardNetwork.create(best)
    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = network.activate(inputs)
        correct = "✓" if abs(output[0] - expected[0]) < 0.5 else "✗"
        print(f"  {inputs} -> {output[0]:.4f} (expected {expected[0]}) {correct}")

    return best, pop.history


if __name__ == "__main__":
    run_xor_experiment()
