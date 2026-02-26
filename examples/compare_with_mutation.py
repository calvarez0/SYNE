"""
Comparison: SYNE vs Mutation-based Evolution

This example compares SYNE (pure symbiogenesis) against a mutation-based
baseline to demonstrate the effectiveness of genome fusion for complexity
growth.

This is key for validating Agüera y Arcas's thesis that symbiogenesis,
not mutation, is the primary engine of evolutionary novelty.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

from syne import Config, Population
from syne.nn import FeedForwardNetwork
from syne.genome import Genome, NodeGene, ConnectionGene, crossover
from syne.species import SpeciesSet
from syne.innovation import get_innovation_tracker
from syne.visualization import analyze_complexity_growth


# XOR task for comparison
XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_OUTPUTS = [[0.0], [1.0], [1.0], [0.0]]


def xor_fitness(genome: Genome) -> float:
    """Evaluate XOR fitness."""
    try:
        network = FeedForwardNetwork.create(genome)
    except Exception:
        return -4.0

    error = 0.0
    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = network.activate(inputs)
        error += (output[0] - expected[0]) ** 2

    return 4.0 - error


class MutationReproduction:
    """
    Mutation-based reproduction for comparison.

    This mimics NEAT-style reproduction with structural mutations
    (add node, add connection) and weight mutations.
    """

    def __init__(self, config: Config):
        self.config = config
        # Mutation probabilities
        self.add_node_prob = 0.03
        self.add_conn_prob = 0.30
        self.weight_mutate_prob = 0.80
        self.weight_perturb_prob = 0.90

    def create_initial_population(self, pop_size: int) -> Dict[int, Genome]:
        """Create initial population."""
        from syne.genome import create_initial_genome
        population = {}
        for _ in range(pop_size):
            genome = create_initial_genome(self.config)
            population[genome.id] = genome
        return population

    def reproduce(
        self,
        species_set: SpeciesSet,
        pop_size: int,
        generation: int
    ) -> Dict[int, Genome]:
        """Reproduce with mutation."""
        new_population: Dict[int, Genome] = {}

        # Get fitness shares
        fitness_shares = species_set.get_species_fitness_shares()

        # Preserve elites
        for species in species_set.species.values():
            elites = species.get_elites(2)
            for elite in elites:
                new_genome = elite.copy()
                new_genome.id = Genome._next_id
                Genome._next_id += 1
                new_population[new_genome.id] = new_genome

        # Generate offspring with mutation
        species_list = list(species_set.species.values())

        while len(new_population) < pop_size:
            # Select species
            species = self._select_species(species_list, fitness_shares)
            if not species or len(species.members) < 2:
                continue

            # Select parents
            parent1, parent2 = self._select_parents(species)

            # Crossover
            child = crossover(parent1, parent2, self.config)

            # Apply mutations
            self._mutate(child)

            new_population[child.id] = child

        return new_population

    def _select_species(self, species_list, fitness_shares):
        if not species_list:
            return None
        weights = [fitness_shares.get(s.key, 0.001) for s in species_list]
        total = sum(weights)
        r = random.random() * total
        cumsum = 0.0
        for s, w in zip(species_list, weights):
            cumsum += w
            if cumsum >= r:
                return s
        return species_list[-1]

    def _select_parents(self, species):
        sorted_members = sorted(
            species.members,
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )
        pool = sorted_members[:max(2, len(sorted_members) // 5)]
        return random.choice(pool), random.choice(pool)

    def _mutate(self, genome: Genome) -> None:
        """Apply mutations to genome."""
        tracker = get_innovation_tracker()
        config = self.config.genome

        # Weight mutation
        if random.random() < self.weight_mutate_prob:
            for conn in genome.connections.values():
                if random.random() < self.weight_perturb_prob:
                    # Perturb
                    conn.weight += random.gauss(0, 0.5)
                else:
                    # Replace
                    conn.weight = random.gauss(config.weight_init_mean, config.weight_init_std)
                conn.weight = max(config.weight_min, min(config.weight_max, conn.weight))

        # Add connection mutation
        if random.random() < self.add_conn_prob:
            self._mutate_add_connection(genome, config, tracker)

        # Add node mutation
        if random.random() < self.add_node_prob:
            self._mutate_add_node(genome, config, tracker)

    def _mutate_add_connection(self, genome, config, tracker):
        """Add a new connection."""
        input_keys = list(genome.nodes.keys())
        output_keys = [k for k, n in genome.nodes.items() if n.node_type != 'input']

        for _ in range(20):  # Try up to 20 times
            in_key = random.choice(input_keys)
            out_key = random.choice(output_keys)

            if in_key == out_key:
                continue
            if (in_key, out_key) in genome.connections:
                continue

            # Check for cycles (simplified)
            if genome.nodes[in_key].node_type == 'output':
                continue

            weight = random.gauss(config.weight_init_mean, config.weight_init_std)
            innovation = tracker.get_connection_innovation(in_key, out_key)

            genome.connections[(in_key, out_key)] = ConnectionGene(
                key=(in_key, out_key),
                weight=weight,
                enabled=True,
                innovation=innovation,
            )
            break

    def _mutate_add_node(self, genome, config, tracker):
        """Add a new node by splitting a connection."""
        enabled_conns = [c for c in genome.connections.values() if c.enabled]
        if not enabled_conns:
            return

        conn = random.choice(enabled_conns)
        conn.enabled = False

        # Create new node
        new_key = max(genome.nodes.keys()) + 1
        genome.nodes[new_key] = NodeGene(
            key=new_key,
            node_type='hidden',
            bias=0.0,
            activation=config.activation_default,
        )

        # Create connections: old_in -> new_node -> old_out
        genome.connections[(conn.in_node, new_key)] = ConnectionGene(
            key=(conn.in_node, new_key),
            weight=1.0,
            enabled=True,
            innovation=tracker.get_connection_innovation(conn.in_node, new_key),
        )

        genome.connections[(new_key, conn.out_node)] = ConnectionGene(
            key=(new_key, conn.out_node),
            weight=conn.weight,
            enabled=True,
            innovation=tracker.get_connection_innovation(new_key, conn.out_node),
        )


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    method: str
    generations: int
    best_fitness: float
    final_complexity: int
    solved: bool
    history: List


def run_syne_experiment(config: Config, max_gens: int, threshold: float) -> ExperimentResult:
    """Run SYNE (symbiogenesis-only)."""
    pop = Population(config)
    pop.initialize()
    best, _ = pop.run(xor_fitness, max_gens, threshold)

    return ExperimentResult(
        method="SYNE (symbiogenesis)",
        generations=pop.generation,
        best_fitness=best.fitness if best else 0,
        final_complexity=best.complexity() if best else 0,
        solved=best.fitness >= threshold if best and best.fitness else False,
        history=pop.history,
    )


def run_mutation_experiment(config: Config, max_gens: int, threshold: float) -> ExperimentResult:
    """Run mutation-based evolution (NEAT-style)."""
    from syne.innovation import reset_innovation_tracker

    reset_innovation_tracker()
    Genome.reset_id_counter()

    reproduction = MutationReproduction(config)
    species_set = SpeciesSet(config)
    population = reproduction.create_initial_population(config.population.population_size)

    best = None
    history = []

    for gen in range(max_gens):
        # Evaluate
        for genome in population.values():
            genome.fitness = xor_fitness(genome)

        # Update best
        current_best = max(population.values(), key=lambda g: g.fitness if g.fitness else float('-inf'))
        if best is None or (current_best.fitness and best.fitness and current_best.fitness > best.fitness):
            best = current_best.copy()

        # Check threshold
        if best.fitness and best.fitness >= threshold:
            break

        # Record history
        mean_fit = sum(g.fitness for g in population.values() if g.fitness) / len(population)
        history.append({
            'generation': gen,
            'best_fitness': best.fitness,
            'mean_fitness': mean_fit,
            'complexity': best.complexity(),
        })

        # Speciate and reproduce
        species_set.speciate(population, gen)
        population = reproduction.reproduce(species_set, config.population.population_size, gen)

    return ExperimentResult(
        method="Mutation-based",
        generations=gen + 1,
        best_fitness=best.fitness if best else 0,
        final_complexity=best.complexity() if best else 0,
        solved=best.fitness >= threshold if best and best.fitness else False,
        history=history,
    )


def run_comparison(num_runs: int = 5):
    """Run comparison between SYNE and mutation-based evolution."""
    print("="*60)
    print("SYNE vs Mutation-based Evolution Comparison")
    print("="*60)
    print(f"\nRunning {num_runs} trials for each method...")
    print()

    config = Config()
    config.genome.num_inputs = 2
    config.genome.num_outputs = 1
    config.genome.activation_default = 'sigmoid'
    config.population.population_size = 100
    config.fusion.fusion_prob = 0.3

    max_gens = 200
    threshold = 3.8

    syne_results = []
    mutation_results = []

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")

        # Reset state
        from syne.innovation import reset_innovation_tracker
        reset_innovation_tracker()
        Genome.reset_id_counter()

        # Run SYNE
        syne_result = run_syne_experiment(config, max_gens, threshold)
        syne_results.append(syne_result)

        # Run mutation-based
        reset_innovation_tracker()
        Genome.reset_id_counter()
        mutation_result = run_mutation_experiment(config, max_gens, threshold)
        mutation_results.append(mutation_result)

    # Print comparison
    print("\n" + "="*60)
    print("Results")
    print("="*60)

    print("\nSYNE (Symbiogenesis-only):")
    print(f"  Success rate: {sum(r.solved for r in syne_results)}/{num_runs}")
    print(f"  Avg generations: {sum(r.generations for r in syne_results)/num_runs:.1f}")
    print(f"  Avg best fitness: {sum(r.best_fitness for r in syne_results)/num_runs:.4f}")
    print(f"  Avg complexity: {sum(r.final_complexity for r in syne_results)/num_runs:.1f}")

    print("\nMutation-based (NEAT-style):")
    print(f"  Success rate: {sum(r.solved for r in mutation_results)}/{num_runs}")
    print(f"  Avg generations: {sum(r.generations for r in mutation_results)/num_runs:.1f}")
    print(f"  Avg best fitness: {sum(r.best_fitness for r in mutation_results)/num_runs:.4f}")
    print(f"  Avg complexity: {sum(r.final_complexity for r in mutation_results)/num_runs:.1f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    run_comparison(num_runs=5)
