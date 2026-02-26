"""
SYNE vs NEAT Comprehensive Comparison

This script runs rigorous benchmarks comparing SYNE (symbiogenesis-based)
against official neat-python (mutation-based) on standard tasks.

Metrics measured (based on MODES framework and neuroevolution literature):
1. Performance: Fitness over generations, success rate, generations to solution
2. Complexity: Network size (nodes, connections) over time
3. Diversity: Species count, population variance
4. Novelty: Rate of new structural innovations
5. Efficiency: Computational cost per generation

References:
- MODES Toolbox (Dolson et al., 2019): https://direct.mit.edu/artl/article/25/1/50/2915
- NEAT paper (Stanley & Miikkulainen, 2002)
- Agüera y Arcas et al. (2024): Computational symbiogenesis
"""

import sys
import os
import json
import time
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SYNE
from syne import Config, Population
from syne.nn import FeedForwardNetwork
from syne.genome import Genome
from syne.innovation import reset_innovation_tracker

# Import neat-python
import neat

# ============================================================================
# BENCHMARK TASKS
# ============================================================================

# XOR truth table
XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_OUTPUTS = [[0.0], [1.0], [1.0], [0.0]]


def xor_fitness_syne(genome: Genome) -> float:
    """XOR fitness for SYNE genome."""
    try:
        network = FeedForwardNetwork.create(genome)
    except Exception:
        return 0.0

    error = 0.0
    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = network.activate(inputs)
        error += (output[0] - expected[0]) ** 2

    return 4.0 - error


def xor_fitness_neat(genomes, config):
    """XOR fitness for NEAT genomes."""
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        error = 0.0
        for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
            output = net.activate(inputs)
            error += (output[0] - expected[0]) ** 2
        genome.fitness = 4.0 - error


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GenerationData:
    """Data collected per generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    fitness_std: float
    num_species: int
    best_nodes: int
    best_connections: int
    mean_nodes: float
    mean_connections: float
    max_complexity: int
    min_complexity: int
    elapsed_time: float
    # SYNE-specific
    fusion_count: int = 0
    crossover_count: int = 0


@dataclass
class RunResult:
    """Results from a single experimental run."""
    algorithm: str
    run_id: int
    solved: bool
    generations_to_solve: Optional[int]
    final_fitness: float
    final_nodes: int
    final_connections: int
    total_time: float
    history: List[GenerationData]


@dataclass
class ExperimentResults:
    """Aggregated results from multiple runs."""
    algorithm: str
    num_runs: int
    success_rate: float
    mean_generations_to_solve: float
    std_generations_to_solve: float
    mean_final_fitness: float
    mean_final_complexity: float
    mean_time_per_generation: float
    all_runs: List[RunResult]


# ============================================================================
# SYNE RUNNER
# ============================================================================

class SYNERunner:
    """Runner for SYNE experiments with detailed metric collection."""

    def __init__(self, max_generations: int = 300, fitness_threshold: float = 3.9):
        self.max_generations = max_generations
        self.fitness_threshold = fitness_threshold

    def run(self, run_id: int) -> RunResult:
        """Run a single SYNE experiment."""
        reset_innovation_tracker()
        Genome.reset_id_counter()

        # Configure SYNE
        config = Config()
        config.genome.num_inputs = 2
        config.genome.num_outputs = 1
        config.genome.activation_default = 'sigmoid'
        config.genome.initial_connectivity = 'full'

        config.fusion.fusion_prob = 0.5
        config.fusion.inter_network_connectivity = 'moderate'
        config.fusion.max_genome_nodes = 50
        config.fusion.fusion_fitness_threshold = 0.3

        config.population.population_size = 150
        config.population.fitness_threshold = self.fitness_threshold

        config.speciation.compatibility_threshold = 1.0
        config.speciation.stagnation_limit = 20
        config.speciation.species_elitism = 3

        # Initialize
        pop = Population(config)
        pop.initialize()

        history = []
        solved = False
        generations_to_solve = None
        start_time = time.time()

        for gen in range(self.max_generations):
            gen_start = time.time()

            # Evaluate fitness
            for genome in pop.population.values():
                genome.fitness = xor_fitness_syne(genome)

            # Collect metrics
            fitnesses = [g.fitness for g in pop.population.values() if g.fitness is not None]
            complexities = [(len(g.nodes), len(g.connections)) for g in pop.population.values()]

            best_genome = max(pop.population.values(), key=lambda g: g.fitness or 0)

            gen_data = GenerationData(
                generation=gen,
                best_fitness=max(fitnesses) if fitnesses else 0,
                mean_fitness=np.mean(fitnesses) if fitnesses else 0,
                fitness_std=np.std(fitnesses) if fitnesses else 0,
                num_species=pop.species_set.num_species,
                best_nodes=len(best_genome.nodes),
                best_connections=len(best_genome.connections),
                mean_nodes=np.mean([c[0] for c in complexities]),
                mean_connections=np.mean([c[1] for c in complexities]),
                max_complexity=max(c[0] + c[1] for c in complexities),
                min_complexity=min(c[0] + c[1] for c in complexities),
                elapsed_time=time.time() - gen_start,
                fusion_count=pop.reproduction.fusion_count,
                crossover_count=pop.reproduction.crossover_count,
            )
            history.append(gen_data)

            # Check for solution
            if gen_data.best_fitness >= self.fitness_threshold:
                solved = True
                generations_to_solve = gen
                break

            # Update best
            pop._update_best()

            # Speciate
            pop.species_set.speciate(pop.population, gen)
            pop.species_set.update_fitness_history(gen)
            pop.species_set.remove_stagnant_species(gen)

            # Reproduce
            pop.reproduction.reset_stats()
            pop.population = pop.reproduction.reproduce(
                pop.species_set, config.population.population_size, gen
            )

        total_time = time.time() - start_time

        return RunResult(
            algorithm="SYNE",
            run_id=run_id,
            solved=solved,
            generations_to_solve=generations_to_solve,
            final_fitness=history[-1].best_fitness if history else 0,
            final_nodes=history[-1].best_nodes if history else 0,
            final_connections=history[-1].best_connections if history else 0,
            total_time=total_time,
            history=history,
        )


# ============================================================================
# NEAT RUNNER
# ============================================================================

class NEATRunner:
    """Runner for official neat-python experiments."""

    def __init__(self, config_path: str, max_generations: int = 300, fitness_threshold: float = 3.9):
        self.config_path = config_path
        self.max_generations = max_generations
        self.fitness_threshold = fitness_threshold

    def run(self, run_id: int) -> RunResult:
        """Run a single NEAT experiment."""
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

        pop = neat.Population(config)

        history = []
        solved = False
        generations_to_solve = None
        start_time = time.time()
        best_genome = None

        for gen in range(self.max_generations):
            gen_start = time.time()

            # Evaluate fitness
            xor_fitness_neat(list(pop.population.items()), config)

            # Collect metrics
            fitnesses = [g.fitness for g in pop.population.values() if g.fitness is not None]

            # Get complexity info
            complexities = []
            for g in pop.population.values():
                num_nodes = len(g.nodes)
                num_conns = len([c for c in g.connections.values() if c.enabled])
                complexities.append((num_nodes, num_conns))

            current_best = max(pop.population.values(), key=lambda g: g.fitness or 0)
            if best_genome is None or (current_best.fitness or 0) > (best_genome.fitness or 0):
                best_genome = current_best

            best_nodes = len(best_genome.nodes)
            best_conns = len([c for c in best_genome.connections.values() if c.enabled])

            gen_data = GenerationData(
                generation=gen,
                best_fitness=max(fitnesses) if fitnesses else 0,
                mean_fitness=np.mean(fitnesses) if fitnesses else 0,
                fitness_std=np.std(fitnesses) if fitnesses else 0,
                num_species=len(pop.species.species),
                best_nodes=best_nodes,
                best_connections=best_conns,
                mean_nodes=np.mean([c[0] for c in complexities]),
                mean_connections=np.mean([c[1] for c in complexities]),
                max_complexity=max(c[0] + c[1] for c in complexities),
                min_complexity=min(c[0] + c[1] for c in complexities),
                elapsed_time=time.time() - gen_start,
            )
            history.append(gen_data)

            # Check for solution
            if gen_data.best_fitness >= self.fitness_threshold:
                solved = True
                generations_to_solve = gen
                break

            # Run reproduction for next generation
            pop.species.speciate(config, pop.population, gen)
            pop.population = pop.reproduction.reproduce(
                config, pop.species, config.pop_size, gen
            )

        total_time = time.time() - start_time

        return RunResult(
            algorithm="NEAT",
            run_id=run_id,
            solved=solved,
            generations_to_solve=generations_to_solve,
            final_fitness=history[-1].best_fitness if history else 0,
            final_nodes=history[-1].best_nodes if history else 0,
            final_connections=history[-1].best_connections if history else 0,
            total_time=total_time,
            history=history,
        )


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(num_runs: int = 10, max_generations: int = 300) -> Tuple[ExperimentResults, ExperimentResults]:
    """Run full comparison experiment."""
    print("="*70)
    print("SYNE vs NEAT Comparison Experiment")
    print("="*70)
    print(f"Runs per algorithm: {num_runs}")
    print(f"Max generations: {max_generations}")
    print(f"Fitness threshold: 3.9 (XOR)")
    print()

    config_path = os.path.join(os.path.dirname(__file__), "neat_config.txt")

    syne_runner = SYNERunner(max_generations=max_generations)
    neat_runner = NEATRunner(config_path, max_generations=max_generations)

    syne_results = []
    neat_results = []

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")

        # Run SYNE
        print("Running SYNE...", end=" ", flush=True)
        syne_result = syne_runner.run(i)
        status = f"Solved in {syne_result.generations_to_solve} gens" if syne_result.solved else "Not solved"
        print(f"{status} (fitness: {syne_result.final_fitness:.4f})")
        syne_results.append(syne_result)

        # Run NEAT
        print("Running NEAT...", end=" ", flush=True)
        neat_result = neat_runner.run(i)
        status = f"Solved in {neat_result.generations_to_solve} gens" if neat_result.solved else "Not solved"
        print(f"{status} (fitness: {neat_result.final_fitness:.4f})")
        neat_results.append(neat_result)

    # Aggregate SYNE results
    syne_solved = [r for r in syne_results if r.solved]
    syne_gens = [r.generations_to_solve for r in syne_solved]

    syne_agg = ExperimentResults(
        algorithm="SYNE",
        num_runs=num_runs,
        success_rate=len(syne_solved) / num_runs,
        mean_generations_to_solve=np.mean(syne_gens) if syne_gens else float('inf'),
        std_generations_to_solve=np.std(syne_gens) if syne_gens else 0,
        mean_final_fitness=np.mean([r.final_fitness for r in syne_results]),
        mean_final_complexity=np.mean([r.final_nodes + r.final_connections for r in syne_results]),
        mean_time_per_generation=np.mean([r.total_time / len(r.history) for r in syne_results if r.history]),
        all_runs=syne_results,
    )

    # Aggregate NEAT results
    neat_solved = [r for r in neat_results if r.solved]
    neat_gens = [r.generations_to_solve for r in neat_solved]

    neat_agg = ExperimentResults(
        algorithm="NEAT",
        num_runs=num_runs,
        success_rate=len(neat_solved) / num_runs,
        mean_generations_to_solve=np.mean(neat_gens) if neat_gens else float('inf'),
        std_generations_to_solve=np.std(neat_gens) if neat_gens else 0,
        mean_final_fitness=np.mean([r.final_fitness for r in neat_results]),
        mean_final_complexity=np.mean([r.final_nodes + r.final_connections for r in neat_results]),
        mean_time_per_generation=np.mean([r.total_time / len(r.history) for r in neat_results if r.history]),
        all_runs=neat_results,
    )

    return syne_agg, neat_agg


def print_results(syne: ExperimentResults, neat: ExperimentResults):
    """Print comparison results."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Metric':<35} {'SYNE':>15} {'NEAT':>15}")
    print("-"*70)
    print(f"{'Success Rate':<35} {syne.success_rate*100:>14.1f}% {neat.success_rate*100:>14.1f}%")
    print(f"{'Mean Generations to Solve':<35} {syne.mean_generations_to_solve:>15.1f} {neat.mean_generations_to_solve:>15.1f}")
    print(f"{'Std Generations to Solve':<35} {syne.std_generations_to_solve:>15.1f} {neat.std_generations_to_solve:>15.1f}")
    print(f"{'Mean Final Fitness':<35} {syne.mean_final_fitness:>15.4f} {neat.mean_final_fitness:>15.4f}")
    print(f"{'Mean Final Complexity':<35} {syne.mean_final_complexity:>15.1f} {neat.mean_final_complexity:>15.1f}")
    print(f"{'Mean Time per Generation (s)':<35} {syne.mean_time_per_generation:>15.4f} {neat.mean_time_per_generation:>15.4f}")


def save_results(syne: ExperimentResults, neat: ExperimentResults, output_dir: str):
    """Save results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to serializable format
    def convert_result(result: ExperimentResults) -> dict:
        d = {
            'algorithm': result.algorithm,
            'num_runs': result.num_runs,
            'success_rate': result.success_rate,
            'mean_generations_to_solve': result.mean_generations_to_solve,
            'std_generations_to_solve': result.std_generations_to_solve,
            'mean_final_fitness': result.mean_final_fitness,
            'mean_final_complexity': result.mean_final_complexity,
            'mean_time_per_generation': result.mean_time_per_generation,
            'runs': []
        }

        for run in result.all_runs:
            run_data = {
                'run_id': run.run_id,
                'solved': run.solved,
                'generations_to_solve': run.generations_to_solve,
                'final_fitness': run.final_fitness,
                'final_nodes': run.final_nodes,
                'final_connections': run.final_connections,
                'total_time': run.total_time,
                'history': [asdict(h) for h in run.history]
            }
            d['runs'].append(run_data)

        return d

    with open(os.path.join(output_dir, 'syne_results.json'), 'w') as f:
        json.dump(convert_result(syne), f, indent=2)

    with open(os.path.join(output_dir, 'neat_results.json'), 'w') as f:
        json.dump(convert_result(neat), f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    # Run experiment
    syne_results, neat_results = run_experiment(num_runs=10, max_generations=300)

    # Print summary
    print_results(syne_results, neat_results)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    save_results(syne_results, neat_results, output_dir)
