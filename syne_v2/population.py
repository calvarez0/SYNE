"""
Population management for SYNE.

Orchestrates the evolutionary process using symbiogenesis.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import random
import time

from syne_v2.genome import Genome
from syne_v2.species import SpeciesSet
from syne_v2.reproduction import SymbioticReproduction
from syne_v2.config import Config
from syne_v2.innovation import reset_innovation_tracker


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: Optional[float]
    mean_fitness: Optional[float]
    num_species: int
    best_genome_id: int
    best_genome_size: Tuple[int, int]  # (nodes, connections)
    fusion_count: int
    crossover_count: int
    elapsed_time: float

    def __str__(self) -> str:
        return (
            f"Gen {self.generation:4d} | "
            f"Best: {self.best_fitness:8.4f} | "
            f"Mean: {self.mean_fitness:8.4f} | "
            f"Species: {self.num_species:3d} | "
            f"Size: {self.best_genome_size[0]:3d}n/{self.best_genome_size[1]:3d}c | "
            f"Fusions: {self.fusion_count:3d} | "
            f"Time: {self.elapsed_time:.2f}s"
        )


class Population:
    """
    Manages the population of genomes and drives evolution.

    This is the main class for running SYNE. It handles:
    - Population initialization
    - Fitness evaluation
    - Speciation
    - Reproduction through symbiogenesis
    - Tracking evolutionary progress
    """

    def __init__(self, config: Config):
        """
        Initialize a new population.

        Args:
            config: Configuration for the evolutionary process
        """
        self.config = config
        self.population: Dict[int, Genome] = {}
        self.species_set = SpeciesSet(config)
        self.reproduction = SymbioticReproduction(config)
        self.generation = 0
        self.best_genome: Optional[Genome] = None
        self.history: List[GenerationStats] = []

        # Callbacks
        self.on_generation: Optional[Callable[[int, 'Population'], None]] = None
        self.on_fitness_evaluated: Optional[Callable[[Dict[int, Genome]], None]] = None

        # Reset state
        reset_innovation_tracker()
        Genome.reset_id_counter()

    def initialize(self) -> None:
        """Create the initial population."""
        pop_size = self.config.population.population_size
        self.population = self.reproduction.create_initial_population(pop_size)
        self.generation = 0

    def run(
        self,
        fitness_function: Callable[[Genome], float],
        max_generations: int = 100,
        fitness_threshold: Optional[float] = None,
    ) -> Tuple[Genome, GenerationStats]:
        """
        Run evolution until termination conditions are met.

        Args:
            fitness_function: Function that evaluates a genome and returns fitness
            max_generations: Maximum number of generations
            fitness_threshold: Stop if fitness exceeds this (optional)

        Returns:
            Tuple of (best_genome, final_stats)
        """
        if not self.population:
            self.initialize()

        threshold = fitness_threshold or self.config.population.fitness_threshold

        for gen in range(max_generations):
            stats = self.run_generation(fitness_function)

            # Check termination
            if threshold is not None and stats.best_fitness is not None:
                if stats.best_fitness >= threshold:
                    print(f"\nFitness threshold {threshold} reached!")
                    break

            # Check for extinction
            if self.species_set.num_species == 0:
                if self.config.population.reset_on_extinction:
                    print("\nExtinction! Resetting population...")
                    self.initialize()
                else:
                    print("\nExtinction! Stopping evolution.")
                    break

        return self.best_genome, self.history[-1] if self.history else None

    def run_generation(self, fitness_function: Callable[[Genome], float]) -> GenerationStats:
        """
        Run a single generation of evolution.

        Args:
            fitness_function: Function that evaluates a genome and returns fitness

        Returns:
            Statistics for this generation
        """
        start_time = time.time()
        self.generation += 1

        # Evaluate fitness
        self._evaluate_fitness(fitness_function)

        # Track best genome
        self._update_best()

        # Speciate
        self.species_set.speciate(self.population, self.generation)

        # Update species fitness history
        self.species_set.update_fitness_history(self.generation)

        # Remove stagnant species
        removed = self.species_set.remove_stagnant_species(self.generation)

        # Reproduce (symbiogenesis + crossover, NO mutation)
        self.reproduction.reset_stats()
        pop_size = self.config.population.population_size
        self.population = self.reproduction.reproduce(
            self.species_set, pop_size, self.generation
        )

        # Collect stats
        elapsed = time.time() - start_time
        repro_stats = self.reproduction.get_stats()

        stats = GenerationStats(
            generation=self.generation,
            best_fitness=self.best_genome.fitness if self.best_genome else None,
            mean_fitness=self._mean_fitness(),
            num_species=self.species_set.num_species,
            best_genome_id=self.best_genome.id if self.best_genome else -1,
            best_genome_size=self.best_genome.size() if self.best_genome else (0, 0),
            fusion_count=repro_stats['fusion_count'],
            crossover_count=repro_stats['crossover_count'],
            elapsed_time=elapsed,
        )

        self.history.append(stats)

        # Call generation callback
        if self.on_generation:
            self.on_generation(self.generation, self)

        return stats

    def _evaluate_fitness(self, fitness_function: Callable[[Genome], float]) -> None:
        """Evaluate fitness for all genomes."""
        for genome in self.population.values():
            genome.fitness = fitness_function(genome)

        if self.on_fitness_evaluated:
            self.on_fitness_evaluated(self.population)

    def _update_best(self) -> None:
        """Update the best genome found so far."""
        current_best = max(
            self.population.values(),
            key=lambda g: g.fitness if g.fitness is not None else float('-inf')
        )

        if self.best_genome is None:
            self.best_genome = current_best.copy()
        elif current_best.fitness is not None and self.best_genome.fitness is not None:
            if current_best.fitness > self.best_genome.fitness:
                self.best_genome = current_best.copy()

    def _mean_fitness(self) -> Optional[float]:
        """Calculate mean fitness of population."""
        fitnesses = [g.fitness for g in self.population.values() if g.fitness is not None]
        return sum(fitnesses) / len(fitnesses) if fitnesses else None

    def get_stats(self) -> Dict:
        """Get current population statistics."""
        species_stats = self.species_set.get_stats()

        # Complexity statistics
        complexities = [g.complexity() for g in self.population.values()]

        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_genome.fitness if self.best_genome else None,
            'mean_fitness': self._mean_fitness(),
            'mean_complexity': sum(complexities) / len(complexities) if complexities else 0,
            'max_complexity': max(complexities) if complexities else 0,
            'min_complexity': min(complexities) if complexities else 0,
            **species_stats,
        }

    def print_stats(self) -> None:
        """Print current population statistics."""
        if self.history:
            print(self.history[-1])


def run_syne(
    config: Config,
    fitness_function: Callable[[Genome], float],
    max_generations: int = 100,
    fitness_threshold: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[Genome, List[GenerationStats]]:
    """
    Convenience function to run SYNE evolution.

    Args:
        config: SYNE configuration
        fitness_function: Fitness evaluation function
        max_generations: Maximum generations to run
        fitness_threshold: Target fitness to stop early
        verbose: Whether to print progress

    Returns:
        Tuple of (best_genome, history)
    """
    pop = Population(config)

    if verbose:
        def print_progress(gen: int, population: Population):
            population.print_stats()
        pop.on_generation = print_progress

    pop.initialize()
    best, final_stats = pop.run(fitness_function, max_generations, fitness_threshold)

    if verbose:
        print(f"\nEvolution complete!")
        print(f"Best genome: {best}")
        print(f"Final complexity: {best.complexity()}")

    return best, pop.history
