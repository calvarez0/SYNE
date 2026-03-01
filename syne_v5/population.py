"""
Population management for SYNE v5.

V5 fixes:
- Properly track cumulative fusion counts
- Track minimum species count seen
- Better statistics reporting
"""

from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import time

from syne_v5.genome import Genome
from syne_v5.species import SpeciesSet
from syne_v5.reproduction import SymbioticReproduction
from syne_v5.config import Config
from syne_v5.innovation import reset_innovation_tracker


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: Optional[float]
    mean_fitness: Optional[float]
    num_species: int
    best_genome_id: int
    best_genome_size: Tuple[int, int]
    fusion_count: int  # This generation
    crossover_count: int  # This generation
    total_fusions: int  # Cumulative V5 FIX
    min_species_seen: int  # Minimum species seen so far V5 FIX
    elapsed_time: float

    def __str__(self) -> str:
        return (
            f"Gen {self.generation:4d} | "
            f"Best: {self.best_fitness:8.4f} | "
            f"Mean: {self.mean_fitness:8.4f} | "
            f"Species: {self.num_species:3d} | "
            f"Size: {self.best_genome_size[0]:3d}n/{self.best_genome_size[1]:3d}c | "
            f"Fusions: {self.fusion_count:3d} (total: {self.total_fusions}) | "
            f"Time: {self.elapsed_time:.2f}s"
        )


class Population:
    """Manages the population of genomes and drives evolution."""

    def __init__(self, config: Config):
        self.config = config
        self.population: Dict[int, Genome] = {}
        self.species_set = SpeciesSet(config)
        self.reproduction = SymbioticReproduction(config)
        self.generation = 0
        self.best_genome: Optional[Genome] = None
        self.history: List[GenerationStats] = []

        self.on_generation: Optional[Callable[[int, 'Population'], None]] = None
        self.on_fitness_evaluated: Optional[Callable[[Dict[int, Genome]], None]] = None

        reset_innovation_tracker()
        Genome.reset_id_counter()

    def initialize(self) -> None:
        """Create the initial population."""
        pop_size = self.config.population.population_size
        self.population = self.reproduction.create_initial_population(pop_size)
        self.generation = 0
        # Reset cumulative stats for new run
        self.reproduction.reset_stats()

    def run(
        self,
        fitness_function: Callable[[Genome], float],
        max_generations: int = 100,
        fitness_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> Tuple[Genome, GenerationStats]:
        """Run evolution until termination conditions are met."""
        if not self.population:
            self.initialize()

        threshold = fitness_threshold or self.config.population.fitness_threshold

        for gen in range(max_generations):
            stats = self.run_generation(fitness_function)

            if verbose:
                print(stats)

            if threshold is not None and stats.best_fitness is not None:
                if stats.best_fitness >= threshold:
                    if verbose:
                        print(f"\nFitness threshold {threshold} reached!")
                    break

            if self.species_set.num_species == 0:
                if self.config.population.reset_on_extinction:
                    if verbose:
                        print("\nExtinction! Resetting population...")
                    self.initialize()
                else:
                    if verbose:
                        print("\nExtinction! Stopping evolution.")
                    break

        return self.best_genome, self.history[-1] if self.history else None

    def run_generation(self, fitness_function: Callable[[Genome], float]) -> GenerationStats:
        """Run a single generation of evolution."""
        start_time = time.time()
        self.generation += 1

        self._evaluate_fitness(fitness_function)
        self._update_best()

        self.species_set.speciate(self.population, self.generation)
        self.species_set.update_fitness_history(self.generation)
        self.species_set.remove_stagnant_species(self.generation)

        # Reset per-generation stats but keep cumulative
        self.reproduction.reset_generation_stats()

        pop_size = self.config.population.population_size
        self.population = self.reproduction.reproduce(
            self.species_set, pop_size, self.generation
        )

        elapsed = time.time() - start_time
        repro_stats = self.reproduction.get_stats()
        cumulative_stats = self.reproduction.get_cumulative_stats()

        stats = GenerationStats(
            generation=self.generation,
            best_fitness=self.best_genome.fitness if self.best_genome else None,
            mean_fitness=self._mean_fitness(),
            num_species=self.species_set.num_species,
            best_genome_id=self.best_genome.id if self.best_genome else -1,
            best_genome_size=self.best_genome.size() if self.best_genome else (0, 0),
            fusion_count=repro_stats['fusion_count'],
            crossover_count=repro_stats['crossover_count'],
            total_fusions=cumulative_stats['total_fusions'],  # V5 FIX
            min_species_seen=self.species_set.min_species_seen,  # V5 FIX
            elapsed_time=elapsed,
        )

        self.history.append(stats)

        if self.on_generation:
            self.on_generation(self.generation, self)

        return stats

    def _evaluate_fitness(self, fitness_function: Callable[[Genome], float]) -> None:
        for genome in self.population.values():
            genome.fitness = fitness_function(genome)

        if self.on_fitness_evaluated:
            self.on_fitness_evaluated(self.population)

    def _update_best(self) -> None:
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
        fitnesses = [g.fitness for g in self.population.values() if g.fitness is not None]
        return sum(fitnesses) / len(fitnesses) if fitnesses else None

    def get_stats(self) -> Dict:
        species_stats = self.species_set.get_stats()
        complexities = [g.complexity() for g in self.population.values()]
        cumulative_stats = self.reproduction.get_cumulative_stats()

        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_genome.fitness if self.best_genome else None,
            'mean_fitness': self._mean_fitness(),
            'mean_complexity': sum(complexities) / len(complexities) if complexities else 0,
            'max_complexity': max(complexities) if complexities else 0,
            'min_complexity': min(complexities) if complexities else 0,
            'total_fusions': cumulative_stats['total_fusions'],
            'total_crossovers': cumulative_stats['total_crossovers'],
            **species_stats,
        }

    def print_stats(self) -> None:
        if self.history:
            print(self.history[-1])


def run_syne(
    config: Config,
    fitness_function: Callable[[Genome], float],
    max_generations: int = 100,
    fitness_threshold: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[Genome, List[GenerationStats]]:
    """Convenience function to run SYNE evolution."""
    pop = Population(config)

    if verbose:
        def print_progress(gen: int, population: Population):
            population.print_stats()
        pop.on_generation = print_progress

    pop.initialize()
    best, final_stats = pop.run(fitness_function, max_generations, fitness_threshold, verbose=not verbose)

    if verbose:
        print(f"\nEvolution complete!")
        print(f"Best genome: {best}")
        print(f"Final complexity: {best.complexity()}")
        cumulative = pop.reproduction.get_cumulative_stats()
        print(f"Total fusions: {cumulative['total_fusions']}")
        print(f"Total crossovers: {cumulative['total_crossovers']}")

    return best, pop.history
