"""
Species management for SYNE v5.

Includes proper species minimum enforcement for fusion diversity.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random

from syne_v5.genome import Genome
from syne_v5.config import Config


@dataclass
class Species:
    """A species is a group of similar genomes."""
    key: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    created_generation: int = 0
    last_improved_generation: int = 0
    fitness_history: List[float] = field(default_factory=list)

    @property
    def fitness(self) -> Optional[float]:
        if not self.members:
            return None
        fitnesses = [m.fitness for m in self.members if m.fitness is not None]
        return max(fitnesses) if fitnesses else None

    @property
    def average_fitness(self) -> Optional[float]:
        if not self.members:
            return None
        fitnesses = [m.fitness for m in self.members if m.fitness is not None]
        return sum(fitnesses) / len(fitnesses) if fitnesses else None

    @property
    def size(self) -> int:
        return len(self.members)

    def update_representative(self) -> None:
        if self.members:
            self.representative = random.choice(self.members)

    def get_best(self) -> Optional[Genome]:
        if not self.members:
            return None
        return max(self.members, key=lambda g: g.fitness if g.fitness is not None else float('-inf'))

    def get_elites(self, n: int) -> List[Genome]:
        sorted_members = sorted(
            self.members,
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )
        return sorted_members[:n]

    def is_stagnant(self, generation: int, stagnation_limit: int) -> bool:
        return (generation - self.last_improved_generation) > stagnation_limit


class SpeciesSet:
    """Manages the set of species in the population."""

    def __init__(self, config: Config):
        self.config = config
        self.species: Dict[int, Species] = {}
        self._next_species_key: int = 0
        self.generation: int = 0

        # V5 FIX: Track minimum species count seen during evolution
        self._min_species_count: int = float('inf')

    def _get_next_species_key(self) -> int:
        key = self._next_species_key
        self._next_species_key += 1
        return key

    @property
    def min_species_seen(self) -> int:
        """Return minimum species count observed during evolution."""
        return self._min_species_count if self._min_species_count != float('inf') else 0

    def speciate(self, population: Dict[int, Genome], generation: int) -> None:
        """Assign genomes to species based on genetic distance."""
        self.generation = generation
        spec_config = self.config.speciation

        # Clear existing members
        for species in self.species.values():
            species.members.clear()

        unspeciated = set(population.keys())

        # First pass: assign to existing species
        for genome_id in list(unspeciated):
            genome = population[genome_id]
            for species in self.species.values():
                distance = genome.distance(species.representative, self.config)
                if distance < spec_config.compatibility_threshold:
                    species.members.append(genome)
                    unspeciated.discard(genome_id)
                    break

        # Second pass: create new species for remaining
        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            found = False
            for species in self.species.values():
                if species.members:
                    distance = genome.distance(species.members[0], self.config)
                    if distance < spec_config.compatibility_threshold:
                        species.members.append(genome)
                        found = True
                        break

            if not found:
                species_key = self._get_next_species_key()
                new_species = Species(
                    key=species_key,
                    representative=genome,
                    members=[genome],
                    created_generation=generation,
                    last_improved_generation=generation,
                )
                self.species[species_key] = new_species

        # Remove empty species
        empty_species = [k for k, s in self.species.items() if not s.members]
        for k in empty_species:
            del self.species[k]

        # Update representatives
        for species in self.species.values():
            species.update_representative()

        # V5 FIX: Track minimum species count
        current_count = len(self.species)
        if current_count < self._min_species_count:
            self._min_species_count = current_count

    def update_fitness_history(self, generation: int) -> None:
        for species in self.species.values():
            current_fitness = species.fitness
            if current_fitness is not None:
                species.fitness_history.append(current_fitness)
                if len(species.fitness_history) >= 2:
                    if current_fitness > max(species.fitness_history[:-1]):
                        species.last_improved_generation = generation

    def remove_stagnant_species(self, generation: int) -> List[int]:
        """
        Remove stagnant species with species minimum enforcement.

        V5: Never remove species if count would drop below species_elitism.
        """
        spec_config = self.config.speciation

        # SPECIES MINIMUM ENFORCEMENT (v2/v5 feature)
        # Don't remove any species if we're at or below minimum
        if len(self.species) <= spec_config.species_elitism:
            return []

        # Sort by fitness to protect top species
        sorted_species = sorted(
            self.species.values(),
            key=lambda s: s.fitness if s.fitness is not None else float('-inf'),
            reverse=True
        )

        protected = set(s.key for s in sorted_species[:spec_config.species_elitism])

        removed = []
        for species in list(self.species.values()):
            if species.key not in protected:
                if species.is_stagnant(generation, spec_config.stagnation_limit):
                    # Only remove if we'd still have enough species
                    if len(self.species) - len(removed) > spec_config.species_elitism:
                        removed.append(species.key)
                        del self.species[species.key]

        return removed

    def get_species_fitness_shares(self) -> Dict[int, float]:
        if not self.species:
            return {}

        total_adjusted = 0.0
        adjusted_fitness: Dict[int, float] = {}

        for key, species in self.species.items():
            avg = species.average_fitness
            if avg is not None and avg > 0:
                adjusted_fitness[key] = avg
                total_adjusted += avg
            else:
                adjusted_fitness[key] = 0.0

        if total_adjusted > 0:
            return {k: v / total_adjusted for k, v in adjusted_fitness.items()}
        else:
            n = len(self.species)
            return {k: 1.0 / n for k in self.species.keys()}

    def get_best_genome(self) -> Optional[Genome]:
        best = None
        best_fitness = float('-inf')

        for species in self.species.values():
            species_best = species.get_best()
            if species_best is not None:
                fitness = species_best.fitness if species_best.fitness is not None else float('-inf')
                if fitness > best_fitness:
                    best = species_best
                    best_fitness = fitness

        return best

    @property
    def num_species(self) -> int:
        return len(self.species)

    def get_stats(self) -> Dict:
        sizes = [s.size for s in self.species.values()]
        fitnesses = [s.fitness for s in self.species.values() if s.fitness is not None]

        return {
            'num_species': len(self.species),
            'species_sizes': sizes,
            'min_species_size': min(sizes) if sizes else 0,
            'max_species_size': max(sizes) if sizes else 0,
            'mean_species_size': sum(sizes) / len(sizes) if sizes else 0,
            'best_species_fitness': max(fitnesses) if fitnesses else None,
            'mean_species_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else None,
            'min_species_seen': self.min_species_seen,  # V5: expose this
        }
