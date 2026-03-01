"""
Species management for SYNE.

Implements speciation to protect newly fused genomes and maintain diversity.
Species are groups of genomes with similar structure (low genetic distance).
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import random

from syne_v2.genome import Genome
from syne_v2.config import Config


@dataclass
class Species:
    """
    A species is a group of similar genomes.

    Species serve to:
    1. Protect structural innovations (especially fused genomes)
    2. Maintain population diversity
    3. Enable fitness sharing within similar genomes
    """
    key: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    created_generation: int = 0
    last_improved_generation: int = 0
    fitness_history: List[float] = field(default_factory=list)

    @property
    def fitness(self) -> Optional[float]:
        """Get the best fitness in this species."""
        if not self.members:
            return None
        fitnesses = [m.fitness for m in self.members if m.fitness is not None]
        return max(fitnesses) if fitnesses else None

    @property
    def average_fitness(self) -> Optional[float]:
        """Get the average fitness in this species."""
        if not self.members:
            return None
        fitnesses = [m.fitness for m in self.members if m.fitness is not None]
        return sum(fitnesses) / len(fitnesses) if fitnesses else None

    @property
    def size(self) -> int:
        """Get the number of members."""
        return len(self.members)

    def update_representative(self) -> None:
        """Update representative to a random member."""
        if self.members:
            self.representative = random.choice(self.members)

    def get_best(self) -> Optional[Genome]:
        """Get the fittest genome in this species."""
        if not self.members:
            return None
        return max(self.members, key=lambda g: g.fitness if g.fitness is not None else float('-inf'))

    def get_elites(self, n: int) -> List[Genome]:
        """Get the top n genomes by fitness."""
        sorted_members = sorted(
            self.members,
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )
        return sorted_members[:n]

    def is_stagnant(self, generation: int, stagnation_limit: int) -> bool:
        """Check if this species has stagnated."""
        return (generation - self.last_improved_generation) > stagnation_limit


class SpeciesSet:
    """
    Manages the set of species in the population.

    Handles speciation (assigning genomes to species) and
    tracks species-level statistics.
    """

    def __init__(self, config: Config):
        self.config = config
        self.species: Dict[int, Species] = {}
        self._next_species_key: int = 0
        self.generation: int = 0

    def _get_next_species_key(self) -> int:
        """Get a new unique species key."""
        key = self._next_species_key
        self._next_species_key += 1
        return key

    def speciate(self, population: Dict[int, Genome], generation: int) -> None:
        """
        Assign genomes to species based on genetic distance.

        Each genome is assigned to the first species whose representative
        it is compatible with. If no compatible species exists, a new
        species is created.
        """
        self.generation = generation
        spec_config = self.config.speciation

        # Clear existing members but keep species structure
        for species in self.species.values():
            species.members.clear()

        # Track unspeciated genomes
        unspeciated = set(population.keys())

        # First, try to assign genomes to existing species
        for genome_id in list(unspeciated):
            genome = population[genome_id]

            # Find a compatible species
            for species in self.species.values():
                distance = genome.distance(species.representative, self.config)
                if distance < spec_config.compatibility_threshold:
                    species.members.append(genome)
                    unspeciated.discard(genome_id)
                    break

        # Create new species for remaining genomes
        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            # Check if this genome can join any existing species
            # (second pass with updated members)
            found = False
            for species in self.species.values():
                if species.members:
                    # Use the first member as alternative representative
                    distance = genome.distance(species.members[0], self.config)
                    if distance < spec_config.compatibility_threshold:
                        species.members.append(genome)
                        found = True
                        break

            if not found:
                # Create new species with this genome as representative
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

    def update_fitness_history(self, generation: int) -> None:
        """Update fitness history for all species."""
        for species in self.species.values():
            current_fitness = species.fitness
            if current_fitness is not None:
                species.fitness_history.append(current_fitness)

                # Check if fitness improved
                if len(species.fitness_history) >= 2:
                    if current_fitness > max(species.fitness_history[:-1]):
                        species.last_improved_generation = generation

    def remove_stagnant_species(self, generation: int) -> List[int]:
        """
        Remove species that have stagnated.

        Species minimum enforcement: Never remove species if it would drop
        below the minimum count needed for fusion diversity.

        Returns list of removed species keys.
        """
        spec_config = self.config.speciation

        # SPECIES MINIMUM ENFORCEMENT
        # Fusion requires multiple species for diversity - never go below minimum
        if len(self.species) <= spec_config.species_elitism:
            return []  # Don't remove any species

        # Sort species by fitness (descending) to protect top species
        sorted_species = sorted(
            self.species.values(),
            key=lambda s: s.fitness if s.fitness is not None else float('-inf'),
            reverse=True
        )

        # Protect top species
        protected = set(s.key for s in sorted_species[:spec_config.species_elitism])

        # Remove stagnant species (except protected ones)
        # BUT ensure we maintain minimum species count
        removed = []
        for species in list(self.species.values()):
            if species.key not in protected:
                if species.is_stagnant(generation, spec_config.stagnation_limit):
                    # Only remove if we'd still have enough species remaining
                    if len(self.species) - len(removed) > spec_config.species_elitism:
                        removed.append(species.key)
                        del self.species[species.key]

        return removed

    def get_species_fitness_shares(self) -> Dict[int, float]:
        """
        Calculate fitness shares for each species.

        Returns the fraction of offspring each species should produce
        based on adjusted fitness (fitness sharing).
        """
        if not self.species:
            return {}

        # Calculate adjusted fitness for each species
        # Adjusted fitness = average_fitness (implicit fitness sharing via speciation)
        total_adjusted = 0.0
        adjusted_fitness: Dict[int, float] = {}

        for key, species in self.species.items():
            avg = species.average_fitness
            if avg is not None and avg > 0:
                adjusted_fitness[key] = avg
                total_adjusted += avg
            else:
                adjusted_fitness[key] = 0.0

        # Normalize to get shares
        if total_adjusted > 0:
            return {k: v / total_adjusted for k, v in adjusted_fitness.items()}
        else:
            # Equal shares if no positive fitness
            n = len(self.species)
            return {k: 1.0 / n for k in self.species.keys()}

    def get_best_genome(self) -> Optional[Genome]:
        """Get the best genome across all species."""
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

    def get_species_for_genome(self, genome_id: int) -> Optional[Species]:
        """Find which species a genome belongs to."""
        for species in self.species.values():
            for member in species.members:
                if member.id == genome_id:
                    return species
        return None

    @property
    def num_species(self) -> int:
        """Get the number of active species."""
        return len(self.species)

    def get_stats(self) -> Dict:
        """Get species statistics."""
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
        }
