"""
Reproduction system for SYNE.

Implements pure symbiogenesis-based reproduction: complexity grows through
genome fusion, NOT through mutation. This is the core distinction from NEAT.
"""

from typing import Dict, List, Tuple, Optional, Set
import random
import math

from syne.genome import Genome, crossover, fuse, create_initial_genome
from syne.species import Species, SpeciesSet
from syne.config import Config
from syne.innovation import get_innovation_tracker


class SymbioticReproduction:
    """
    Manages reproduction through symbiogenesis.

    Unlike NEAT which uses mutation for structural innovation, SYNE
    achieves complexity growth purely through genome fusion. The
    reproduction system:

    1. Selects fit genomes for reproduction
    2. Decides between crossover (within species) and fusion (between species)
    3. Creates offspring without any mutation
    """

    def __init__(self, config: Config):
        self.config = config
        self.generation = 0

        # Track fusion statistics
        self.fusion_count = 0
        self.crossover_count = 0

    def create_initial_population(self, pop_size: int) -> Dict[int, Genome]:
        """
        Create the initial population of genomes.

        Creates a diverse initial population where some genomes have
        hidden nodes. This diversity is ESSENTIAL for symbiogenesis
        to work - without structural variation, fusion can't create
        useful new structures.
        """
        population = {}

        # Create diverse initial population
        # ~30% simple (no hidden), ~70% with 1-3 hidden nodes
        for _ in range(pop_size):
            if random.random() < 0.3:
                # Simple genome without hidden nodes
                genome = create_initial_genome(self.config, with_hidden=False)
            else:
                # Genome with hidden node(s) for diversity
                genome = create_initial_genome(self.config, with_hidden=True)
                # Add more hidden nodes with decreasing probability
                while random.random() < 0.3:
                    self._add_hidden_node(genome)

            population[genome.id] = genome

        return population

    def _add_hidden_node(self, genome: Genome) -> None:
        """Add a hidden node connected to random inputs/outputs."""
        from syne.genome import NodeGene, ConnectionGene
        from syne.innovation import get_innovation_tracker

        tracker = get_innovation_tracker()
        config = self.config.genome

        # Create hidden node
        hidden_key = max(genome.nodes.keys()) + 1
        genome.nodes[hidden_key] = NodeGene(
            key=hidden_key,
            node_type='hidden',
            bias=random.gauss(config.bias_init_mean, config.bias_init_std),
            activation=random.choice(config.activation_options) if config.activation_options else config.activation_default,
            response=1.0,
            origin_genome_id=genome.id,
        )

        # Connect to random input and output
        input_keys = [k for k, n in genome.nodes.items() if n.node_type == 'input']
        output_keys = [k for k, n in genome.nodes.items() if n.node_type == 'output']

        in_key = random.choice(input_keys)
        out_key = random.choice(output_keys)

        # Input -> Hidden
        weight1 = random.gauss(config.weight_init_mean, config.weight_init_std)
        key1 = (in_key, hidden_key)
        if key1 not in genome.connections:
            genome.connections[key1] = ConnectionGene(
                key=key1,
                weight=weight1,
                enabled=True,
                innovation=tracker.get_connection_innovation(in_key, hidden_key),
                origin_genome_id=genome.id,
            )

        # Hidden -> Output
        weight2 = random.gauss(config.weight_init_mean, config.weight_init_std)
        key2 = (hidden_key, out_key)
        if key2 not in genome.connections:
            genome.connections[key2] = ConnectionGene(
                key=key2,
                weight=weight2,
                enabled=True,
                innovation=tracker.get_connection_innovation(hidden_key, out_key),
                origin_genome_id=genome.id,
            )

    def reproduce(
        self,
        species_set: SpeciesSet,
        pop_size: int,
        generation: int
    ) -> Dict[int, Genome]:
        """
        Create a new generation through symbiogenesis.

        The process:
        1. Preserve elites from each species
        2. Calculate offspring allocation per species
        3. For each offspring:
           - With fusion_prob, fuse two genomes from different species
           - Otherwise, crossover two genomes from the same species
        """
        self.generation = generation
        new_population: Dict[int, Genome] = {}

        # Get fitness shares for each species
        fitness_shares = species_set.get_species_fitness_shares()

        # Calculate number of offspring per species
        offspring_counts = self._allocate_offspring(
            species_set, fitness_shares, pop_size
        )

        # Preserve elites from each species
        elitism = self.config.speciation.elitism
        for species_id, species in species_set.species.items():
            elites = species.get_elites(min(elitism, len(species.members)))
            for elite in elites:
                new_genome = elite.copy()
                new_genome.id = Genome._next_id
                Genome._next_id += 1
                new_population[new_genome.id] = new_genome

        # Get list of species with their best genomes for fusion
        species_list = list(species_set.species.values())

        # Generate remaining offspring
        remaining = pop_size - len(new_population)

        for species_id, count in offspring_counts.items():
            species = species_set.species.get(species_id)
            if species is None or len(species.members) < 2:
                continue

            # Adjust count to account for elites already preserved
            actual_count = min(count, remaining)
            if actual_count <= 0:
                continue

            for _ in range(actual_count):
                if len(new_population) >= pop_size:
                    break

                offspring = self._create_offspring(species, species_list, generation)
                if offspring is not None:
                    new_population[offspring.id] = offspring

        # Fill any remaining slots with crossover from top species
        while len(new_population) < pop_size:
            # Pick a random species weighted by fitness
            species = self._select_species_by_fitness(species_list, fitness_shares)
            if species and len(species.members) >= 2:
                parent1, parent2 = self._select_parents(species)
                offspring = crossover(parent1, parent2, self.config)
                new_population[offspring.id] = offspring
                self.crossover_count += 1
            else:
                # Fallback: create new random genome
                genome = create_initial_genome(self.config)
                new_population[genome.id] = genome

        # Clear innovation cache for next generation
        get_innovation_tracker().clear_generation_cache()

        return new_population

    def _allocate_offspring(
        self,
        species_set: SpeciesSet,
        fitness_shares: Dict[int, float],
        pop_size: int
    ) -> Dict[int, int]:
        """Allocate offspring counts to each species based on fitness."""
        offspring_counts: Dict[int, int] = {}
        remaining = pop_size

        # Ensure minimum species size
        min_size = self.config.reproduction.min_species_size
        for species_id in species_set.species:
            offspring_counts[species_id] = min_size
            remaining -= min_size

        # Distribute remaining based on fitness shares
        if remaining > 0:
            for species_id, share in fitness_shares.items():
                additional = int(remaining * share)
                offspring_counts[species_id] += additional

        return offspring_counts

    def _create_offspring(
        self,
        species: Species,
        all_species: List[Species],
        generation: int
    ) -> Optional[Genome]:
        """
        Create a single offspring.

        Decides between fusion (inter-species symbiogenesis) and
        crossover (intra-species recombination).
        """
        fusion_config = self.config.fusion

        # Decide: fusion or crossover?
        if (len(all_species) > 1 and
            random.random() < fusion_config.fusion_prob and
            self._can_fuse(species)):

            # FUSION: Select genomes from two different species
            other_species = self._select_fusion_partner(species, all_species)
            if other_species is not None:
                genome1 = self._select_fit_genome(species)
                genome2 = self._select_fit_genome(other_species)

                if genome1 is not None and genome2 is not None:
                    # Check size constraints
                    combined_nodes = len(genome1.nodes) + len(genome2.nodes)
                    if combined_nodes <= fusion_config.max_genome_nodes:
                        self.fusion_count += 1
                        return fuse(genome1, genome2, self.config, generation)

        # CROSSOVER: Select two genomes from the same species
        parent1, parent2 = self._select_parents(species)
        self.crossover_count += 1
        return crossover(parent1, parent2, self.config)

    def _can_fuse(self, species: Species) -> bool:
        """Check if genomes in this species are eligible for fusion."""
        fusion_config = self.config.fusion

        # Check if species has fit enough genomes
        if not species.members:
            return False

        best = species.get_best()
        if best is None:
            return False

        # Check genome size constraint
        if len(best.nodes) > fusion_config.max_genome_nodes // 2:
            return False

        return True

    def _select_fusion_partner(
        self,
        species: Species,
        all_species: List[Species]
    ) -> Optional[Species]:
        """Select another species for fusion."""
        candidates = [s for s in all_species
                      if s.key != species.key and
                      s.size > 0 and
                      self._can_fuse(s)]

        if not candidates:
            return None

        # Weight by fitness
        fitnesses = []
        for s in candidates:
            f = s.fitness if s.fitness is not None else 0.0
            fitnesses.append(max(0.001, f))  # Ensure positive

        total = sum(fitnesses)
        r = random.random() * total
        cumsum = 0.0

        for s, f in zip(candidates, fitnesses):
            cumsum += f
            if cumsum >= r:
                return s

        return candidates[-1] if candidates else None

    def _select_parents(self, species: Species) -> Tuple[Genome, Genome]:
        """Select two parents from a species for crossover."""
        # Sort by fitness (descending)
        sorted_members = sorted(
            species.members,
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )

        # Use survival threshold to determine mating pool
        survival_threshold = self.config.reproduction.survival_threshold
        pool_size = max(2, int(len(sorted_members) * survival_threshold))
        mating_pool = sorted_members[:pool_size]

        # Select two different parents
        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)

        # Ensure different parents if possible
        if len(mating_pool) > 1:
            while parent2.id == parent1.id:
                parent2 = random.choice(mating_pool)

        return parent1, parent2

    def _select_fit_genome(self, species: Species) -> Optional[Genome]:
        """Select a fit genome from a species for fusion."""
        if not species.members:
            return None

        fusion_config = self.config.fusion

        # Filter by fitness threshold
        sorted_members = sorted(
            species.members,
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )

        threshold_idx = int(len(sorted_members) * fusion_config.fusion_fitness_threshold)
        threshold_idx = max(1, threshold_idx)
        candidates = sorted_members[:threshold_idx]

        return random.choice(candidates) if candidates else None

    def _select_species_by_fitness(
        self,
        species_list: List[Species],
        fitness_shares: Dict[int, float]
    ) -> Optional[Species]:
        """Select a species weighted by fitness share."""
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

    def get_stats(self) -> Dict:
        """Get reproduction statistics."""
        total = self.fusion_count + self.crossover_count
        return {
            'total_offspring': total,
            'fusion_count': self.fusion_count,
            'crossover_count': self.crossover_count,
            'fusion_rate': self.fusion_count / total if total > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """Reset reproduction statistics."""
        self.fusion_count = 0
        self.crossover_count = 0
