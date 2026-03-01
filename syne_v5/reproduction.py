"""
Reproduction system for SYNE v5.

V5 fixes:
- Proper accumulation of fusion counts across generations
- Better tracking statistics
"""

from typing import Dict, List, Tuple, Optional
import random

from syne_v5.genome import Genome, crossover, fuse, create_initial_genome
from syne_v5.species import Species, SpeciesSet
from syne_v5.config import Config
from syne_v5.innovation import get_innovation_tracker


class SymbioticReproduction:
    """Manages reproduction through symbiogenesis."""

    def __init__(self, config: Config):
        self.config = config
        self.generation = 0

        # V5 FIX: Track both per-generation and cumulative stats
        self.fusion_count_generation = 0
        self.crossover_count_generation = 0
        self.total_fusions = 0  # Cumulative across all generations
        self.total_crossovers = 0  # Cumulative across all generations

    def create_initial_population(self, pop_size: int) -> Dict[int, Genome]:
        """Create the initial population of genomes."""
        population = {}

        for _ in range(pop_size):
            if random.random() < 0.3:
                genome = create_initial_genome(self.config, with_hidden=False)
            else:
                genome = create_initial_genome(self.config, with_hidden=True)
                while random.random() < 0.3:
                    self._add_hidden_node(genome)

            population[genome.id] = genome

        return population

    def _add_hidden_node(self, genome: Genome) -> None:
        """Add a hidden node connected to random inputs/outputs."""
        from syne_v5.genome import NodeGene, ConnectionGene
        from syne_v5.innovation import get_innovation_tracker

        tracker = get_innovation_tracker()
        config = self.config.genome

        hidden_key = max(genome.nodes.keys()) + 1
        genome.nodes[hidden_key] = NodeGene(
            key=hidden_key,
            node_type='hidden',
            bias=random.gauss(config.bias_init_mean, config.bias_init_std),
            activation=random.choice(config.activation_options) if config.activation_options else config.activation_default,
            response=1.0,
            origin_genome_id=genome.id,
        )

        input_keys = [k for k, n in genome.nodes.items() if n.node_type == 'input']
        output_keys = [k for k, n in genome.nodes.items() if n.node_type == 'output']

        in_key = random.choice(input_keys)
        out_key = random.choice(output_keys)

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
        """Create a new generation through symbiogenesis."""
        self.generation = generation
        new_population: Dict[int, Genome] = {}

        fitness_shares = species_set.get_species_fitness_shares()
        offspring_counts = self._allocate_offspring(species_set, fitness_shares, pop_size)

        # Preserve elites
        elitism = self.config.speciation.elitism
        for species_id, species in species_set.species.items():
            elites = species.get_elites(min(elitism, len(species.members)))
            for elite in elites:
                new_genome = elite.copy()
                new_genome.id = Genome._next_id
                Genome._next_id += 1
                new_population[new_genome.id] = new_genome

        species_list = list(species_set.species.values())
        remaining = pop_size - len(new_population)

        for species_id, count in offspring_counts.items():
            species = species_set.species.get(species_id)
            if species is None or len(species.members) < 2:
                continue

            actual_count = min(count, remaining)
            if actual_count <= 0:
                continue

            for _ in range(actual_count):
                if len(new_population) >= pop_size:
                    break

                offspring = self._create_offspring(species, species_list, generation)
                if offspring is not None:
                    new_population[offspring.id] = offspring

        while len(new_population) < pop_size:
            species = self._select_species_by_fitness(species_list, fitness_shares)
            if species and len(species.members) >= 2:
                parent1, parent2 = self._select_parents(species)
                offspring = crossover(parent1, parent2, self.config)
                new_population[offspring.id] = offspring
                self.crossover_count_generation += 1
                self.total_crossovers += 1
            else:
                genome = create_initial_genome(self.config)
                new_population[genome.id] = genome

        get_innovation_tracker().clear_generation_cache()

        return new_population

    def _allocate_offspring(
        self,
        species_set: SpeciesSet,
        fitness_shares: Dict[int, float],
        pop_size: int
    ) -> Dict[int, int]:
        offspring_counts: Dict[int, int] = {}
        remaining = pop_size

        min_size = self.config.reproduction.min_species_size
        for species_id in species_set.species:
            offspring_counts[species_id] = min_size
            remaining -= min_size

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
        """Create a single offspring via fusion or crossover."""
        fusion_config = self.config.fusion

        # Decide: fusion or crossover?
        if (len(all_species) > 1 and
            random.random() < fusion_config.fusion_prob and
            self._can_fuse(species)):

            other_species = self._select_fusion_partner(species, all_species)
            if other_species is not None:
                genome1 = self._select_fit_genome(species)
                genome2 = self._select_fit_genome(other_species)

                if genome1 is not None and genome2 is not None:
                    combined_nodes = len(genome1.nodes) + len(genome2.nodes)
                    if combined_nodes <= fusion_config.max_genome_nodes:
                        # V5 FIX: Increment both counters
                        self.fusion_count_generation += 1
                        self.total_fusions += 1
                        return fuse(genome1, genome2, self.config, generation)

        # Fallback to crossover
        parent1, parent2 = self._select_parents(species)
        self.crossover_count_generation += 1
        self.total_crossovers += 1
        return crossover(parent1, parent2, self.config)

    def _can_fuse(self, species: Species) -> bool:
        fusion_config = self.config.fusion

        if not species.members:
            return False

        best = species.get_best()
        if best is None:
            return False

        if len(best.nodes) > fusion_config.max_genome_nodes // 2:
            return False

        return True

    def _select_fusion_partner(
        self,
        species: Species,
        all_species: List[Species]
    ) -> Optional[Species]:
        candidates = [s for s in all_species
                      if s.key != species.key and
                      s.size > 0 and
                      self._can_fuse(s)]

        if not candidates:
            return None

        fitnesses = []
        for s in candidates:
            f = s.fitness if s.fitness is not None else 0.0
            fitnesses.append(max(0.001, f))

        total = sum(fitnesses)
        r = random.random() * total
        cumsum = 0.0

        for s, f in zip(candidates, fitnesses):
            cumsum += f
            if cumsum >= r:
                return s

        return candidates[-1] if candidates else None

    def _select_parents(self, species: Species) -> Tuple[Genome, Genome]:
        sorted_members = sorted(
            species.members,
            key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
            reverse=True
        )

        survival_threshold = self.config.reproduction.survival_threshold
        pool_size = max(2, int(len(sorted_members) * survival_threshold))
        mating_pool = sorted_members[:pool_size]

        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)

        if len(mating_pool) > 1:
            while parent2.id == parent1.id:
                parent2 = random.choice(mating_pool)

        return parent1, parent2

    def _select_fit_genome(self, species: Species) -> Optional[Genome]:
        if not species.members:
            return None

        fusion_config = self.config.fusion

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
        """Get reproduction statistics for current generation."""
        total = self.fusion_count_generation + self.crossover_count_generation
        return {
            'total_offspring': total,
            'fusion_count': self.fusion_count_generation,
            'crossover_count': self.crossover_count_generation,
            'fusion_rate': self.fusion_count_generation / total if total > 0 else 0.0,
        }

    def get_cumulative_stats(self) -> Dict:
        """Get cumulative stats across all generations."""
        total = self.total_fusions + self.total_crossovers
        return {
            'total_fusions': self.total_fusions,
            'total_crossovers': self.total_crossovers,
            'total_offspring': total,
            'cumulative_fusion_rate': self.total_fusions / total if total > 0 else 0.0,
        }

    def reset_generation_stats(self) -> None:
        """Reset per-generation stats (but keep cumulative)."""
        self.fusion_count_generation = 0
        self.crossover_count_generation = 0

    def reset_stats(self) -> None:
        """Reset both generation and cumulative stats."""
        self.fusion_count_generation = 0
        self.crossover_count_generation = 0
        self.total_fusions = 0
        self.total_crossovers = 0
