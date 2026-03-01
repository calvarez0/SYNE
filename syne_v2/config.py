"""
Configuration system for SYNE.

Handles hyperparameters for the symbiogenetic neuroevolution algorithm.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import math


@dataclass
class GenomeConfig:
    """Configuration for genome structure and initialization."""

    # Network structure
    num_inputs: int = 2
    num_outputs: int = 1

    # Initial connectivity: 'none', 'partial', 'full'
    initial_connectivity: str = 'full'
    partial_connectivity_prob: float = 0.5

    # Weight initialization
    weight_init_mean: float = 0.0
    weight_init_std: float = 1.0
    weight_min: float = -30.0
    weight_max: float = 30.0

    # Bias initialization
    bias_init_mean: float = 0.0
    bias_init_std: float = 1.0
    bias_min: float = -30.0
    bias_max: float = 30.0

    # Activation function: 'sigmoid', 'tanh', 'relu', 'identity'
    activation_default: str = 'sigmoid'
    activation_options: List[str] = field(default_factory=lambda: ['sigmoid', 'tanh', 'relu'])

    # Aggregation function: 'sum', 'product', 'max', 'min', 'mean'
    aggregation_default: str = 'sum'


@dataclass
class FusionConfig:
    """Configuration for genome fusion (symbiogenesis)."""

    # Probability of fusion vs. standard crossover per reproduction event
    fusion_prob: float = 0.3

    # How to wire fused genomes: 'sparse', 'moderate', 'dense'
    # sparse: ~5% of possible inter-network connections
    # moderate: ~15% of possible inter-network connections
    # dense: ~30% of possible inter-network connections
    inter_network_connectivity: str = 'sparse'

    # Initial weight magnitude for inter-network connections
    inter_weight_init_std: float = 0.5

    # Whether to prefer connecting outputs of one to inputs of another (sequential)
    # vs. random cross-connections (parallel)
    fusion_topology: str = 'mixed'  # 'sequential', 'parallel', 'mixed'

    # Minimum fitness percentile for fusion candidates
    fusion_fitness_threshold: float = 0.5

    # Maximum genome size (nodes) before fusion is disabled for that genome
    max_genome_nodes: int = 100


@dataclass
class SpeciationConfig:
    """Configuration for speciation and compatibility."""

    # Compatibility distance threshold for speciation
    compatibility_threshold: float = 3.0

    # Coefficients for compatibility distance calculation
    # d = c1*E/N + c2*D/N + c3*W
    # E = excess genes, D = disjoint genes, W = avg weight diff, N = normalizing factor
    excess_coefficient: float = 1.0
    disjoint_coefficient: float = 1.0
    weight_coefficient: float = 0.5

    # Minimum species size before extinction
    min_species_size: int = 2

    # Number of generations before stagnant species are removed
    stagnation_limit: int = 15

    # Elitism: number of top genomes preserved unchanged per species
    elitism: int = 2

    # Species-level elitism: protect top N species from extinction
    species_elitism: int = 2


@dataclass
class ReproductionConfig:
    """Configuration for reproduction."""

    # Fraction of species allowed to reproduce
    survival_threshold: float = 0.2

    # Minimum number of genomes to retain per species
    min_species_size: int = 2


@dataclass
class PopulationConfig:
    """Configuration for population management."""

    population_size: int = 150

    # Fitness criterion: 'max' or 'min'
    fitness_criterion: str = 'max'

    # Fitness threshold for termination (None = no threshold)
    fitness_threshold: Optional[float] = None

    # Whether to reset on extinction
    reset_on_extinction: bool = False


@dataclass
class Config:
    """Master configuration for SYNE."""

    genome: GenomeConfig = field(default_factory=GenomeConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    speciation: SpeciationConfig = field(default_factory=SpeciationConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        """Create config from nested dictionary."""
        genome = GenomeConfig(**d.get('genome', {}))
        fusion = FusionConfig(**d.get('fusion', {}))
        speciation = SpeciationConfig(**d.get('speciation', {}))
        reproduction = ReproductionConfig(**d.get('reproduction', {}))
        population = PopulationConfig(**d.get('population', {}))
        return cls(genome, fusion, speciation, reproduction, population)

    def to_dict(self) -> dict:
        """Convert config to nested dictionary."""
        from dataclasses import asdict
        return {
            'genome': asdict(self.genome),
            'fusion': asdict(self.fusion),
            'speciation': asdict(self.speciation),
            'reproduction': asdict(self.reproduction),
            'population': asdict(self.population),
        }


# Activation functions
def sigmoid(x: float) -> float:
    """Sigmoid activation."""
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x: float) -> float:
    """Hyperbolic tangent activation."""
    return math.tanh(x)


def relu(x: float) -> float:
    """Rectified linear unit activation."""
    return max(0.0, x)


def identity(x: float) -> float:
    """Identity activation."""
    return x


def softplus(x: float) -> float:
    """Softplus activation."""
    x = max(-60.0, min(60.0, x))
    return math.log(1.0 + math.exp(x))


def gaussian(x: float) -> float:
    """Gaussian activation."""
    return math.exp(-x * x)


ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'identity': identity,
    'softplus': softplus,
    'gaussian': gaussian,
}


# Aggregation functions
def sum_agg(values: List[float]) -> float:
    """Sum aggregation."""
    return sum(values)


def product_agg(values: List[float]) -> float:
    """Product aggregation."""
    result = 1.0
    for v in values:
        result *= v
    return result


def max_agg(values: List[float]) -> float:
    """Maximum aggregation."""
    return max(values) if values else 0.0


def min_agg(values: List[float]) -> float:
    """Minimum aggregation."""
    return min(values) if values else 0.0


def mean_agg(values: List[float]) -> float:
    """Mean aggregation."""
    return sum(values) / len(values) if values else 0.0


AGGREGATION_FUNCTIONS = {
    'sum': sum_agg,
    'product': product_agg,
    'max': max_agg,
    'min': min_agg,
    'mean': mean_agg,
}
