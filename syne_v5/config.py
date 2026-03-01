"""
Configuration system for SYNE v5.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class GenomeConfig:
    """Configuration for genome structure and initialization."""
    num_inputs: int = 2
    num_outputs: int = 1
    initial_connectivity: str = 'full'  # 'none', 'partial', 'full'
    partial_connectivity_prob: float = 0.5

    weight_init_mean: float = 0.0
    weight_init_std: float = 1.0
    weight_min: float = -30.0
    weight_max: float = 30.0

    bias_init_mean: float = 0.0
    bias_init_std: float = 1.0
    bias_min: float = -30.0
    bias_max: float = 30.0

    activation_default: str = 'sigmoid'
    activation_options: List[str] = field(default_factory=lambda: ['sigmoid', 'tanh', 'relu'])
    aggregation_default: str = 'sum'


@dataclass
class FusionConfig:
    """Configuration for genome fusion (symbiogenesis)."""
    fusion_prob: float = 0.3
    inter_network_connectivity: str = 'sparse'  # 'sparse', 'moderate', 'dense'
    inter_weight_init_std: float = 0.5
    fusion_topology: str = 'mixed'  # 'sequential', 'parallel', 'mixed'
    fusion_fitness_threshold: float = 0.5
    max_genome_nodes: int = 100


@dataclass
class SpeciationConfig:
    """Configuration for speciation and compatibility."""
    compatibility_threshold: float = 3.0
    excess_coefficient: float = 1.0
    disjoint_coefficient: float = 1.0
    weight_coefficient: float = 0.5
    min_species_size: int = 2
    stagnation_limit: int = 15
    elitism: int = 2
    species_elitism: int = 2  # Minimum species to maintain for fusion diversity


@dataclass
class ReproductionConfig:
    """Configuration for reproduction."""
    survival_threshold: float = 0.2
    min_species_size: int = 2


@dataclass
class PopulationConfig:
    """Configuration for population management."""
    population_size: int = 150
    fitness_criterion: str = 'max'
    fitness_threshold: Optional[float] = None
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
        genome = GenomeConfig(**d.get('genome', {}))
        fusion = FusionConfig(**d.get('fusion', {}))
        speciation = SpeciationConfig(**d.get('speciation', {}))
        reproduction = ReproductionConfig(**d.get('reproduction', {}))
        population = PopulationConfig(**d.get('population', {}))
        return cls(genome, fusion, speciation, reproduction, population)

    def to_dict(self) -> dict:
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
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x: float) -> float:
    return math.tanh(x)

def relu(x: float) -> float:
    return max(0.0, x)

def identity(x: float) -> float:
    return x

def softplus(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return math.log(1.0 + math.exp(x))

def gaussian(x: float) -> float:
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
    return sum(values)

def product_agg(values: List[float]) -> float:
    result = 1.0
    for v in values:
        result *= v
    return result

def max_agg(values: List[float]) -> float:
    return max(values) if values else 0.0

def min_agg(values: List[float]) -> float:
    return min(values) if values else 0.0

def mean_agg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


AGGREGATION_FUNCTIONS = {
    'sum': sum_agg,
    'product': product_agg,
    'max': max_agg,
    'min': min_agg,
    'mean': mean_agg,
}
