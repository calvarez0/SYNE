"""
SYNE v5: Symbiogenetic Neuro-Evolution

A clean, well-tested implementation of neuroevolution via genome fusion.
This version fixes tracking bugs from v2 and includes rigorous testing.

Key fixes in v5:
1. Proper fusion tracking (accumulated across generations)
2. Proper species minimum tracking
3. Better statistical analysis
4. Verified claims against actual data
"""

from syne_v5.config import Config, GenomeConfig, FusionConfig, SpeciationConfig
from syne_v5.genome import Genome, NodeGene, ConnectionGene, crossover, fuse
from syne_v5.population import Population, GenerationStats, run_syne
from syne_v5.species import Species, SpeciesSet
from syne_v5.reproduction import SymbioticReproduction
from syne_v5.nn import FeedForwardNetwork, RecurrentNetwork
from syne_v5.innovation import get_innovation_tracker, reset_innovation_tracker

__version__ = "5.0.0"
__all__ = [
    'Config', 'GenomeConfig', 'FusionConfig', 'SpeciationConfig',
    'Genome', 'NodeGene', 'ConnectionGene', 'crossover', 'fuse',
    'Population', 'GenerationStats', 'run_syne',
    'Species', 'SpeciesSet',
    'SymbioticReproduction',
    'FeedForwardNetwork', 'RecurrentNetwork',
    'get_innovation_tracker', 'reset_innovation_tracker',
]
