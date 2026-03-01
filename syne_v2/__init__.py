"""
SYNE: Symbiogenetic Neuro-Evolution

A neuroevolutionary algorithm based on symbiogenesis rather than mutation.
Inspired by Agüera y Arcas's computational symbiogenesis work and building
on NEAT's genome representation.

Unlike NEAT which complexifies through structural mutations, SYNE achieves
complexity growth purely through genome fusion - the merging of complete
neural network genomes into larger, more complex chimeric networks.
"""

from syne_v2.genome import Genome, NodeGene, ConnectionGene
from syne_v2.population import Population
from syne_v2.species import Species, SpeciesSet
from syne_v2.reproduction import SymbioticReproduction
from syne_v2.config import Config
from syne_v2.nn import FeedForwardNetwork
from syne_v2.innovation import InnovationTracker

__version__ = "0.1.0"
__author__ = "SYNE Contributors"

__all__ = [
    "Genome",
    "NodeGene",
    "ConnectionGene",
    "Population",
    "Species",
    "SpeciesSet",
    "SymbioticReproduction",
    "Config",
    "FeedForwardNetwork",
    "InnovationTracker",
]
