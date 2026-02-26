# SYNE: Symbiogenetic Neuro-Evolution

**SYNE** is a neuroevolutionary algorithm that achieves complexity growth through *symbiogenesis* rather than mutation. Inspired by [Blaise Agüera y Arcas's computational symbiogenesis research](https://arxiv.org/abs/2406.19108) and building on NEAT's genome representation, SYNE demonstrates that evolutionary novelty can emerge purely through the fusion of complete genomes.

## Key Concept

In traditional neuroevolution (like NEAT), networks grow more complex through **mutation**: adding nodes, adding connections, perturbing weights. SYNE takes a radically different approach inspired by biological symbiogenesis — the merging of distinct organisms into a new, more complex entity (as in the origin of mitochondria and chloroplasts).

In SYNE:
- **No mutation** — networks never undergo random structural or weight changes
- **Genome fusion** — two complete neural networks merge into a larger chimeric network
- **Symbiotic integration** — the merged network contains both original sub-networks, connected by new inter-network wiring

This mirrors Agüera y Arcas's finding that "symbiogenesis, not mutation, is the primary engine of evolutionary novelty."

## Installation

```bash
pip install syne
```

Or from source:
```bash
git clone https://github.com/syne-evolution/syne
cd syne
pip install -e .
```

## Quick Start

```python
from syne import Config, Population
from syne.nn import FeedForwardNetwork

# Define fitness function
def my_fitness(genome):
    network = FeedForwardNetwork.create(genome)
    # Evaluate network and return fitness score
    return fitness_score

# Configure SYNE
config = Config()
config.genome.num_inputs = 4
config.genome.num_outputs = 1
config.fusion.fusion_prob = 0.3  # 30% fusion, 70% crossover

# Run evolution
pop = Population(config)
pop.initialize()
best_genome, stats = pop.run(my_fitness, max_generations=100)

# Use the evolved network
network = FeedForwardNetwork.create(best_genome)
output = network.activate([1.0, 2.0, 3.0, 4.0])
```

## How It Works

### Genome Representation

Like NEAT, SYNE uses a genome that encodes:
- **Node genes**: Neurons with bias, activation function, and response
- **Connection genes**: Weighted connections with innovation numbers

### Evolution Process

1. **Initialize** — Create population of simple networks
2. **Evaluate** — Compute fitness for each genome
3. **Speciate** — Group similar genomes into species
4. **Reproduce** — Create offspring via:
   - **Fusion** (symbiogenesis): Merge two genomes from different species into a larger chimera
   - **Crossover**: Recombine genes from two genomes in the same species
5. **Repeat** — No mutation step!

### Genome Fusion

When two genomes fuse:
1. All nodes from both genomes are preserved
2. All connections from both genomes are preserved
3. New inter-network connections are created linking the two sub-networks
4. The child genome is placed in a new species (it's structurally unique)

```
Genome A (3 nodes)  +  Genome B (4 nodes)  →  Genome C (7 nodes + inter-connections)
     ↓                      ↓                         ↓
  [I]─[H]─[O]           [I]─[H]─[O]          [I]─[H]═══╗
                            │                    ║    ║
                           [H]               [I]─[H]─[O]
                                                 │
                                                [H]
                                             (chimera)
```

## Configuration

```python
config = Config()

# Genome structure
config.genome.num_inputs = 2
config.genome.num_outputs = 1
config.genome.activation_default = 'sigmoid'

# Symbiogenesis settings
config.fusion.fusion_prob = 0.3          # Probability of fusion vs crossover
config.fusion.inter_network_connectivity = 'sparse'  # sparse, moderate, dense
config.fusion.max_genome_nodes = 100     # Max complexity before fusion disabled

# Population
config.population.population_size = 150
config.population.fitness_threshold = 3.9

# Speciation
config.speciation.compatibility_threshold = 3.0
config.speciation.stagnation_limit = 15
```

## Examples

### XOR Problem
```bash
python examples/xor.py
```

### Pole Balancing
```bash
python examples/pole_balancing.py
```

### Compare with Mutation-based Evolution
```bash
python examples/compare_with_mutation.py
```

## Theoretical Background

SYNE is motivated by two key insights:

### 1. Computational Symbiogenesis (Agüera y Arcas et al., 2024)

In ["Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction"](https://arxiv.org/abs/2406.19108), Agüera y Arcas et al. showed that:
- Self-replicating programs emerge from random code through symbiogenesis
- Complexity increases primarily through program merging, not mutation
- Blocking symbiogenetic ancestry prevents the emergence of complex replicators

### 2. SANE and Historical Context (Moriarty & Miikkulainen, 1996)

[SANE (Symbiotic Adaptive Neuro-Evolution)](https://link.springer.com/article/10.1007/BF00114722) introduced symbiotic cooperation in neuroevolution — evolving neurons that must cooperate to form networks. However, SANE didn't implement true symbiogenesis (permanent fusion of complete organisms).

SYNE closes this gap by implementing actual genome fusion at the whole-network level, following the biological model of symbiogenesis where two complete organisms merge into a new, more complex entity.

## Research Applications

SYNE is designed for investigating:

- **Open-ended evolution**: Does fusion enable unbounded complexity growth?
- **Major evolutionary transitions**: Can fusion produce eukaryote-like emergent individuals?
- **Complexity without mutation**: Validating Agüera y Arcas's theoretical claims
- **Modular network evolution**: Do fused sub-networks retain functional identity?

## Citation

If you use SYNE in your research, please cite:

```bibtex
@software{syne2025,
  title = {SYNE: Symbiogenetic Neuro-Evolution},
  year = {2025},
  url = {https://github.com/syne-evolution/syne}
}
```

## License

MIT License

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
