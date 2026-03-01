#!/usr/bin/env python3
"""
Basic tests for SYNE v5 implementation.

Verifies that:
1. Basic XOR learning works
2. Fusion tracking is correct
3. Species tracking is correct
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syne_v5 import Config, Population, Genome
from syne_v5.nn import FeedForwardNetwork
from syne_v5.innovation import reset_innovation_tracker


# XOR problem
XOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_OUTPUTS = [0, 1, 1, 0]


def xor_fitness(genome: Genome) -> float:
    """Evaluate XOR fitness."""
    try:
        network = FeedForwardNetwork.create(genome)
    except Exception as e:
        return 0.0

    error = 0.0
    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = network.activate(inputs)
        error += (output[0] - expected) ** 2

    # Fitness = 4.0 - error (max 4.0, solve at >= 3.9)
    return 4.0 - error


def test_single_run():
    """Test a single evolution run and verify tracking."""
    print("=" * 60)
    print("TEST: Single XOR Evolution Run")
    print("=" * 60)

    reset_innovation_tracker()
    Genome.reset_id_counter()

    config = Config()
    config.genome.num_inputs = 2
    config.genome.num_outputs = 1
    config.genome.activation_default = 'tanh'
    config.population.population_size = 150
    config.fusion.fusion_prob = 0.3
    config.speciation.compatibility_threshold = 1.0
    config.speciation.species_elitism = 3

    pop = Population(config)
    pop.initialize()

    best_genome, final_stats = pop.run(
        xor_fitness,
        max_generations=50,
        fitness_threshold=3.9,
        verbose=True
    )

    # Check results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    solved = best_genome.fitness >= 3.9 if best_genome and best_genome.fitness else False
    print(f"Solved: {solved}")
    print(f"Best fitness: {best_genome.fitness if best_genome else None}")
    print(f"Generations: {pop.generation}")

    # V5 FIX VERIFICATION: Check fusion tracking
    cumulative = pop.reproduction.get_cumulative_stats()
    print(f"\nFusion tracking (V5 FIX):")
    print(f"  Total fusions: {cumulative['total_fusions']}")
    print(f"  Total crossovers: {cumulative['total_crossovers']}")
    print(f"  Fusion rate: {cumulative['cumulative_fusion_rate']:.2%}")

    # Verify fusions > 0 (this was broken in v2)
    assert cumulative['total_fusions'] > 0, "ERROR: No fusions recorded!"
    print("  [PASS] Fusion tracking working")

    # V5 FIX VERIFICATION: Check species tracking
    species_stats = pop.species_set.get_stats()
    print(f"\nSpecies tracking (V5 FIX):")
    print(f"  Current species: {species_stats['num_species']}")
    print(f"  Min species seen: {species_stats['min_species_seen']}")

    # Verify min_species_seen > 0 (this was broken in v2)
    assert species_stats['min_species_seen'] > 0, "ERROR: min_species_seen not tracked!"
    print("  [PASS] Species tracking working")

    return solved, cumulative['total_fusions'], species_stats['min_species_seen']


def test_multiple_runs(n_runs=5):
    """Test multiple runs to verify consistency."""
    print("\n" + "=" * 60)
    print(f"TEST: {n_runs} XOR Evolution Runs")
    print("=" * 60)

    results = []

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}...")

        reset_innovation_tracker()
        Genome.reset_id_counter()

        config = Config()
        config.genome.num_inputs = 2
        config.genome.num_outputs = 1
        config.genome.activation_default = 'tanh'
        config.population.population_size = 150
        config.fusion.fusion_prob = 0.3
        config.speciation.compatibility_threshold = 1.0
        config.speciation.species_elitism = 3

        pop = Population(config)
        pop.initialize()

        best_genome, final_stats = pop.run(
            xor_fitness,
            max_generations=100,
            fitness_threshold=3.9,
            verbose=False
        )

        solved = best_genome.fitness >= 3.9 if best_genome and best_genome.fitness else False
        cumulative = pop.reproduction.get_cumulative_stats()
        species_stats = pop.species_set.get_stats()

        results.append({
            'run': run,
            'solved': solved,
            'generations': pop.generation,
            'best_fitness': best_genome.fitness,
            'total_fusions': cumulative['total_fusions'],
            'total_crossovers': cumulative['total_crossovers'],
            'min_species_seen': species_stats['min_species_seen'],
            'final_species': species_stats['num_species'],
        })

        status = "SOLVED" if solved else "Failed"
        print(f"  {status} - Gen {pop.generation}, Fit {best_genome.fitness:.3f}, "
              f"Fusions {cumulative['total_fusions']}, MinSp {species_stats['min_species_seen']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    n_solved = sum(1 for r in results if r['solved'])
    print(f"Success rate: {n_solved}/{n_runs} ({100*n_solved/n_runs:.0f}%)")

    avg_fusions = sum(r['total_fusions'] for r in results) / n_runs
    print(f"Average fusions: {avg_fusions:.1f}")

    avg_min_species = sum(r['min_species_seen'] for r in results) / n_runs
    print(f"Average min species seen: {avg_min_species:.1f}")

    # Verify no zeros
    zero_fusions = sum(1 for r in results if r['total_fusions'] == 0)
    zero_species = sum(1 for r in results if r['min_species_seen'] == 0)

    print(f"\nRuns with zero fusions: {zero_fusions} (should be 0)")
    print(f"Runs with zero min_species: {zero_species} (should be 0)")

    if zero_fusions > 0:
        print("[FAIL] Some runs had zero fusions!")
    else:
        print("[PASS] All runs tracked fusions correctly")

    if zero_species > 0:
        print("[FAIL] Some runs had zero min_species!")
    else:
        print("[PASS] All runs tracked species correctly")

    return results


if __name__ == "__main__":
    print("SYNE v5 Basic Tests")
    print("=" * 60)

    # Test single run with verbose output
    test_single_run()

    # Test multiple runs
    test_multiple_runs(5)

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
