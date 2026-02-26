"""
Pole Balancing Benchmark: SYNE vs NEAT

This is the classic control task used in the original SANE paper (1996).
Historical significance: SANE was 9-16x faster than AHCN and 2x faster than Q-learning.
"""

import sys
import os
import math
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syne import Config, Population
from syne.nn import FeedForwardNetwork
from syne.genome import Genome
from syne.innovation import reset_innovation_tracker

import neat

# Physics constants
GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = CART_MASS + POLE_MASS
POLE_LENGTH = 0.5
POLE_MASS_LENGTH = POLE_MASS * POLE_LENGTH
FORCE_MAG = 10.0
TAU = 0.02
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12 * math.pi / 180


class CartPole:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        self.steps = 0

    def step(self, action: float) -> bool:
        force = action * FORCE_MAG
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        temp = (force + POLE_MASS_LENGTH * self.theta_dot ** 2 * sin_theta) / TOTAL_MASS
        theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
            POLE_LENGTH * (4.0/3.0 - POLE_MASS * cos_theta ** 2 / TOTAL_MASS)
        )
        x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

        self.x += TAU * self.x_dot
        self.x_dot += TAU * x_acc
        self.theta += TAU * self.theta_dot
        self.theta_dot += TAU * theta_acc
        self.steps += 1

        failed = (
            self.x < -X_THRESHOLD or
            self.x > X_THRESHOLD or
            self.theta < -THETA_THRESHOLD or
            self.theta > THETA_THRESHOLD
        )
        return not failed

    def get_state(self) -> list:
        return [
            self.x / X_THRESHOLD,
            self.x_dot / 2.0,
            self.theta / THETA_THRESHOLD,
            self.theta_dot / 2.0,
        ]


def pole_fitness_syne(genome: Genome, max_steps: int = 500) -> float:
    try:
        network = FeedForwardNetwork.create(genome)
    except Exception:
        return 0.0

    env = CartPole()
    total_steps = 0
    num_trials = 3

    for trial in range(num_trials):
        env.reset()
        env.theta = 0.01 * (trial - 1)

        for _ in range(max_steps):
            state = env.get_state()
            output = network.activate(state)
            action = output[0] * 2.0 - 1.0
            action = max(-1.0, min(1.0, action))

            if not env.step(action):
                break
            total_steps += 1

    return total_steps / num_trials


def pole_fitness_neat(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = CartPole()
        total_steps = 0
        num_trials = 3

        for trial in range(num_trials):
            env.reset()
            env.theta = 0.01 * (trial - 1)

            for _ in range(500):
                state = env.get_state()
                output = net.activate(state)
                action = output[0] * 2.0 - 1.0
                action = max(-1.0, min(1.0, action))

                if not env.step(action):
                    break
                total_steps += 1

        genome.fitness = total_steps / num_trials


@dataclass
class RunResult:
    algorithm: str
    run_id: int
    solved: bool
    generations_to_solve: Optional[int]
    final_fitness: float
    final_nodes: int
    final_connections: int
    total_time: float


def run_syne(run_id: int, max_generations: int = 100) -> RunResult:
    reset_innovation_tracker()
    Genome.reset_id_counter()

    config = Config()
    config.genome.num_inputs = 4
    config.genome.num_outputs = 1
    config.genome.activation_default = 'tanh'
    config.genome.initial_connectivity = 'full'

    config.fusion.fusion_prob = 0.4
    config.fusion.inter_network_connectivity = 'moderate'
    config.fusion.max_genome_nodes = 30
    config.fusion.fusion_fitness_threshold = 0.3

    config.population.population_size = 150
    config.population.fitness_threshold = 490

    config.speciation.compatibility_threshold = 1.0
    config.speciation.stagnation_limit = 15
    config.speciation.species_elitism = 3

    pop = Population(config)
    pop.initialize()

    solved = False
    generations_to_solve = None
    best_fitness = 0
    best_genome = None
    start_time = time.time()

    for gen in range(max_generations):
        for genome in pop.population.values():
            genome.fitness = pole_fitness_syne(genome)

        current_best = max(pop.population.values(), key=lambda g: g.fitness or 0)
        if current_best.fitness and current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_genome = current_best

        if best_fitness >= 490:
            solved = True
            generations_to_solve = gen
            break

        pop._update_best()
        pop.species_set.speciate(pop.population, gen)
        pop.species_set.update_fitness_history(gen)
        pop.species_set.remove_stagnant_species(gen)
        pop.reproduction.reset_stats()
        pop.population = pop.reproduction.reproduce(
            pop.species_set, config.population.population_size, gen
        )

    total_time = time.time() - start_time

    return RunResult(
        algorithm="SYNE",
        run_id=run_id,
        solved=solved,
        generations_to_solve=generations_to_solve,
        final_fitness=best_fitness,
        final_nodes=len(best_genome.nodes) if best_genome else 0,
        final_connections=len(best_genome.connections) if best_genome else 0,
        total_time=total_time,
    )


def run_neat(run_id: int, config_path: str, max_generations: int = 100) -> RunResult:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    solved = False
    generations_to_solve = None
    best_fitness = 0
    best_genome = None
    start_time = time.time()

    for gen in range(max_generations):
        pole_fitness_neat(list(pop.population.items()), config)

        current_best = max(pop.population.values(), key=lambda g: g.fitness or 0)
        if current_best.fitness and current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_genome = current_best

        if best_fitness >= 490:
            solved = True
            generations_to_solve = gen
            break

        pop.species.speciate(config, pop.population, gen)
        pop.population = pop.reproduction.reproduce(
            config, pop.species, config.pop_size, gen
        )

    total_time = time.time() - start_time

    return RunResult(
        algorithm="NEAT",
        run_id=run_id,
        solved=solved,
        generations_to_solve=generations_to_solve,
        final_fitness=best_fitness,
        final_nodes=len(best_genome.nodes) if best_genome else 0,
        final_connections=len([c for c in best_genome.connections.values() if c.enabled]) if best_genome else 0,
        total_time=total_time,
    )


def main():
    print("="*70)
    print("Pole Balancing Benchmark: SYNE vs NEAT")
    print("="*70)

    num_runs = 5
    max_generations = 100

    config_path = os.path.join(os.path.dirname(__file__), "pole_config.txt")

    syne_results = []
    neat_results = []

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")

        print("Running SYNE...", end=" ", flush=True)
        syne_result = run_syne(i, max_generations)
        status = f"Solved in {syne_result.generations_to_solve} gens" if syne_result.solved else "Not solved"
        print(f"{status} (fitness: {syne_result.final_fitness:.1f})")
        syne_results.append(syne_result)

        print("Running NEAT...", end=" ", flush=True)
        neat_result = run_neat(i, config_path, max_generations)
        status = f"Solved in {neat_result.generations_to_solve} gens" if neat_result.solved else "Not solved"
        print(f"{status} (fitness: {neat_result.final_fitness:.1f})")
        neat_results.append(neat_result)

    # Summary
    print("\n" + "="*70)
    print("POLE BALANCING RESULTS")
    print("="*70)

    syne_solved = [r for r in syne_results if r.solved]
    neat_solved = [r for r in neat_results if r.solved]

    print(f"\n{'Metric':<35} {'SYNE':>15} {'NEAT':>15}")
    print("-"*70)
    print(f"{'Success Rate':<35} {len(syne_solved)/num_runs*100:>14.1f}% {len(neat_solved)/num_runs*100:>14.1f}%")

    if syne_solved:
        syne_gens = np.mean([r.generations_to_solve for r in syne_solved])
    else:
        syne_gens = float('inf')

    if neat_solved:
        neat_gens = np.mean([r.generations_to_solve for r in neat_solved])
    else:
        neat_gens = float('inf')

    print(f"{'Mean Generations to Solve':<35} {syne_gens:>15.1f} {neat_gens:>15.1f}")
    print(f"{'Mean Final Fitness':<35} {np.mean([r.final_fitness for r in syne_results]):>15.1f} {np.mean([r.final_fitness for r in neat_results]):>15.1f}")
    print(f"{'Mean Final Complexity':<35} {np.mean([r.final_nodes + r.final_connections for r in syne_results]):>15.1f} {np.mean([r.final_nodes + r.final_connections for r in neat_results]):>15.1f}")

    # Save results
    results = {
        'syne': [asdict(r) for r in syne_results],
        'neat': [asdict(r) for r in neat_results],
    }

    output_path = os.path.join(os.path.dirname(__file__), 'data', 'pole_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
