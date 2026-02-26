"""
Pole Balancing Example for SYNE.

This is the classic control benchmark used in the original SANE paper.
A cart-pole system where the goal is to balance a pole on a moving cart.

This benchmark is historically significant because SANE demonstrated
faster learning than Q-learning and AHCN on this task in 1996.
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syne import Config, Population
from syne.nn import FeedForwardNetwork
from syne.genome import Genome
from syne.visualization import print_genome_structure, plot_history_text


# Physics constants
GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = CART_MASS + POLE_MASS
POLE_LENGTH = 0.5  # Half the pole's length
POLE_MASS_LENGTH = POLE_MASS * POLE_LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # Time step

# Failure thresholds
X_THRESHOLD = 2.4  # Cart position limit
THETA_THRESHOLD = 12 * math.pi / 180  # Pole angle limit (12 degrees)


class CartPole:
    """Simple cart-pole simulation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.x = 0.0  # Cart position
        self.x_dot = 0.0  # Cart velocity
        self.theta = 0.0  # Pole angle
        self.theta_dot = 0.0  # Pole angular velocity
        self.steps = 0

    def step(self, action: float) -> bool:
        """
        Take a step in the simulation.

        Args:
            action: Force to apply (-1 to 1, scaled to FORCE_MAG)

        Returns:
            True if episode continues, False if failed
        """
        force = action * FORCE_MAG

        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        # Physics equations
        temp = (force + POLE_MASS_LENGTH * self.theta_dot ** 2 * sin_theta) / TOTAL_MASS
        theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
            POLE_LENGTH * (4.0/3.0 - POLE_MASS * cos_theta ** 2 / TOTAL_MASS)
        )
        x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

        # Update state
        self.x += TAU * self.x_dot
        self.x_dot += TAU * x_acc
        self.theta += TAU * self.theta_dot
        self.theta_dot += TAU * theta_acc
        self.steps += 1

        # Check failure
        failed = (
            self.x < -X_THRESHOLD or
            self.x > X_THRESHOLD or
            self.theta < -THETA_THRESHOLD or
            self.theta > THETA_THRESHOLD
        )

        return not failed

    def get_state(self) -> list:
        """Get normalized state vector."""
        return [
            self.x / X_THRESHOLD,
            self.x_dot / 2.0,  # Normalize velocity
            self.theta / THETA_THRESHOLD,
            self.theta_dot / 2.0,
        ]


def pole_balancing_fitness(genome: Genome, max_steps: int = 500) -> float:
    """
    Evaluate genome fitness on pole balancing.

    Fitness is the number of steps the pole was balanced.
    """
    try:
        network = FeedForwardNetwork.create(genome)
    except Exception:
        return 0.0

    env = CartPole()
    total_steps = 0

    # Run multiple trials with different initial conditions
    num_trials = 3
    for trial in range(num_trials):
        env.reset()

        # Slight perturbation for each trial
        env.theta = 0.01 * (trial - 1)

        for _ in range(max_steps):
            state = env.get_state()
            output = network.activate(state)

            # Convert network output to action (-1 to 1)
            action = output[0] * 2.0 - 1.0
            action = max(-1.0, min(1.0, action))

            if not env.step(action):
                break

            total_steps += 1

    return total_steps / num_trials


def run_pole_balancing_experiment():
    """Run the pole balancing experiment."""
    print("="*60)
    print("SYNE: Symbiogenetic Neuro-Evolution")
    print("Pole Balancing Benchmark")
    print("="*60)
    print()

    # Create configuration
    config = Config()
    config.genome.num_inputs = 4  # x, x_dot, theta, theta_dot
    config.genome.num_outputs = 1  # Force direction
    config.genome.activation_default = 'tanh'
    config.genome.initial_connectivity = 'full'

    # Symbiogenesis settings - higher fusion for this task
    config.fusion.fusion_prob = 0.4
    config.fusion.inter_network_connectivity = 'moderate'
    config.fusion.max_genome_nodes = 30

    # Population settings
    config.population.population_size = 150
    config.population.fitness_threshold = 490  # Near-perfect balancing

    # Speciation
    config.speciation.compatibility_threshold = 3.0
    config.speciation.stagnation_limit = 15

    # Create and run population
    pop = Population(config)
    pop.initialize()

    print(f"Initial population: {len(pop.population)} genomes")
    print(f"Running evolution...")
    print()

    # Run evolution
    best, stats = pop.run(
        fitness_function=pole_balancing_fitness,
        max_generations=100,
        fitness_threshold=490,
    )

    # Print results
    print()
    print(plot_history_text(pop.history))

    # Show best network
    print_genome_structure(best)

    # Test the best network
    print("\nBest Network Evaluation (5 trials):")
    print("-"*40)
    network = FeedForwardNetwork.create(best)

    for trial in range(5):
        env = CartPole()
        env.theta = 0.02 * (trial - 2)  # Different starting angles

        for _ in range(500):
            state = env.get_state()
            output = network.activate(state)
            action = output[0] * 2.0 - 1.0
            action = max(-1.0, min(1.0, action))
            if not env.step(action):
                break

        status = "✓ Balanced" if env.steps >= 500 else f"✗ Failed at step {env.steps}"
        print(f"  Trial {trial + 1}: {status}")

    return best, pop.history


if __name__ == "__main__":
    run_pole_balancing_experiment()
