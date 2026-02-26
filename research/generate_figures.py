"""
Generate figures for SYNE vs NEAT comparison.

Creates publication-quality visualizations of:
1. Fitness curves over generations
2. Complexity growth dynamics
3. Species diversity
4. Comparison summary
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150


def load_results(data_dir: str):
    """Load experiment results from JSON files."""
    with open(os.path.join(data_dir, 'syne_results.json'), 'r') as f:
        syne = json.load(f)
    with open(os.path.join(data_dir, 'neat_results.json'), 'r') as f:
        neat = json.load(f)
    return syne, neat


def plot_fitness_curves(syne, neat, output_dir):
    """Plot fitness over generations for both algorithms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Colors
    syne_color = '#2ecc71'  # Green
    neat_color = '#3498db'  # Blue

    # Plot individual runs and mean for SYNE
    max_gen_syne = 0
    all_syne_fitness = []
    for run in syne['runs']:
        gens = [h['generation'] for h in run['history']]
        fitness = [h['best_fitness'] for h in run['history']]
        ax1.plot(gens, fitness, alpha=0.3, color=syne_color, linewidth=1)
        max_gen_syne = max(max_gen_syne, max(gens))
        all_syne_fitness.append(fitness)

    # Plot individual runs and mean for NEAT
    max_gen_neat = 0
    all_neat_fitness = []
    for run in neat['runs']:
        gens = [h['generation'] for h in run['history']]
        fitness = [h['best_fitness'] for h in run['history']]
        ax1.plot(gens, fitness, alpha=0.3, color=neat_color, linewidth=1)
        max_gen_neat = max(max_gen_neat, max(gens))
        all_neat_fitness.append(fitness)

    # Calculate and plot means
    # Pad shorter runs
    max_len = max(max(len(f) for f in all_syne_fitness), max(len(f) for f in all_neat_fitness))

    def pad_and_mean(fitness_list, max_len):
        padded = []
        for f in fitness_list:
            if len(f) < max_len:
                padded.append(f + [f[-1]] * (max_len - len(f)))
            else:
                padded.append(f[:max_len])
        return np.mean(padded, axis=0), np.std(padded, axis=0)

    syne_mean, syne_std = pad_and_mean(all_syne_fitness, max_len)
    neat_mean, neat_std = pad_and_mean(all_neat_fitness, max_len)

    gens = list(range(max_len))
    ax1.plot(gens, syne_mean, color=syne_color, linewidth=2.5, label='SYNE (mean)')
    ax1.plot(gens, neat_mean, color=neat_color, linewidth=2.5, label='NEAT (mean)')

    ax1.axhline(y=3.9, color='red', linestyle='--', alpha=0.7, label='Solution threshold')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Fitness Over Generations')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 150)
    ax1.set_ylim(2.9, 4.05)

    # Plot early generations zoom
    ax2.plot(gens[:50], syne_mean[:50], color=syne_color, linewidth=2.5, label='SYNE')
    ax2.fill_between(gens[:50], syne_mean[:50] - syne_std[:50], syne_mean[:50] + syne_std[:50],
                     color=syne_color, alpha=0.2)
    ax2.plot(gens[:50], neat_mean[:50], color=neat_color, linewidth=2.5, label='NEAT')
    ax2.fill_between(gens[:50], neat_mean[:50] - neat_std[:50], neat_mean[:50] + neat_std[:50],
                     color=neat_color, alpha=0.2)
    ax2.axhline(y=3.9, color='red', linestyle='--', alpha=0.7, label='Threshold')

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title('Early Generations (Zoomed)')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_curves.png'), bbox_inches='tight')
    plt.close()
    print("Saved: fitness_curves.png")


def plot_complexity_growth(syne, neat, output_dir):
    """Plot complexity (nodes + connections) over generations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    syne_color = '#2ecc71'
    neat_color = '#3498db'

    # Plot 1: Best genome complexity
    ax1 = axes[0]
    for run in syne['runs']:
        gens = [h['generation'] for h in run['history']]
        complexity = [h['best_nodes'] + h['best_connections'] for h in run['history']]
        ax1.plot(gens, complexity, alpha=0.4, color=syne_color, linewidth=1)

    for run in neat['runs']:
        gens = [h['generation'] for h in run['history']]
        complexity = [h['best_nodes'] + h['best_connections'] for h in run['history']]
        ax1.plot(gens, complexity, alpha=0.4, color=neat_color, linewidth=1)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Complexity (Nodes + Connections)')
    ax1.set_title('Best Genome Complexity')

    syne_patch = mpatches.Patch(color=syne_color, label='SYNE')
    neat_patch = mpatches.Patch(color=neat_color, label='NEAT')
    ax1.legend(handles=[syne_patch, neat_patch])

    # Plot 2: Mean population complexity
    ax2 = axes[1]
    for run in syne['runs']:
        gens = [h['generation'] for h in run['history']]
        complexity = [h['mean_nodes'] + h['mean_connections'] for h in run['history']]
        ax2.plot(gens, complexity, alpha=0.4, color=syne_color, linewidth=1)

    for run in neat['runs']:
        gens = [h['generation'] for h in run['history']]
        complexity = [h['mean_nodes'] + h['mean_connections'] for h in run['history']]
        ax2.plot(gens, complexity, alpha=0.4, color=neat_color, linewidth=1)

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Mean Complexity')
    ax2.set_title('Mean Population Complexity')
    ax2.legend(handles=[syne_patch, neat_patch])

    # Plot 3: Final complexity distribution
    ax3 = axes[2]
    syne_final = [r['final_nodes'] + r['final_connections'] for r in syne['runs']]
    neat_final = [r['final_nodes'] + r['final_connections'] for r in neat['runs']]

    positions = [1, 2]
    bp = ax3.boxplot([syne_final, neat_final], positions=positions, widths=0.6,
                     patch_artist=True)

    bp['boxes'][0].set_facecolor(syne_color)
    bp['boxes'][1].set_facecolor(neat_color)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(['SYNE', 'NEAT'])
    ax3.set_ylabel('Final Complexity')
    ax3.set_title('Final Genome Complexity Distribution')

    # Add individual points
    ax3.scatter([1]*len(syne_final), syne_final, color='darkgreen', alpha=0.6, s=30)
    ax3.scatter([2]*len(neat_final), neat_final, color='darkblue', alpha=0.6, s=30)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_growth.png'), bbox_inches='tight')
    plt.close()
    print("Saved: complexity_growth.png")


def plot_species_diversity(syne, neat, output_dir):
    """Plot species count over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    syne_color = '#2ecc71'
    neat_color = '#3498db'

    # Collect species data
    syne_species = []
    neat_species = []

    for run in syne['runs']:
        species = [h['num_species'] for h in run['history']]
        syne_species.append(species)
        ax.plot(range(len(species)), species, alpha=0.3, color=syne_color, linewidth=1)

    for run in neat['runs']:
        species = [h['num_species'] for h in run['history']]
        neat_species.append(species)
        ax.plot(range(len(species)), species, alpha=0.3, color=neat_color, linewidth=1)

    # Calculate means
    max_len = max(max(len(s) for s in syne_species), max(len(s) for s in neat_species))

    def pad_and_mean(data, max_len):
        padded = []
        for d in data:
            if len(d) < max_len:
                padded.append(d + [d[-1]] * (max_len - len(d)))
            else:
                padded.append(d[:max_len])
        return np.mean(padded, axis=0)

    syne_mean = pad_and_mean(syne_species, max_len)
    neat_mean = pad_and_mean(neat_species, max_len)

    ax.plot(range(len(syne_mean)), syne_mean, color=syne_color, linewidth=2.5, label='SYNE (mean)')
    ax.plot(range(len(neat_mean)), neat_mean, color=neat_color, linewidth=2.5, label='NEAT (mean)')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Number of Species')
    ax.set_title('Species Diversity Over Time')
    ax.legend()
    ax.set_xlim(0, 150)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'species_diversity.png'), bbox_inches='tight')
    plt.close()
    print("Saved: species_diversity.png")


def plot_summary_comparison(syne, neat, output_dir):
    """Create summary comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    syne_color = '#2ecc71'
    neat_color = '#3498db'

    # 1. Success rate
    ax = axes[0, 0]
    rates = [syne['success_rate'] * 100, neat['success_rate'] * 100]
    bars = ax.bar(['SYNE', 'NEAT'], rates, color=[syne_color, neat_color])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate')
    ax.set_ylim(0, 110)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Generations to solve
    ax = axes[0, 1]
    syne_gens = [r['generations_to_solve'] for r in syne['runs'] if r['solved']]
    neat_gens = [r['generations_to_solve'] for r in neat['runs'] if r['solved']]

    if syne_gens and neat_gens:
        bp = ax.boxplot([syne_gens, neat_gens], patch_artist=True)
        bp['boxes'][0].set_facecolor(syne_color)
        bp['boxes'][1].set_facecolor(neat_color)
        ax.set_xticklabels(['SYNE', 'NEAT'])
        ax.set_ylabel('Generations to Solve')
        ax.set_title('Speed to Solution\n(lower is better)')

    # 3. Final fitness
    ax = axes[0, 2]
    syne_fit = [r['final_fitness'] for r in syne['runs']]
    neat_fit = [r['final_fitness'] for r in neat['runs']]
    bp = ax.boxplot([syne_fit, neat_fit], patch_artist=True)
    bp['boxes'][0].set_facecolor(syne_color)
    bp['boxes'][1].set_facecolor(neat_color)
    ax.set_xticklabels(['SYNE', 'NEAT'])
    ax.set_ylabel('Final Fitness')
    ax.set_title('Final Fitness Distribution')
    ax.axhline(y=3.9, color='red', linestyle='--', alpha=0.7)

    # 4. Final complexity
    ax = axes[1, 0]
    syne_comp = [r['final_nodes'] + r['final_connections'] for r in syne['runs']]
    neat_comp = [r['final_nodes'] + r['final_connections'] for r in neat['runs']]
    bars = ax.bar(['SYNE', 'NEAT'], [np.mean(syne_comp), np.mean(neat_comp)],
                  yerr=[np.std(syne_comp), np.std(neat_comp)],
                  color=[syne_color, neat_color], capsize=5)
    ax.set_ylabel('Mean Final Complexity')
    ax.set_title('Final Network Complexity\n(nodes + connections)')

    # 5. Time per generation
    ax = axes[1, 1]
    times = [syne['mean_time_per_generation'] * 1000, neat['mean_time_per_generation'] * 1000]
    bars = ax.bar(['SYNE', 'NEAT'], times, color=[syne_color, neat_color])
    ax.set_ylabel('Time per Generation (ms)')
    ax.set_title('Computational Cost')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}ms', ha='center', va='bottom')

    # 6. Efficiency: generations * complexity
    ax = axes[1, 2]
    # Calculate "efficiency" as inverse of (generations to solve * final complexity)
    syne_eff = []
    neat_eff = []
    for run in syne['runs']:
        if run['solved']:
            eff = run['generations_to_solve']
            syne_eff.append(eff)
    for run in neat['runs']:
        if run['solved']:
            eff = run['generations_to_solve']
            neat_eff.append(eff)

    if syne_eff and neat_eff:
        bp = ax.boxplot([syne_eff, neat_eff], patch_artist=True)
        bp['boxes'][0].set_facecolor(syne_color)
        bp['boxes'][1].set_facecolor(neat_color)
        ax.set_xticklabels(['SYNE', 'NEAT'])
        ax.set_ylabel('Generations (when solved)')
        ax.set_title('Solution Efficiency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved: summary_comparison.png")


def plot_fusion_analysis(syne, output_dir):
    """Analyze fusion events in SYNE (SYNE-specific)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Cumulative fusions over time
    ax1 = axes[0]
    for run in syne['runs']:
        gens = [h['generation'] for h in run['history']]
        fusions = [h['fusion_count'] for h in run['history']]
        # Convert to cumulative if not already
        ax1.plot(gens, fusions, alpha=0.5, linewidth=1.5)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Cumulative Fusions')
    ax1.set_title('Fusion Events Over Time')

    # Plot 2: Fusion vs Crossover ratio
    ax2 = axes[1]
    fusion_rates = []
    for run in syne['runs']:
        total_fusions = run['history'][-1]['fusion_count'] if run['history'] else 0
        total_crossovers = run['history'][-1]['crossover_count'] if run['history'] else 0
        total = total_fusions + total_crossovers
        if total > 0:
            fusion_rates.append(total_fusions / total * 100)

    ax2.hist(fusion_rates, bins=10, color='#2ecc71', edgecolor='darkgreen', alpha=0.7)
    ax2.axvline(x=np.mean(fusion_rates), color='red', linestyle='--',
                label=f'Mean: {np.mean(fusion_rates):.1f}%')
    ax2.set_xlabel('Fusion Rate (%)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Fusion Rates')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_analysis.png'), bbox_inches='tight')
    plt.close()
    print("Saved: fusion_analysis.png")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading results...")
    syne, neat = load_results(data_dir)

    print("\nGenerating figures...")
    plot_fitness_curves(syne, neat, figures_dir)
    plot_complexity_growth(syne, neat, figures_dir)
    plot_species_diversity(syne, neat, figures_dir)
    plot_summary_comparison(syne, neat, figures_dir)
    plot_fusion_analysis(syne, figures_dir)

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
