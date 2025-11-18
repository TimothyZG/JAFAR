#!/usr/bin/env python3
"""
Plot ensemble evaluation results.

Usage:
    python evaluation/plot_ensemble_results.py --config config/eval_ensemble.yaml --results ensemble_results/ensemble_evaluation.csv
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
from collections import defaultdict

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_csv(filepath):
    """Load CSV file into a list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['mIoU_majvote'] = float(row['mIoU_majvote'])
            row['accuracy_majvote'] = float(row['accuracy_majvote'])
            row['mIoU_oracle'] = float(row['mIoU_oracle'])
            row['accuracy_oracle'] = float(row['accuracy_oracle'])
            data.append(row)
    return data


def parse_ensemble_name(ensemble_name):
    """Parse ensemble name to get member indices."""
    if ensemble_name.startswith('ensemble_'):
        indices_str = ensemble_name.replace('ensemble_', '')
        if indices_str:
            indices = [int(i) for i in indices_str.split('_')]
            return indices
    return []


def get_ensemble_size(ensemble_name):
    """Get size of ensemble."""
    indices = parse_ensemble_name(ensemble_name)
    return len(indices)


def greedy_ensemble_selection(data, backbone_names, metric='mIoU_majvote'):
    """Greedily select best ensemble by adding one member at a time."""
    n_models = len(backbone_names)
    selected = []
    scores = []
    ensembles = []

    # Create lookup dict
    ensemble_dict = {row['ensemble']: row for row in data}

    # Start with best single model
    single_models = [(row['ensemble'], row[metric]) for row in data
                     if get_ensemble_size(row['ensemble']) == 1]
    best_single = max(single_models, key=lambda x: x[1])

    indices = parse_ensemble_name(best_single[0])
    selected.append(indices[0])
    scores.append(best_single[1])
    ensembles.append(best_single[0])

    # Greedily add remaining models
    remaining = [i for i in range(n_models) if i not in selected]

    while remaining:
        best_score = -1
        best_model = None
        best_ensemble_name = None

        for candidate in remaining:
            # Build ensemble name
            test_indices = sorted(selected + [candidate])
            ensemble_name = 'ensemble_' + '_'.join(map(str, test_indices))

            # Check if this ensemble exists
            if ensemble_name in ensemble_dict:
                score = ensemble_dict[ensemble_name][metric]
                if score > best_score:
                    best_score = score
                    best_model = candidate
                    best_ensemble_name = ensemble_name

        if best_model is not None:
            selected.append(best_model)
            scores.append(best_score)
            ensembles.append(best_ensemble_name)
            remaining.remove(best_model)
        else:
            break

    return selected, scores, ensembles


def plot_individual_models(data, backbone_names, output_dir):
    """Plot performance of individual models."""
    # Get single model results
    single_models = [(row['ensemble'], row) for row in data
                     if get_ensemble_size(row['ensemble']) == 1]
    single_models.sort(key=lambda x: parse_ensemble_name(x[0])[0])

    model_names = [backbone_names[parse_ensemble_name(ens)[0]] for ens, _ in single_models]
    miou_maj = [row['mIoU_majvote'] for _, row in single_models]
    miou_ora = [row['mIoU_oracle'] for _, row in single_models]
    acc_maj = [row['accuracy_majvote'] for _, row in single_models]
    acc_ora = [row['accuracy_oracle'] for _, row in single_models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # mIoU comparison
    ax = axes[0]
    x = np.arange(len(model_names))
    width = 0.35

    ax.bar(x - width/2, miou_maj, width, label='Majority Vote', alpha=0.8)
    ax.bar(x + width/2, miou_ora, width, label='Oracle', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('mIoU')
    ax.set_title('Individual Model Performance - mIoU')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Accuracy comparison
    ax = axes[1]
    ax.bar(x - width/2, acc_maj, width, label='Majority Vote', alpha=0.8)
    ax.bar(x + width/2, acc_ora, width, label='Oracle', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Individual Model Performance - Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'individual_models.png', bbox_inches='tight')
    plt.close()

    return single_models


def plot_ensemble_size_vs_performance(data, backbone_names, output_dir):
    """Plot how performance varies with ensemble size."""
    # Group by ensemble size
    size_groups = defaultdict(lambda: {'miou_maj': [], 'acc_maj': [], 'miou_ora': [], 'acc_ora': []})

    for row in data:
        size = get_ensemble_size(row['ensemble'])
        size_groups[size]['miou_maj'].append(row['mIoU_majvote'])
        size_groups[size]['acc_maj'].append(row['accuracy_majvote'])
        size_groups[size]['miou_ora'].append(row['mIoU_oracle'])
        size_groups[size]['acc_ora'].append(row['accuracy_oracle'])

    sizes = sorted(size_groups.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('miou_maj', 'mIoU (Majority Vote)'),
        ('acc_maj', 'Accuracy (Majority Vote)'),
        ('miou_ora', 'mIoU (Oracle)'),
        ('acc_ora', 'Accuracy (Oracle)')
    ]

    for idx, (metric_key, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Box plot by ensemble size
        data_to_plot = [size_groups[s][metric_key] for s in sizes]

        bp = ax.boxplot(data_to_plot, positions=sizes, widths=0.6, patch_artist=True)

        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        # Add mean line
        means = [np.mean(size_groups[s][metric_key]) for s in sizes]
        ax.plot(sizes, means, 'r--', linewidth=2, label='Mean', alpha=0.7)

        ax.set_xlabel('Ensemble Size')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Ensemble Size')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

        # Add count annotations
        ylim = ax.get_ylim()
        for s in sizes:
            ax.text(s, ylim[0], f'n={len(size_groups[s][metric_key])}',
                   ha='center', va='top', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_size_vs_performance.png', bbox_inches='tight')
    plt.close()


def plot_greedy_selection(data, backbone_names, output_dir):
    """Plot greedy ensemble selection results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('mIoU_majvote', 'mIoU (Majority Vote)'),
        ('accuracy_majvote', 'Accuracy (Majority Vote)'),
        ('mIoU_oracle', 'mIoU (Oracle)'),
        ('accuracy_oracle', 'Accuracy (Oracle)')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        selected, scores, ensembles = greedy_ensemble_selection(data, backbone_names, metric)

        # Plot scores
        x = np.arange(len(scores))
        ax.plot(x, scores, 'o-', linewidth=2, markersize=8, label='Greedy Selection')

        # Annotate with model names
        for i, (idx_val, score) in enumerate(zip(selected, scores)):
            if i == 0:
                ax.annotate(backbone_names[idx_val], (i, score),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8, weight='bold')
            else:
                ax.annotate(f'+{backbone_names[idx_val]}', (i, score),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8)

        ax.set_xlabel('Step (# Models)')
        ax.set_ylabel(label)
        ax.set_title(f'Greedy Ensemble Building - {label}')
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in x])

        # Add improvement annotations
        if len(scores) > 1:
            for i in range(1, len(scores)):
                improvement = scores[i] - scores[i-1]
                color = 'green' if improvement > 0 else 'red'
                mid_y = scores[i-1] + (scores[i] - scores[i-1])/2
                ax.annotate(f'+{improvement:.3f}' if improvement > 0 else f'{improvement:.3f}',
                           xy=(i, scores[i]), xytext=(i+0.1, mid_y),
                           ha='left', fontsize=7, color=color, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'greedy_selection.png', bbox_inches='tight')
    plt.close()

    # Print greedy selection results
    print("\n=== Greedy Selection Results ===")
    for metric, label in metrics:
        selected, scores, ensembles = greedy_ensemble_selection(data, backbone_names, metric)
        print(f"\n{label}:")
        for i, (idx_val, score, ens) in enumerate(zip(selected, scores, ensembles)):
            models = ' + '.join([backbone_names[j] for j in sorted(selected[:i+1])])
            print(f"  Step {i+1}: {models} = {score:.4f}")


def plot_pairwise_combinations(data, backbone_names, output_dir):
    """Plot heatmap of pairwise model combinations."""
    n_models = len(backbone_names)

    # Create lookup dict
    ensemble_dict = {row['ensemble']: row for row in data}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics = [
        ('mIoU_majvote', 'mIoU (Majority Vote)'),
        ('accuracy_majvote', 'Accuracy (Majority Vote)')
    ]

    for ax, (metric, label) in zip(axes, metrics):
        # Create matrix for pairwise combinations
        matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    # Single model
                    ensemble_name = f'ensemble_{i}'
                else:
                    # Pair of models
                    indices = sorted([i, j])
                    ensemble_name = f'ensemble_{indices[0]}_{indices[1]}'

                if ensemble_name in ensemble_dict:
                    matrix[i, j] = ensemble_dict[ensemble_name][metric]

        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=matrix[matrix>0].min(), vmax=matrix.max())

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)

        # Set ticks and labels
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(backbone_names, rotation=45, ha='right')
        ax.set_yticklabels(backbone_names)

        # Add values in cells
        for i in range(n_models):
            for j in range(n_models):
                if matrix[i, j] > 0:
                    text_color = "black" if matrix[i, j] < matrix.max()*0.7 else "white"
                    ax.text(j, i, f'{matrix[i, j]:.3f}',
                           ha="center", va="center", color=text_color,
                           fontsize=8)

        ax.set_title(f'Pairwise Ensemble Performance - {label}')
        ax.set_xlabel('Model')
        ax.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_combinations.png', bbox_inches='tight')
    plt.close()


def plot_oracle_vs_majority(data, backbone_names, output_dir):
    """Plot oracle vs majority vote performance."""
    # Extract data
    ensemble_sizes = [get_ensemble_size(row['ensemble']) for row in data]
    miou_maj = [row['mIoU_majvote'] for row in data]
    miou_ora = [row['mIoU_oracle'] for row in data]
    acc_maj = [row['accuracy_majvote'] for row in data]
    acc_ora = [row['accuracy_oracle'] for row in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # mIoU scatter
    ax = axes[0]
    scatter = ax.scatter(miou_maj, miou_ora, c=ensemble_sizes, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add diagonal line
    lim = [min(min(miou_maj), min(miou_ora)), max(max(miou_maj), max(miou_ora))]
    ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('mIoU (Majority Vote)')
    ax.set_ylabel('mIoU (Oracle)')
    ax.set_title('Oracle vs Majority Vote - mIoU')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Ensemble Size')

    # Accuracy scatter
    ax = axes[1]
    scatter = ax.scatter(acc_maj, acc_ora, c=ensemble_sizes, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add diagonal line
    lim = [min(min(acc_maj), min(acc_ora)), max(max(acc_maj), max(acc_ora))]
    ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Accuracy (Majority Vote)')
    ax.set_ylabel('Accuracy (Oracle)')
    ax.set_title('Oracle vs Majority Vote - Accuracy')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Ensemble Size')

    plt.tight_layout()
    plt.savefig(output_dir / 'oracle_vs_majority.png', bbox_inches='tight')
    plt.close()


def plot_best_ensembles_by_size(data, backbone_names, output_dir):
    """Plot best ensembles for each size."""
    # Group by size and find best
    size_groups = defaultdict(list)
    for row in data:
        size = get_ensemble_size(row['ensemble'])
        size_groups[size].append(row)

    best_by_size = []
    for size in sorted(size_groups.keys()):
        best = max(size_groups[size], key=lambda x: x['mIoU_majvote'])
        best_by_size.append((size, best))

    sizes = [s for s, _ in best_by_size]
    miou_maj = [row['mIoU_majvote'] for _, row in best_by_size]
    acc_maj = [row['accuracy_majvote'] for _, row in best_by_size]
    miou_ora = [row['mIoU_oracle'] for _, row in best_by_size]
    acc_ora = [row['accuracy_oracle'] for _, row in best_by_size]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sizes))
    width = 0.2

    ax.bar(x - width*1.5, miou_maj, width, label='mIoU (Majority)', alpha=0.8)
    ax.bar(x - width*0.5, acc_maj, width, label='Accuracy (Majority)', alpha=0.8)
    ax.bar(x + width*0.5, miou_ora, width, label='mIoU (Oracle)', alpha=0.8)
    ax.bar(x + width*1.5, acc_ora, width, label='Accuracy (Oracle)', alpha=0.8)

    ax.set_xlabel('Ensemble Size')
    ax.set_ylabel('Score')
    ax.set_title('Best Ensemble Performance by Size')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Annotate with ensemble members
    for i, (size, row) in enumerate(best_by_size):
        indices = parse_ensemble_name(row['ensemble'])
        names = [backbone_names[idx] for idx in indices]
        label = ', '.join([n[:4] for n in names])  # Abbreviate names
        ax.text(i, ax.get_ylim()[0]*0.95, label,
               ha='center', va='top', fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'best_ensembles_by_size.png', bbox_inches='tight')
    plt.close()

    # Print best ensembles
    print("\n=== Best Ensembles by Size ===")
    for size, row in best_by_size:
        indices = parse_ensemble_name(row['ensemble'])
        names = [backbone_names[idx] for idx in indices]
        print(f"Size {size}: {' + '.join(names)}")
        print(f"  mIoU (maj): {row['mIoU_majvote']:.4f}, Acc (maj): {row['accuracy_majvote']:.4f}")
        print(f"  mIoU (ora): {row['mIoU_oracle']:.4f}, Acc (ora): {row['accuracy_oracle']:.4f}")


def generate_summary_statistics(data, backbone_names, output_dir):
    """Generate summary statistics."""
    size_groups = defaultdict(lambda: {'miou_maj': [], 'acc_maj': [], 'miou_ora': [], 'acc_ora': []})

    for row in data:
        size = get_ensemble_size(row['ensemble'])
        size_groups[size]['miou_maj'].append(row['mIoU_majvote'])
        size_groups[size]['acc_maj'].append(row['accuracy_majvote'])
        size_groups[size]['miou_ora'].append(row['mIoU_oracle'])
        size_groups[size]['acc_ora'].append(row['accuracy_oracle'])

    print("\n=== Summary Statistics by Ensemble Size ===")
    print(f"{'Size':<6} {'Metric':<20} {'Mean':<10} {'Std':<10} {'Max':<10} {'Min':<10} {'Count':<6}")
    print("-" * 72)

    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write(f"{'Size':<6} {'Metric':<20} {'Mean':<10} {'Std':<10} {'Max':<10} {'Min':<10} {'Count':<6}\n")
        f.write("-" * 72 + "\n")

        for size in sorted(size_groups.keys()):
            for metric_key, metric_name in [('miou_maj', 'mIoU (maj)'),
                                            ('acc_maj', 'Acc (maj)'),
                                            ('miou_ora', 'mIoU (ora)'),
                                            ('acc_ora', 'Acc (ora)')]:
                values = size_groups[size][metric_key]
                mean = np.mean(values)
                std = np.std(values)
                max_val = np.max(values)
                min_val = np.min(values)
                count = len(values)

                line = f"{size:<6} {metric_name:<20} {mean:<10.4f} {std:<10.4f} {max_val:<10.4f} {min_val:<10.4f} {count:<6}\n"
                print(line.strip())
                f.write(line)


def main():
    parser = argparse.ArgumentParser(description='Plot ensemble evaluation results')
    parser.add_argument('--config', type=str, default='config/eval_ensemble.yaml',
                       help='Path to config file')
    parser.add_argument('--results', type=str, default='ensemble_results/ensemble_evaluation.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='ensemble_results_plots',
                       help='Output directory for plots')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    backbone_names = cfg['ensemble']['backbones']

    # Load results
    data = load_csv(args.results)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loaded {len(data)} ensemble results")
    print(f"Backbone models: {backbone_names}")
    print(f"Saving plots to: {output_dir}")

    # Generate all plots
    print("\n[1/7] Plotting individual models...")
    single_models = plot_individual_models(data, backbone_names, output_dir)

    print("[2/7] Plotting ensemble size vs performance...")
    plot_ensemble_size_vs_performance(data, backbone_names, output_dir)

    print("[3/7] Plotting greedy selection...")
    plot_greedy_selection(data, backbone_names, output_dir)

    print("[4/7] Plotting pairwise combinations...")
    plot_pairwise_combinations(data, backbone_names, output_dir)

    print("[5/7] Plotting oracle vs majority vote...")
    plot_oracle_vs_majority(data, backbone_names, output_dir)

    print("[6/7] Plotting best ensembles by size...")
    plot_best_ensembles_by_size(data, backbone_names, output_dir)

    print("[7/7] Generating summary statistics...")
    generate_summary_statistics(data, backbone_names, output_dir)

    # Save configuration info
    with open(output_dir / 'README.txt', 'w') as f:
        f.write("Ensemble Evaluation Plots\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Results: {args.results}\n")
        f.write(f"Backbone models: {', '.join(backbone_names)}\n")
        f.write(f"\nModel indices:\n")
        for i, name in enumerate(backbone_names):
            f.write(f"  {i}: {name}\n")

    print(f"\nAll plots saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
