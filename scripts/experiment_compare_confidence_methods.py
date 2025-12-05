"""
Compare different confidence calculation methods for adaptive stopping.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Import the experiment class
sys.path.append('/mnt/data1/projects/Conf_Agg/scripts')
from experiment_adaptive_stopping_v2 import AdaptiveStoppingExperiment


def compare_confidence_methods():
    """Compare different confidence calculation methods."""

    DATA_PATH = "/mnt/data1/projects/Conf_Agg/output_s/generated/dataset_4000_think.parquet"
    OUTPUT_DIR = Path("/mnt/data1/projects/Conf_Agg/experiments")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Test different confidence methods
    CONFIDENCE_METHODS = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'lowest_group_confidence',
        'tail_confidence'
    ]

    # Thresholds for each method (adjusted based on value ranges)
    THRESHOLDS = {
        'mean_group_confidence': [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        'bottom_10_percent_confidence': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        'lowest_group_confidence': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        'tail_confidence': [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    }

    print("="*80)
    print("Comparing Different Confidence Calculation Methods")
    print("="*80)

    all_method_results = {}

    for method in CONFIDENCE_METHODS:
        print(f"\n{'='*80}")
        print(f"Testing: {method}")
        print(f"{'='*80}")

        exp = AdaptiveStoppingExperiment(DATA_PATH, max_samples=32)

        results = exp.run_threshold_sweep(
            thresholds=THRESHOLDS[method],
            confidence_method=method,
            step_size=1,
            min_samples=2
        )

        all_method_results[method] = results

        # Save individual results
        method_output_path = OUTPUT_DIR / f"adaptive_stopping_{method}.png"
        exp.plot_results(results, str(method_output_path))

        summary_df = exp.create_summary_table(results)
        summary_path = OUTPUT_DIR / f"adaptive_stopping_{method}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

    # Create comparison plot
    print("\n" + "="*80)
    print("Creating comparison plot...")
    create_comparison_plot(all_method_results, OUTPUT_DIR)

    # Create comparison summary
    create_comparison_summary(all_method_results, OUTPUT_DIR)

    print("\n" + "="*80)
    print("Comparison completed successfully!")
    print("="*80)


def create_comparison_plot(all_results: Dict, output_dir: Path):
    """Create comparison plot across different confidence methods."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    colors = {
        'mean_group_confidence': '#2E86AB',
        'bottom_10_percent_confidence': '#A23B72',
        'lowest_group_confidence': '#F18F01',
        'tail_confidence': '#06A77D'
    }

    labels = {
        'mean_group_confidence': 'Mean Group',
        'bottom_10_percent_confidence': 'Bottom 10%',
        'lowest_group_confidence': 'Lowest Group',
        'tail_confidence': 'Tail'
    }

    # Extract data for each method
    for method, results in all_results.items():
        baseline = [r for r in results if r.get('is_baseline', False)][0]
        exp_results = [r for r in results if not r.get('is_baseline', False)]

        token_ratios = [r['avg_token_ratio'] for r in exp_results]
        accuracies = [r['accuracy'] for r in exp_results]
        thresholds = [r['threshold'] for r in exp_results]
        avg_samples = [r['avg_samples'] for r in exp_results]

        # Plot 1: Pareto Efficiency
        ax1 = axes[0, 0]
        ax1.plot(token_ratios, accuracies, 'o-', label=labels[method],
                color=colors[method], linewidth=2, markersize=8, alpha=0.7)

        # Plot 2: Token Savings vs Accuracy Drop
        ax2 = axes[0, 1]
        token_savings = [(1 - r) * 100 for r in token_ratios]
        acc_drops = [(baseline['accuracy'] - acc) * 100 for acc in accuracies]
        ax2.plot(token_savings, acc_drops, 'o-', label=labels[method],
                color=colors[method], linewidth=2, markersize=8, alpha=0.7)

        # Plot 3: Efficiency Score
        ax3 = axes[1, 0]
        efficiency = [(acc / baseline['accuracy']) / ratio if ratio > 0 else 0
                     for acc, ratio in zip(accuracies, token_ratios)]
        ax3.plot(thresholds, efficiency, 'o-', label=labels[method],
                color=colors[method], linewidth=2, markersize=8, alpha=0.7)

        # Plot 4: Avg Samples vs Threshold
        ax4 = axes[1, 1]
        ax4.plot(thresholds, avg_samples, 'o-', label=labels[method],
                color=colors[method], linewidth=2, markersize=8, alpha=0.7)

    # Configure Plot 1: Pareto Efficiency
    ax1.axhline(y=baseline['accuracy'], color='red', linestyle='--',
               label='Baseline Accuracy', linewidth=2, alpha=0.7)
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax1.set_xlabel('Token Usage Ratio', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Pareto Efficiency Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Configure Plot 2: Token Savings vs Accuracy Drop
    ax2.axhline(y=0, color='red', linestyle='--', label='No Accuracy Drop',
               linewidth=2, alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax2.set_xlabel('Token Savings (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy Drop (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Trade-off: Cost Savings vs Accuracy Loss', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Configure Plot 3: Efficiency Score
    ax3.axhline(y=1.0, color='red', linestyle='--', label='Break-even',
               linewidth=2, alpha=0.7)
    ax3.set_xlabel('Confidence Threshold', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Efficiency Score', fontsize=13, fontweight='bold')
    ax3.set_title('Cost-Benefit Efficiency by Method', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Configure Plot 4: Avg Samples
    ax4.axhline(y=32, color='red', linestyle='--', label='Max Samples',
               linewidth=2, alpha=0.7)
    ax4.set_xlabel('Confidence Threshold', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Avg Inference Count', fontsize=13, fontweight='bold')
    ax4.set_title('Early Stopping Aggressiveness', fontsize=15, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = output_dir / "confidence_methods_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


def create_comparison_summary(all_results: Dict, output_dir: Path):
    """Create comparison summary table."""

    summary_data = []

    for method, results in all_results.items():
        baseline = [r for r in results if r.get('is_baseline', False)][0]
        exp_results = [r for r in results if not r.get('is_baseline', False)]

        # Find best efficiency configuration
        best_eff = max(exp_results,
                      key=lambda x: (x['accuracy'] / baseline['accuracy']) / x['avg_token_ratio'])

        # Find configuration with minimal accuracy drop
        min_drop = min(exp_results, key=lambda x: abs(x['accuracy'] - baseline['accuracy']))

        # Find configuration with highest token savings while maintaining accuracy
        good_acc = [r for r in exp_results if r['accuracy'] >= baseline['accuracy'] * 0.99]
        if good_acc:
            best_savings = min(good_acc, key=lambda x: x['avg_token_ratio'])
        else:
            best_savings = best_eff

        summary_data.append({
            'Method': method.replace('_', ' ').title(),
            'Baseline Acc': f"{baseline['accuracy']:.4f}",
            'Best Eff. Threshold': f"{best_eff['threshold']:.1f}",
            'Best Eff. Acc': f"{best_eff['accuracy']:.4f}",
            'Best Eff. Token Savings': f"{(1 - best_eff['avg_token_ratio']) * 100:.1f}%",
            'Best Eff. Score': f"{(best_eff['accuracy'] / baseline['accuracy']) / best_eff['avg_token_ratio']:.2f}",
            'Min Drop Threshold': f"{min_drop['threshold']:.1f}",
            'Min Drop Acc': f"{min_drop['accuracy']:.4f}",
            'Min Drop Token Savings': f"{(1 - min_drop['avg_token_ratio']) * 100:.1f}%"
        })

    df = pd.DataFrame(summary_data)

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    summary_path = output_dir / "confidence_methods_comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    compare_confidence_methods()
