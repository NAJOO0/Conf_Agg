"""
Adaptive Stopping Experiment with Confidence-based Early Termination (Version 2)

This version uses appropriate threshold ranges based on the confidence value distribution.
Confidence values are calculated as -logprobs, so they are positive values typically in range 3-19.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path


class AdaptiveStoppingExperiment:
    """Experiment class for testing adaptive stopping strategies."""

    def __init__(self, data_path: str, max_samples: int = 32):
        """
        Initialize experiment.

        Args:
            data_path: Path to parquet dataset
            max_samples: Maximum number of inference samples per problem
        """
        print(f"Loading dataset from {data_path}...")
        self.df = pd.read_parquet(data_path)
        self.max_samples = max_samples

        # Group by problem_id
        print("Grouping responses by problem...")
        self.problems = self._group_by_problem()
        print(f"Found {len(self.problems)} unique problems")

    def _group_by_problem(self) -> Dict[str, pd.DataFrame]:
        """Group responses by problem_id."""
        problems = {}
        for problem_id, group in self.df.groupby('problem_id'):
            # Take only first max_samples responses
            group = group.head(self.max_samples).reset_index(drop=True)
            problems[problem_id] = group
        return problems

    def calculate_majority_confidence(
        self,
        answers: List[str],
        confidences: List[float],
        confidence_method: str = 'mean_group_confidence'
    ) -> Tuple[str, float, int]:
        """
        Calculate majority answer, its confidence score, and vote count.

        Args:
            answers: List of answer strings
            confidences: List of confidence scores
            confidence_method: Which confidence column to use

        Returns:
            (majority_answer, confidence_score, vote_count)
        """
        # Count answers
        answer_counts = Counter(answers)

        # Get majority answer
        majority_answer, vote_count = answer_counts.most_common(1)[0]

        # Calculate weighted confidence for majority answer
        majority_confidences = [
            conf for ans, conf in zip(answers, confidences)
            if ans == majority_answer
        ]

        if not majority_confidences:
            return majority_answer, 0.0, vote_count

        # Average confidence of majority answer
        avg_confidence = np.mean(majority_confidences)

        return majority_answer, avg_confidence, vote_count

    def simulate_adaptive_stopping(
        self,
        threshold: float,
        confidence_method: str = 'mean_group_confidence',
        step_size: int = 1,
        min_samples: int = 2
    ) -> Dict:
        """
        Simulate adaptive stopping with given threshold.

        Args:
            threshold: Confidence threshold for early stopping
            confidence_method: Which confidence metric to use
            step_size: How many samples to add each step
            min_samples: Minimum number of samples before allowing early stop

        Returns:
            Dictionary with results
        """
        results = []

        for problem_id, problem_data in self.problems.items():
            ground_truth = problem_data['ground_truth'].iloc[0]

            # Simulate sequential inference
            stopped_at = None
            final_answer = None
            final_confidence = 0.0
            final_vote_count = 0

            for n in range(step_size, len(problem_data) + 1, step_size):
                # Get first n samples
                current_samples = problem_data.head(n)

                # Extract answers and confidences
                answers = current_samples['final_answer'].tolist()
                confidences = current_samples[confidence_method].tolist()

                # Calculate majority and confidence
                majority_answer, confidence, vote_count = self.calculate_majority_confidence(
                    answers, confidences, confidence_method
                )

                # Check stopping condition (only after min_samples)
                if n >= min_samples and confidence >= threshold:
                    stopped_at = n
                    final_answer = majority_answer
                    final_confidence = confidence
                    final_vote_count = vote_count
                    break

            # If didn't stop early, use all samples
            if stopped_at is None:
                stopped_at = len(problem_data)
                answers = problem_data['final_answer'].tolist()
                confidences = problem_data[confidence_method].tolist()
                final_answer, final_confidence, final_vote_count = self.calculate_majority_confidence(
                    answers, confidences, confidence_method
                )

            # Check correctness
            is_correct = (final_answer == ground_truth)

            # Calculate token usage
            tokens_used = problem_data.head(stopped_at)['output_token_count'].sum()
            total_tokens = problem_data['output_token_count'].sum()

            results.append({
                'problem_id': problem_id,
                'stopped_at': stopped_at,
                'total_samples': len(problem_data),
                'tokens_used': tokens_used,
                'total_tokens': total_tokens,
                'token_ratio': tokens_used / total_tokens if total_tokens > 0 else 0,
                'is_correct': is_correct,
                'final_confidence': final_confidence,
                'vote_count': final_vote_count,
                'ground_truth': ground_truth,
                'predicted_answer': final_answer
            })

        return {
            'threshold': threshold,
            'confidence_method': confidence_method,
            'min_samples': min_samples,
            'results': results,
            'accuracy': np.mean([r['is_correct'] for r in results]),
            'avg_samples': np.mean([r['stopped_at'] for r in results]),
            'avg_token_ratio': np.mean([r['token_ratio'] for r in results]),
            'total_problems': len(results)
        }

    def run_threshold_sweep(
        self,
        thresholds: List[float],
        confidence_method: str = 'mean_group_confidence',
        step_size: int = 1,
        min_samples: int = 2
    ) -> List[Dict]:
        """
        Run experiment with multiple threshold values.

        Args:
            thresholds: List of threshold values to test
            confidence_method: Which confidence metric to use
            step_size: How many samples to add each step
            min_samples: Minimum samples before allowing early stop

        Returns:
            List of results for each threshold
        """
        all_results = []

        # Add baseline first (no early stopping)
        print("\nRunning baseline (no early stopping)...")
        baseline = self._calculate_baseline(confidence_method)
        all_results.append(baseline)
        print(f"  Accuracy: {baseline['accuracy']:.4f}")
        print(f"  Avg samples: {baseline['avg_samples']:.2f}/{self.max_samples}")
        print(f"  Token ratio: {baseline['avg_token_ratio']:.4f}")

        # Run threshold experiments
        for threshold in thresholds:
            print(f"\nRunning experiment with threshold={threshold:.2f}...")
            result = self.simulate_adaptive_stopping(
                threshold, confidence_method, step_size, min_samples
            )
            all_results.append(result)

            acc_diff = (result['accuracy'] - baseline['accuracy']) * 100
            token_savings = (1 - result['avg_token_ratio']) * 100

            print(f"  Accuracy: {result['accuracy']:.4f} ({acc_diff:+.2f}%)")
            print(f"  Avg samples: {result['avg_samples']:.2f}/{self.max_samples}")
            print(f"  Token ratio: {result['avg_token_ratio']:.4f} (Savings: {token_savings:.1f}%)")

        return all_results

    def _calculate_baseline(self, confidence_method: str) -> Dict:
        """Calculate baseline results (always use all samples)."""
        results = []

        for problem_id, problem_data in self.problems.items():
            ground_truth = problem_data['ground_truth'].iloc[0]

            # Use all samples
            answers = problem_data['final_answer'].tolist()
            confidences = problem_data[confidence_method].tolist()
            final_answer, final_confidence, vote_count = self.calculate_majority_confidence(
                answers, confidences, confidence_method
            )

            is_correct = (final_answer == ground_truth)
            tokens_used = problem_data['output_token_count'].sum()

            results.append({
                'problem_id': problem_id,
                'stopped_at': len(problem_data),
                'total_samples': len(problem_data),
                'tokens_used': tokens_used,
                'total_tokens': tokens_used,
                'token_ratio': 1.0,
                'is_correct': is_correct,
                'final_confidence': final_confidence,
                'vote_count': vote_count,
                'ground_truth': ground_truth,
                'predicted_answer': final_answer
            })

        return {
            'threshold': float('inf'),  # No threshold (baseline)
            'confidence_method': confidence_method,
            'min_samples': 0,
            'results': results,
            'accuracy': np.mean([r['is_correct'] for r in results]),
            'avg_samples': np.mean([r['stopped_at'] for r in results]),
            'avg_token_ratio': 1.0,
            'total_problems': len(results),
            'is_baseline': True
        }

    def plot_results(
        self,
        results: List[Dict],
        output_path: str = "adaptive_stopping_results.png"
    ):
        """
        Create visualization of results.

        Args:
            results: List of experiment results
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Get baseline
        baseline = [r for r in results if r.get('is_baseline', False)][0]
        baseline_acc = baseline['accuracy']
        baseline_tokens = 1.0

        # Extract data (exclude baseline)
        exp_results = [r for r in results if not r.get('is_baseline', False)]
        thresholds = [r['threshold'] for r in exp_results]
        accuracies = [r['accuracy'] for r in exp_results]
        token_ratios = [r['avg_token_ratio'] for r in exp_results]
        avg_samples = [r['avg_samples'] for r in exp_results]

        # Plot 1: Pareto Efficiency (Accuracy vs Token Ratio)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(token_ratios, accuracies, s=150, alpha=0.7,
                            c=thresholds, cmap='viridis', edgecolors='black', linewidth=1.5)
        ax1.scatter([baseline_tokens], [baseline_acc], s=300, color='red', marker='*',
                   label='Baseline (All 32 samples)', zorder=5, edgecolors='black', linewidth=2)
        ax1.axhline(y=baseline_acc, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax1.axvline(x=baseline_tokens, color='red', linestyle='--', alpha=0.3, linewidth=2)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Confidence Threshold', fontsize=11)

        ax1.set_xlabel('Token Usage Ratio', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax1.set_title('Pareto Efficiency: Accuracy vs Cost', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(-0.05, 1.1)

        # Plot 2: Threshold vs Accuracy & Token Savings
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()

        line1 = ax2.plot(thresholds, accuracies, 'o-', linewidth=2.5, markersize=10,
                        color='#2E86AB', label='Accuracy')
        ax2.axhline(y=baseline_acc, color='red', linestyle='--',
                   label=f'Baseline: {baseline_acc:.4f}', linewidth=2, alpha=0.7)

        token_savings = [(1 - ratio) * 100 for ratio in token_ratios]
        line2 = ax2_twin.plot(thresholds, token_savings, 's-', linewidth=2.5, markersize=10,
                             color='#06A77D', label='Token Savings (%)')

        ax2.set_xlabel('Confidence Threshold', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold', color='#2E86AB')
        ax2_twin.set_ylabel('Token Savings (%)', fontsize=13, fontweight='bold', color='#06A77D')
        ax2.set_title('Threshold Impact on Accuracy & Cost', fontsize=15, fontweight='bold')

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='y', labelcolor='#2E86AB')
        ax2_twin.tick_params(axis='y', labelcolor='#06A77D')

        # Plot 3: Threshold vs Avg Samples
        ax3 = axes[1, 0]
        ax3.plot(thresholds, avg_samples, 'o-', linewidth=2.5, markersize=10,
                color='#A23B72')
        ax3.axhline(y=self.max_samples, color='red', linestyle='--',
                   label=f'Max: {self.max_samples}', linewidth=2, alpha=0.7)
        ax3.fill_between(thresholds, avg_samples, alpha=0.3, color='#A23B72')

        ax3.set_xlabel('Confidence Threshold', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Avg Inference Count', fontsize=13, fontweight='bold')
        ax3.set_title('Early Stopping Effectiveness', fontsize=15, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # Plot 4: Cost-Benefit Analysis
        ax4 = axes[1, 1]

        # Calculate efficiency metric: (accuracy / baseline_accuracy) / token_ratio
        efficiency = [(acc / baseline_acc) / ratio if ratio > 0 else 0
                     for acc, ratio in zip(accuracies, token_ratios)]

        colors = ['green' if eff > 1.0 else 'orange' for eff in efficiency]
        bars = ax4.bar(range(len(thresholds)), efficiency, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                   label='Break-even (Baseline)', alpha=0.7)

        ax4.set_xlabel('Confidence Threshold', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Efficiency Score\n(Relative Accuracy / Token Ratio)',
                      fontsize=13, fontweight='bold')
        ax4.set_title('Cost-Benefit Efficiency Score', fontsize=15, fontweight='bold')
        ax4.set_xticks(range(len(thresholds)))
        ax4.set_xticklabels([f'{t:.1f}' for t in thresholds])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add value labels on bars
        for i, (bar, eff) in enumerate(zip(bars, efficiency)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        plt.close()

    def create_summary_table(self, results: List[Dict]) -> pd.DataFrame:
        """Create summary table of results."""
        # Find baseline for comparison first
        baseline = [r for r in results if r.get('is_baseline', False)][0]

        summary_data = []

        for r in results:
            is_baseline = r.get('is_baseline', False)

            # Calculate efficiency score
            if is_baseline:
                efficiency = 1.0
            else:
                rel_acc = r['accuracy'] / baseline['accuracy'] if baseline['accuracy'] > 0 else 0
                efficiency = rel_acc / r['avg_token_ratio'] if r['avg_token_ratio'] > 0 else 0

            summary_data.append({
                'Threshold': 'Baseline (∞)' if is_baseline else f"{r['threshold']:.1f}",
                'Accuracy': f"{r['accuracy']:.4f}",
                'Acc. Drop (%)': f"{(baseline['accuracy'] - r['accuracy']) * 100:+.2f}"
                    if not is_baseline else "0.00",
                'Avg Samples': f"{r['avg_samples']:.1f}",
                'Token Ratio': f"{r['avg_token_ratio']:.3f}",
                'Token Savings (%)': f"{(1 - r['avg_token_ratio']) * 100:.1f}",
                'Efficiency': f"{efficiency:.2f}"
            })

        df = pd.DataFrame(summary_data)

        # Reorder to put baseline first
        baseline_row = df[df['Threshold'].str.contains('Baseline')]
        other_rows = df[~df['Threshold'].str.contains('Baseline')]
        df = pd.concat([baseline_row, other_rows], ignore_index=True)

        return df


def main():
    """Main experiment runner."""
    # Configuration
    DATA_PATH = "/mnt/data1/projects/Conf_Agg/output_s/generated/dataset_4000_think.parquet"
    OUTPUT_DIR = Path("/mnt/data1/projects/Conf_Agg/experiments")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Experiment parameters
    # Confidence values are -logprobs, typically in range 3-19
    # Mean: 10.4, Median: 10.35, 25th: 9.13, 75th: 11.67
    THRESHOLDS = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    CONFIDENCE_METHOD = 'mean_group_confidence'
    STEP_SIZE = 1  # Add 1 sample at a time
    MIN_SAMPLES = 2  # Require at least 2 samples before early stopping

    print("="*80)
    print("Adaptive Stopping Experiment with Confidence-based Early Termination v2")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data: {DATA_PATH}")
    print(f"  Confidence Method: {CONFIDENCE_METHOD}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Step Size: {STEP_SIZE}")
    print(f"  Min Samples: {MIN_SAMPLES}")
    print()

    # Initialize experiment
    exp = AdaptiveStoppingExperiment(DATA_PATH, max_samples=32)

    # Run threshold sweep
    results = exp.run_threshold_sweep(
        thresholds=THRESHOLDS,
        confidence_method=CONFIDENCE_METHOD,
        step_size=STEP_SIZE,
        min_samples=MIN_SAMPLES
    )

    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    plot_path = OUTPUT_DIR / "adaptive_stopping_analysis_v2.png"
    exp.plot_results(results, str(plot_path))

    # Create summary table
    print("\nCreating summary table...")
    summary_df = exp.create_summary_table(results)
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Save results
    summary_path = OUTPUT_DIR / "adaptive_stopping_summary_v2.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Save detailed results
    results_path = OUTPUT_DIR / "adaptive_stopping_detailed_v2.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_path}")

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    baseline = [r for r in results if r.get('is_baseline', False)][0]
    exp_results = [r for r in results if not r.get('is_baseline', False)]

    # Find best efficiency
    best_eff = max(exp_results, key=lambda x: (x['accuracy'] / baseline['accuracy']) / x['avg_token_ratio'])

    print(f"\n1. Baseline Performance (All {baseline['avg_samples']:.0f} samples):")
    print(f"   - Accuracy: {baseline['accuracy']:.4f}")
    print(f"   - Token Usage: 100%")

    print(f"\n2. Best Efficiency Configuration (Threshold = {best_eff['threshold']:.1f}):")
    print(f"   - Accuracy: {best_eff['accuracy']:.4f} ({(best_eff['accuracy'] - baseline['accuracy']) * 100:+.2f}%)")
    print(f"   - Avg Samples: {best_eff['avg_samples']:.1f}")
    print(f"   - Token Savings: {(1 - best_eff['avg_token_ratio']) * 100:.1f}%")

    # Find minimal accuracy drop
    min_drop = min(exp_results, key=lambda x: abs(x['accuracy'] - baseline['accuracy']))

    print(f"\n3. Minimal Accuracy Drop (Threshold = {min_drop['threshold']:.1f}):")
    print(f"   - Accuracy: {min_drop['accuracy']:.4f} ({(min_drop['accuracy'] - baseline['accuracy']) * 100:+.2f}%)")
    print(f"   - Avg Samples: {min_drop['avg_samples']:.1f}")
    print(f"   - Token Savings: {(1 - min_drop['avg_token_ratio']) * 100:.1f}%")

    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
