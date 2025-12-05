"""
Adaptive Stopping Experiment with Confidence-based Early Termination

This script tests whether confidence scores can help reduce inference costs
while maintaining accuracy through dynamic stopping strategies.
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
    ) -> Tuple[str, float]:
        """
        Calculate majority answer and its confidence score.

        Args:
            answers: List of answer strings
            confidences: List of confidence scores
            confidence_method: Which confidence column to use

        Returns:
            (majority_answer, confidence_score)
        """
        # Count answers
        answer_counts = Counter(answers)

        # Get majority answer
        majority_answer = answer_counts.most_common(1)[0][0]

        # Calculate weighted confidence for majority answer
        majority_confidences = [
            conf for ans, conf in zip(answers, confidences)
            if ans == majority_answer
        ]

        if not majority_confidences:
            return majority_answer, 0.0

        # Average confidence of majority answer
        avg_confidence = np.mean(majority_confidences)

        return majority_answer, avg_confidence

    def simulate_adaptive_stopping(
        self,
        threshold: float,
        confidence_method: str = 'mean_group_confidence',
        step_size: int = 1
    ) -> Dict:
        """
        Simulate adaptive stopping with given threshold.

        Args:
            threshold: Confidence threshold for early stopping
            confidence_method: Which confidence metric to use
            step_size: How many samples to add each step

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

            for n in range(step_size, len(problem_data) + 1, step_size):
                # Get first n samples
                current_samples = problem_data.head(n)

                # Extract answers and confidences
                answers = current_samples['final_answer'].tolist()
                confidences = current_samples[confidence_method].tolist()

                # Calculate majority and confidence
                majority_answer, confidence = self.calculate_majority_confidence(
                    answers, confidences, confidence_method
                )

                # Check stopping condition
                if confidence >= threshold:
                    stopped_at = n
                    final_answer = majority_answer
                    final_confidence = confidence
                    break

            # If didn't stop early, use all samples
            if stopped_at is None:
                stopped_at = len(problem_data)
                answers = problem_data['final_answer'].tolist()
                confidences = problem_data[confidence_method].tolist()
                final_answer, final_confidence = self.calculate_majority_confidence(
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
                'ground_truth': ground_truth,
                'predicted_answer': final_answer
            })

        return {
            'threshold': threshold,
            'confidence_method': confidence_method,
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
        step_size: int = 1
    ) -> List[Dict]:
        """
        Run experiment with multiple threshold values.

        Args:
            thresholds: List of threshold values to test
            confidence_method: Which confidence metric to use
            step_size: How many samples to add each step

        Returns:
            List of results for each threshold
        """
        all_results = []

        for threshold in thresholds:
            print(f"\nRunning experiment with threshold={threshold:.3f}...")
            result = self.simulate_adaptive_stopping(
                threshold, confidence_method, step_size
            )
            all_results.append(result)

            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Avg samples: {result['avg_samples']:.2f}/{self.max_samples}")
            print(f"  Avg token ratio: {result['avg_token_ratio']:.4f}")

        # Add baseline (no early stopping)
        print("\nRunning baseline (no early stopping)...")
        baseline = self._calculate_baseline(confidence_method)
        all_results.append(baseline)
        print(f"  Accuracy: {baseline['accuracy']:.4f}")
        print(f"  Avg samples: {baseline['avg_samples']:.2f}/{self.max_samples}")
        print(f"  Avg token ratio: {baseline['avg_token_ratio']:.4f}")

        return all_results

    def _calculate_baseline(self, confidence_method: str) -> Dict:
        """Calculate baseline results (always use all samples)."""
        results = []

        for problem_id, problem_data in self.problems.items():
            ground_truth = problem_data['ground_truth'].iloc[0]

            # Use all samples
            answers = problem_data['final_answer'].tolist()
            confidences = problem_data[confidence_method].tolist()
            final_answer, final_confidence = self.calculate_majority_confidence(
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
                'ground_truth': ground_truth,
                'predicted_answer': final_answer
            })

        return {
            'threshold': float('inf'),  # No threshold (baseline)
            'confidence_method': confidence_method,
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

        # Extract data
        thresholds = [r['threshold'] for r in results if not r.get('is_baseline', False)]
        accuracies = [r['accuracy'] for r in results if not r.get('is_baseline', False)]
        token_ratios = [r['avg_token_ratio'] for r in results if not r.get('is_baseline', False)]
        avg_samples = [r['avg_samples'] for r in results if not r.get('is_baseline', False)]

        # Get baseline
        baseline = [r for r in results if r.get('is_baseline', False)][0]
        baseline_acc = baseline['accuracy']
        baseline_tokens = 1.0

        # Plot 1: Accuracy vs Token Ratio (Pareto Efficiency)
        ax1 = axes[0, 0]
        ax1.scatter(token_ratios, accuracies, s=100, alpha=0.6, c=thresholds, cmap='viridis')
        ax1.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline Accuracy: {baseline_acc:.4f}')
        ax1.axvline(x=baseline_tokens, color='r', linestyle='--', alpha=0.5)
        ax1.scatter([baseline_tokens], [baseline_acc], s=200, color='red', marker='*',
                   label='Baseline (No Early Stop)', zorder=5)
        ax1.set_xlabel('Avg Token Usage Ratio', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Pareto Efficiency: Accuracy vs Cost', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Threshold vs Accuracy
        ax2 = axes[0, 1]
        ax2.plot(thresholds, accuracies, 'o-', linewidth=2, markersize=8)
        ax2.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline: {baseline_acc:.4f}')
        ax2.set_xlabel('Confidence Threshold', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Threshold vs Token Savings
        ax3 = axes[1, 0]
        token_savings = [(1 - ratio) * 100 for ratio in token_ratios]
        ax3.plot(thresholds, token_savings, 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Confidence Threshold', fontsize=12)
        ax3.set_ylabel('Token Savings (%)', fontsize=12)
        ax3.set_title('Token Cost Reduction', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Threshold vs Avg Samples
        ax4 = axes[1, 1]
        ax4.plot(thresholds, avg_samples, 'o-', linewidth=2, markersize=8, color='purple')
        ax4.axhline(y=self.max_samples, color='r', linestyle='--',
                   label=f'Max Samples: {self.max_samples}')
        ax4.set_xlabel('Confidence Threshold', fontsize=12)
        ax4.set_ylabel('Avg Samples Used', fontsize=12)
        ax4.set_title('Average Inference Count', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

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
            summary_data.append({
                'Threshold': r['threshold'] if not r.get('is_baseline') else 'Baseline',
                'Accuracy': f"{r['accuracy']:.4f}",
                'Avg Samples': f"{r['avg_samples']:.2f}",
                'Token Ratio': f"{r['avg_token_ratio']:.4f}",
                'Token Savings (%)': f"{(1 - r['avg_token_ratio']) * 100:.2f}%",
                'Accuracy Drop': f"{(baseline['accuracy'] - r['accuracy']) * 100:.2f}%"
                    if not r.get('is_baseline') else "0.00%"
            })

        df = pd.DataFrame(summary_data)
        return df


def main():
    """Main experiment runner."""
    # Configuration
    DATA_PATH = "/mnt/data1/projects/Conf_Agg/output_s/generated/dataset_4000_think.parquet"
    OUTPUT_DIR = Path("/mnt/data1/projects/Conf_Agg/experiments")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Experiment parameters
    THRESHOLDS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    CONFIDENCE_METHOD = 'mean_group_confidence'  # or 'bottom_10_percent_confidence', 'tail_confidence'
    STEP_SIZE = 1  # Add 1 sample at a time

    print("="*80)
    print("Adaptive Stopping Experiment with Confidence-based Early Termination")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data: {DATA_PATH}")
    print(f"  Confidence Method: {CONFIDENCE_METHOD}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Step Size: {STEP_SIZE}")
    print()

    # Initialize experiment
    exp = AdaptiveStoppingExperiment(DATA_PATH, max_samples=32)

    # Run threshold sweep
    results = exp.run_threshold_sweep(
        thresholds=THRESHOLDS,
        confidence_method=CONFIDENCE_METHOD,
        step_size=STEP_SIZE
    )

    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    plot_path = OUTPUT_DIR / "adaptive_stopping_analysis.png"
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
    summary_path = OUTPUT_DIR / "adaptive_stopping_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Save detailed results
    results_path = OUTPUT_DIR / "adaptive_stopping_detailed.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_path}")

    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
