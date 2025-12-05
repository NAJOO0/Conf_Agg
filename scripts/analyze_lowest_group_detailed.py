"""
Detailed analysis of Lowest Group Confidence for adaptive stopping.

This script performs in-depth analysis of why Lowest Group Confidence
performs best and provides insights for production deployment.
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class LowestGroupAnalyzer:
    """Deep analysis of Lowest Group Confidence strategy."""

    def __init__(self, data_path: str, max_samples: int = 32):
        print(f"Loading dataset from {data_path}...")
        self.df = pd.read_parquet(data_path)
        self.max_samples = max_samples
        self.problems = self._group_by_problem()
        print(f"Found {len(self.problems)} unique problems")

    def _group_by_problem(self) -> Dict[str, pd.DataFrame]:
        """Group responses by problem_id."""
        problems = {}
        for problem_id, group in self.df.groupby('problem_id'):
            group = group.head(self.max_samples).reset_index(drop=True)
            problems[problem_id] = group
        return problems

    def analyze_threshold_sensitivity(self, thresholds: List[float]) -> pd.DataFrame:
        """Analyze performance across different thresholds."""
        results = []

        for threshold in thresholds:
            metrics = self._calculate_metrics(threshold)
            results.append(metrics)

        return pd.DataFrame(results)

    def _calculate_metrics(self, threshold: float) -> Dict:
        """Calculate detailed metrics for a given threshold."""
        correct_count = 0
        total_samples_used = 0
        total_tokens_used = 0
        total_tokens_available = 0
        early_stop_count = 0
        problem_results = []

        for problem_id, problem_data in self.problems.items():
            ground_truth = problem_data['ground_truth'].iloc[0]

            # Simulate sequential inference
            stopped_at = None
            final_answer = None

            for n in range(2, len(problem_data) + 1):
                current_samples = problem_data.head(n)
                answers = current_samples['final_answer'].tolist()
                confidences = current_samples['lowest_group_confidence'].tolist()

                # Majority voting
                answer_counts = Counter(answers)
                majority_answer = answer_counts.most_common(1)[0][0]

                # Average confidence of majority answer
                majority_confidences = [
                    conf for ans, conf in zip(answers, confidences)
                    if ans == majority_answer
                ]
                avg_confidence = np.mean(majority_confidences) if majority_confidences else 0

                if avg_confidence >= threshold:
                    stopped_at = n
                    final_answer = majority_answer
                    break

            if stopped_at is None:
                stopped_at = len(problem_data)
                final_answer = problem_data['final_answer'].iloc[-1]

            is_correct = (final_answer == ground_truth)
            if is_correct:
                correct_count += 1

            tokens_used = problem_data.head(stopped_at)['output_token_count'].sum()
            tokens_available = problem_data['output_token_count'].sum()

            total_samples_used += stopped_at
            total_tokens_used += tokens_used
            total_tokens_available += tokens_available

            if stopped_at < len(problem_data):
                early_stop_count += 1

            problem_results.append({
                'problem_id': problem_id,
                'stopped_at': stopped_at,
                'is_correct': is_correct,
                'tokens_used': tokens_used
            })

        return {
            'threshold': threshold,
            'accuracy': correct_count / len(self.problems),
            'avg_samples': total_samples_used / len(self.problems),
            'token_ratio': total_tokens_used / total_tokens_available,
            'token_savings_pct': (1 - total_tokens_used / total_tokens_available) * 100,
            'early_stop_rate': early_stop_count / len(self.problems) * 100,
            'problem_results': problem_results
        }

    def analyze_problem_characteristics(self, threshold: float = 8.0) -> pd.DataFrame:
        """Analyze characteristics of problems that benefit from early stopping."""
        problem_analysis = []

        for problem_id, problem_data in self.problems.items():
            ground_truth = problem_data['ground_truth'].iloc[0]

            # Baseline (all samples)
            baseline_answers = problem_data['final_answer'].tolist()
            baseline_majority = Counter(baseline_answers).most_common(1)[0][0]
            baseline_correct = (baseline_majority == ground_truth)

            # With early stopping
            stopped_at = None
            for n in range(2, len(problem_data) + 1):
                current_samples = problem_data.head(n)
                answers = current_samples['final_answer'].tolist()
                confidences = current_samples['lowest_group_confidence'].tolist()

                answer_counts = Counter(answers)
                majority_answer = answer_counts.most_common(1)[0][0]
                majority_confidences = [
                    conf for ans, conf in zip(answers, confidences)
                    if ans == majority_answer
                ]
                avg_confidence = np.mean(majority_confidences) if majority_confidences else 0

                if avg_confidence >= threshold:
                    stopped_at = n
                    break

            if stopped_at is None:
                stopped_at = len(problem_data)

            early_stopped = (stopped_at < len(problem_data))
            early_answer = problem_data.head(stopped_at)['final_answer'].mode()[0]
            early_correct = (early_answer == ground_truth)

            # Calculate confidence statistics
            avg_lowest_conf = problem_data['lowest_group_confidence'].mean()
            std_lowest_conf = problem_data['lowest_group_confidence'].std()
            max_lowest_conf = problem_data['lowest_group_confidence'].max()
            min_lowest_conf = problem_data['lowest_group_confidence'].min()

            # Calculate answer consistency
            unique_answers = len(set(baseline_answers))
            answer_diversity = unique_answers / len(baseline_answers)

            problem_analysis.append({
                'problem_id': problem_id,
                'baseline_correct': baseline_correct,
                'early_correct': early_correct,
                'early_stopped': early_stopped,
                'stopped_at': stopped_at,
                'samples_saved': len(problem_data) - stopped_at,
                'savings_pct': (1 - stopped_at / len(problem_data)) * 100,
                'avg_lowest_conf': avg_lowest_conf,
                'std_lowest_conf': std_lowest_conf,
                'max_lowest_conf': max_lowest_conf,
                'min_lowest_conf': min_lowest_conf,
                'answer_diversity': answer_diversity,
                'result_category': self._categorize_result(baseline_correct, early_correct, early_stopped)
            })

        return pd.DataFrame(problem_analysis)

    def _categorize_result(self, baseline_correct: bool, early_correct: bool, early_stopped: bool) -> str:
        """Categorize problem result."""
        if not early_stopped:
            return 'No Early Stop'
        elif early_correct and baseline_correct:
            return 'Win: Saved Cost, Kept Accuracy'
        elif early_correct and not baseline_correct:
            return 'Super Win: Saved Cost, Improved Accuracy'
        elif not early_correct and baseline_correct:
            return 'Loss: Saved Cost, Lost Accuracy'
        else:
            return 'Neutral: Both Wrong'

    def create_visualizations(self, threshold_df: pd.DataFrame, problem_df: pd.DataFrame, output_dir: Path):
        """Create comprehensive visualizations."""

        # Figure 1: Threshold sensitivity analysis
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Accuracy vs Threshold
        ax1 = axes1[0, 0]
        ax1.plot(threshold_df['threshold'], threshold_df['accuracy'] * 100,
                'o-', linewidth=2.5, markersize=10, color='#2E86AB')
        baseline_acc = threshold_df[threshold_df['threshold'] == threshold_df['threshold'].max()]['accuracy'].iloc[0]
        ax1.axhline(y=baseline_acc * 100, color='red', linestyle='--',
                   label=f'Baseline: {baseline_acc*100:.2f}%', linewidth=2)
        ax1.fill_between(threshold_df['threshold'], threshold_df['accuracy'] * 100,
                        baseline_acc * 100, alpha=0.2, color='#2E86AB')
        ax1.set_xlabel('Threshold', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Accuracy vs Threshold', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Token Savings vs Threshold
        ax2 = axes1[0, 1]
        ax2.plot(threshold_df['threshold'], threshold_df['token_savings_pct'],
                'o-', linewidth=2.5, markersize=10, color='#06A77D')
        ax2.fill_between(threshold_df['threshold'], threshold_df['token_savings_pct'],
                        alpha=0.3, color='#06A77D')
        ax2.set_xlabel('Threshold', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Token Savings (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Cost Reduction vs Threshold', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Avg Samples vs Threshold
        ax3 = axes1[1, 0]
        ax3.plot(threshold_df['threshold'], threshold_df['avg_samples'],
                'o-', linewidth=2.5, markersize=10, color='#A23B72')
        ax3.axhline(y=32, color='red', linestyle='--', label='Max: 32', linewidth=2)
        ax3.fill_between(threshold_df['threshold'], threshold_df['avg_samples'],
                        alpha=0.3, color='#A23B72')
        ax3.set_xlabel('Threshold', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Avg Samples Used', fontsize=13, fontweight='bold')
        ax3.set_title('Inference Count vs Threshold', fontsize=15, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Early Stop Rate vs Threshold
        ax4 = axes1[1, 1]
        ax4.plot(threshold_df['threshold'], threshold_df['early_stop_rate'],
                'o-', linewidth=2.5, markersize=10, color='#F18F01')
        ax4.fill_between(threshold_df['threshold'], threshold_df['early_stop_rate'],
                        alpha=0.3, color='#F18F01')
        ax4.set_xlabel('Threshold', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Early Stop Rate (%)', fontsize=13, fontweight='bold')
        ax4.set_title('Early Stopping Frequency', fontsize=15, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig1.savefig(output_dir / 'lowest_group_threshold_analysis.png',
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'lowest_group_threshold_analysis.png'}")
        plt.close()

        # Figure 2: Problem characteristics analysis
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Result categories distribution
        ax1 = axes2[0, 0]
        category_counts = problem_df['result_category'].value_counts()
        colors_cat = ['#06A77D', '#2E86AB', '#F18F01', '#A23B72', '#999999']
        wedges, texts, autotexts = ax1.pie(category_counts.values, labels=category_counts.index,
                                           autopct='%1.1f%%', startangle=90, colors=colors_cat)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        ax1.set_title('Problem Outcome Distribution\n(Threshold=8.0)', fontsize=15, fontweight='bold')

        # Plot 2: Samples saved distribution
        ax2 = axes2[0, 1]
        early_stopped = problem_df[problem_df['early_stopped']]
        ax2.hist(early_stopped['samples_saved'], bins=30, color='#06A77D',
                alpha=0.7, edgecolor='black', linewidth=1.2)
        ax2.axvline(x=early_stopped['samples_saved'].mean(), color='red',
                   linestyle='--', linewidth=2, label=f'Mean: {early_stopped["samples_saved"].mean():.1f}')
        ax2.set_xlabel('Samples Saved', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        ax2.set_title('Distribution of Samples Saved\n(Early Stopped Problems)',
                     fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Confidence vs Early Stopping
        ax3 = axes2[1, 0]
        early_yes = problem_df[problem_df['early_stopped']]
        early_no = problem_df[~problem_df['early_stopped']]

        positions = [1, 2]
        box_data = [early_yes['avg_lowest_conf'], early_no['avg_lowest_conf']]
        bp = ax3.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True)

        colors_box = ['#06A77D', '#F18F01']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_xticklabels(['Early Stopped', 'No Early Stop'])
        ax3.set_ylabel('Avg Lowest Group Confidence', fontsize=13, fontweight='bold')
        ax3.set_title('Confidence Distribution by Early Stop Status', fontsize=15, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Scatter: Confidence vs Answer Diversity
        ax4 = axes2[1, 1]
        colors_scatter = ['#06A77D' if cat == 'Win: Saved Cost, Kept Accuracy'
                         else '#F18F01' if cat == 'Loss: Saved Cost, Lost Accuracy'
                         else '#2E86AB' for cat in problem_df['result_category']]

        ax4.scatter(problem_df['answer_diversity'], problem_df['avg_lowest_conf'],
                   c=colors_scatter, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Answer Diversity', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Avg Lowest Confidence', fontsize=13, fontweight='bold')
        ax4.set_title('Confidence vs Answer Consistency', fontsize=15, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#06A77D', label='Win'),
            Patch(facecolor='#F18F01', label='Loss'),
            Patch(facecolor='#2E86AB', label='Other')
        ]
        ax4.legend(handles=legend_elements, loc='best', fontsize=10)

        plt.tight_layout()
        fig2.savefig(output_dir / 'lowest_group_problem_analysis.png',
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'lowest_group_problem_analysis.png'}")
        plt.close()


def main():
    DATA_PATH = "/mnt/data1/projects/Conf_Agg/output_s/generated/dataset_4000_think.parquet"
    OUTPUT_DIR = Path("/mnt/data1/projects/Conf_Agg/experiments/lowest_group_analysis")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("Lowest Group Confidence - Deep Dive Analysis")
    print("="*80)

    analyzer = LowestGroupAnalyzer(DATA_PATH, max_samples=32)

    # Analyze threshold sensitivity
    print("\n1. Analyzing threshold sensitivity...")
    thresholds = [6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0]
    threshold_df = analyzer.analyze_threshold_sensitivity(thresholds)

    print("\nThreshold Sensitivity Results:")
    print(threshold_df.to_string(index=False))
    threshold_df.to_csv(OUTPUT_DIR / 'threshold_sensitivity.csv', index=False)

    # Analyze problem characteristics
    print("\n2. Analyzing problem characteristics (Threshold=8.0)...")
    problem_df = analyzer.analyze_problem_characteristics(threshold=8.0)

    print("\nProblem Analysis Summary:")
    print(problem_df['result_category'].value_counts())
    problem_df.to_csv(OUTPUT_DIR / 'problem_characteristics.csv', index=False)

    # Create visualizations
    print("\n3. Creating visualizations...")
    analyzer.create_visualizations(threshold_df, problem_df, OUTPUT_DIR)

    # Generate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    optimal_threshold = threshold_df.loc[threshold_df['accuracy'].idxmax()]
    print(f"\nOptimal Threshold: {optimal_threshold['threshold']}")
    print(f"  Accuracy: {optimal_threshold['accuracy']*100:.2f}%")
    print(f"  Token Savings: {optimal_threshold['token_savings_pct']:.1f}%")
    print(f"  Avg Samples: {optimal_threshold['avg_samples']:.1f}")
    print(f"  Early Stop Rate: {optimal_threshold['early_stop_rate']:.1f}%")

    # Best efficiency
    threshold_df['efficiency'] = threshold_df['accuracy'] / threshold_df['token_ratio']
    best_efficiency = threshold_df.loc[threshold_df['efficiency'].idxmax()]
    print(f"\nBest Efficiency Threshold: {best_efficiency['threshold']}")
    print(f"  Accuracy: {best_efficiency['accuracy']*100:.2f}%")
    print(f"  Token Savings: {best_efficiency['token_savings_pct']:.1f}%")
    print(f"  Efficiency Score: {best_efficiency['efficiency']:.2f}")

    # Problem category breakdown
    print("\n" + "="*80)
    print("PROBLEM OUTCOME BREAKDOWN (Threshold=8.0)")
    print("="*80)
    for category, count in problem_df['result_category'].value_counts().items():
        pct = count / len(problem_df) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")

    # Win rate analysis
    wins = problem_df[problem_df['result_category'].str.contains('Win')]
    if len(wins) > 0:
        print(f"\nWin Scenarios Statistics:")
        print(f"  Average samples saved: {wins['samples_saved'].mean():.1f}")
        print(f"  Average savings: {wins['savings_pct'].mean():.1f}%")
        print(f"  Average confidence: {wins['avg_lowest_conf'].mean():.2f}")

    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
