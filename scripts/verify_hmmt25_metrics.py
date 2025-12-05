"""
HMMT25 메트릭 검증 스크립트
pass@1, majority_voting_set_8, confidence_bottom_10_percent_confidence_set_8 재계산
"""
import json
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.math_verifier import MathVerifier

def create_sequential_sets(num_solutions: int, set_size: int = 8, max_groups: int = None) -> list:
    """순차 분할 셋 생성"""
    num_sets = num_solutions // set_size
    if max_groups is not None:
        num_sets = min(num_sets, max_groups)

    sequential_sets = []
    for i in range(num_sets):
        start_idx = i * set_size
        end_idx = start_idx + set_size
        sequential_sets.append(list(range(start_idx, end_idx)))

    return sequential_sets

def create_sampled_sets(solutions: list, k: int, num_sets: int, seed: int = 42) -> list:
    """Pass@k용 샘플링 셋 생성"""
    np.random.seed(seed)
    total_solutions = len(solutions)

    sampled_sets = []
    for _ in range(num_sets):
        sampled_indices = np.random.choice(total_solutions, size=k, replace=False)
        sampled_sets.append(sampled_indices.tolist())

    return sampled_sets

# Load data
data_path = '/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/MathArena_hmmt_feb_2025_baseline_generated.json'
with open(data_path, 'r') as f:
    data = json.load(f)

# Initialize math verifier
math_verifier = MathVerifier(timeout=30)

problems = data['generated_solutions']
print(f"Total problems: {len(problems)}\n")

# Metrics accumulators
pass_at_1_per_problem = []
majority_voting_set_8_per_problem = []
confidence_voting_set_8_per_problem = []

for idx, problem in enumerate(problems):
    solutions = problem['solutions']
    ground_truth = problem['ground_truth']

    print(f"Problem {idx + 1}:")
    print(f"  Ground truth: {ground_truth}")
    print(f"  Total solutions: {len(solutions)}")

    # 1. Pass@1 calculation (with sampling)
    sampled_sets = create_sampled_sets(solutions, k=1, num_sets=64, seed=42 + idx)
    correct_count = 0
    for sampled_indices in sampled_sets:
        sol = solutions[sampled_indices[0]]
        final_answer = sol.get('final_answer')
        if final_answer and math_verifier.verify_answer(final_answer, ground_truth):
            correct_count += 1
    pass_at_1 = correct_count / 64
    pass_at_1_per_problem.append(pass_at_1)
    print(f"  Pass@1: {pass_at_1:.6f} ({correct_count}/64)")

    # 2. Majority Voting set_8
    sequential_sets = create_sequential_sets(len(solutions), set_size=8, max_groups=32)
    print(f"  Sequential sets (size 8): {len(sequential_sets)} groups")

    majority_correct = 0
    for set_indices in sequential_sets:
        answers = []
        for idx_sol in set_indices:
            sol = solutions[idx_sol]
            if sol.get('final_answer'):
                answers.append(sol['final_answer'])

        if answers:
            answer_counts = defaultdict(int)
            for answer in answers:
                answer_counts[answer] += 1
            majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
            if math_verifier.verify_answer(majority_answer, ground_truth):
                majority_correct += 1

    majority_acc = majority_correct / len(sequential_sets) if len(sequential_sets) > 0 else 0.0
    majority_voting_set_8_per_problem.append(majority_acc)
    print(f"  Majority Voting (set 8): {majority_acc:.6f} ({majority_correct}/{len(sequential_sets)})")

    # 3. Confidence Weighted Voting set_8
    confidence_correct = 0
    for set_indices in sequential_sets:
        weighted_votes = defaultdict(float)
        for idx_sol in set_indices:
            sol = solutions[idx_sol]
            answer = sol.get('final_answer')
            if not answer:
                continue
            conf = sol.get('confidence_scores', {}).get('bottom_10_percent_confidence', 0.0)
            weighted_votes[answer] += conf

        if weighted_votes:
            best_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
            if math_verifier.verify_answer(best_answer, ground_truth):
                confidence_correct += 1

    confidence_acc = confidence_correct / len(sequential_sets) if len(sequential_sets) > 0 else 0.0
    confidence_voting_set_8_per_problem.append(confidence_acc)
    print(f"  Confidence Voting (set 8): {confidence_acc:.6f} ({confidence_correct}/{len(sequential_sets)})")
    print()

# Calculate averages
print("=" * 60)
print("OVERALL METRICS:")
print("=" * 60)
print(f"Pass@1: {np.mean(pass_at_1_per_problem):.6f}")
print(f"Majority Voting (set 8): {np.mean(majority_voting_set_8_per_problem):.6f}")
print(f"Confidence Voting (set 8): {np.mean(confidence_voting_set_8_per_problem):.6f}")

print("\n" + "=" * 60)
print("EXPECTED FROM METRICS FILE:")
print("=" * 60)
with open('/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/MathArena_hmmt_feb_2025_metrics.json', 'r') as f:
    metrics = json.load(f)
print(f"pass_at_1: {metrics['baseline']['pass_at_1']}")
print(f"majority_voting_set_8: {metrics['baseline']['majority_voting_set_8']}")
print(f"confidence_bottom_10_percent_confidence_set_8: {metrics['baseline']['confidence_bottom_10_percent_confidence_set_8']}")
