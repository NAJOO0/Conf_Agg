"""
모든 벤치마크의 메트릭 검증 스크립트
AIME24, AIME25, HMMT24, HMMT25의 pass@1, majority_voting_set_8, confidence_bottom_10_percent_confidence_set_8 검증
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

def verify_benchmark(dataset_name: str, dataset_path: str, base_dir: str, math_verifier: MathVerifier):
    """단일 벤치마크 검증"""
    dataset_safe_name = dataset_path.replace('/', '_')

    # Load generated data
    data_path = f"{base_dir}/{dataset_safe_name}_baseline_generated.json"
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Load metrics file
    metrics_path = f"{base_dir}/{dataset_safe_name}_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    problems = data['generated_solutions']

    # Metrics accumulators
    pass_at_1_per_problem = []
    majority_voting_set_8_per_problem = []
    confidence_voting_set_8_per_problem = []

    # Track differences
    diff_problems = []

    for idx, problem in enumerate(problems):
        solutions = problem['solutions']
        ground_truth = problem['ground_truth']

        # 1. Pass@1 calculation (with sampling)
        # For 64 solutions: 64 sets, for 128 solutions: 64 sets
        num_sets = 64 if len(solutions) >= 64 else 32
        sampled_sets = create_sampled_sets(solutions, k=1, num_sets=num_sets, seed=42 + idx)
        correct_count = 0
        for sampled_indices in sampled_sets:
            sol = solutions[sampled_indices[0]]
            final_answer = sol.get('final_answer')
            if final_answer and math_verifier.verify_answer(final_answer, ground_truth):
                correct_count += 1
        pass_at_1 = correct_count / num_sets
        pass_at_1_per_problem.append(pass_at_1)

        # 2. Majority Voting set_8
        sequential_sets = create_sequential_sets(len(solutions), set_size=8, max_groups=32)

        majority_correct = 0
        confidence_correct = 0

        for set_indices in sequential_sets:
            # Majority voting
            answers = []
            for idx_sol in set_indices:
                sol = solutions[idx_sol]
                if sol.get('final_answer'):
                    answers.append(sol['final_answer'])

            majority_answer = None
            if answers:
                answer_counts = defaultdict(int)
                for answer in answers:
                    answer_counts[answer] += 1
                majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                if math_verifier.verify_answer(majority_answer, ground_truth):
                    majority_correct += 1

            # Confidence Weighted Voting set_8
            weighted_votes = defaultdict(float)
            for idx_sol in set_indices:
                sol = solutions[idx_sol]
                answer = sol.get('final_answer')
                if not answer:
                    continue
                conf = sol.get('confidence_scores', {}).get('bottom_10_percent_confidence', 0.0)
                weighted_votes[answer] += conf

            best_answer = None
            if weighted_votes:
                best_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
                if math_verifier.verify_answer(best_answer, ground_truth):
                    confidence_correct += 1

            # Track if they differ
            if majority_answer != best_answer:
                maj_correct = math_verifier.verify_answer(majority_answer, ground_truth) if majority_answer else False
                conf_correct = math_verifier.verify_answer(best_answer, ground_truth) if best_answer else False
                if maj_correct != conf_correct:
                    diff_problems.append((idx + 1, set_indices, majority_answer, best_answer))

        majority_acc = majority_correct / len(sequential_sets) if len(sequential_sets) > 0 else 0.0
        confidence_acc = confidence_correct / len(sequential_sets) if len(sequential_sets) > 0 else 0.0

        majority_voting_set_8_per_problem.append(majority_acc)
        confidence_voting_set_8_per_problem.append(confidence_acc)

    # Calculate averages
    calculated_pass_at_1 = np.mean(pass_at_1_per_problem)
    calculated_majority = np.mean(majority_voting_set_8_per_problem)
    calculated_confidence = np.mean(confidence_voting_set_8_per_problem)

    # Expected values from metrics file
    expected_pass_at_1 = metrics['baseline']['pass_at_1']
    expected_majority = metrics['baseline']['majority_voting_set_8']
    expected_confidence = metrics['baseline']['confidence_bottom_10_percent_confidence_set_8']

    return {
        'dataset_name': dataset_name,
        'total_problems': len(problems),
        'calculated': {
            'pass_at_1': calculated_pass_at_1,
            'majority_voting_set_8': calculated_majority,
            'confidence_voting_set_8': calculated_confidence
        },
        'expected': {
            'pass_at_1': expected_pass_at_1,
            'majority_voting_set_8': expected_majority,
            'confidence_voting_set_8': expected_confidence
        },
        'diff_count': len([p for p, _, _, _ in diff_problems]),
        'diff_problems': diff_problems
    }

# Initialize math verifier
math_verifier = MathVerifier(timeout=30)

base_dir = "/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768"

benchmarks = [
    {"name": "AIME24", "path": "math-ai/aime24"},
    {"name": "AIME25", "path": "math-ai/aime25"},
    {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
    {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
]

print("=" * 80)
print("벤치마크 메트릭 검증")
print("=" * 80)
print()

all_results = []

for benchmark in benchmarks:
    print(f"\n{'=' * 80}")
    print(f"검증 중: {benchmark['name']} ({benchmark['path']})")
    print(f"{'=' * 80}")

    result = verify_benchmark(benchmark['name'], benchmark['path'], base_dir, math_verifier)
    all_results.append(result)

    print(f"\n총 문제 수: {result['total_problems']}")
    print(f"\n계산된 값:")
    print(f"  Pass@1: {result['calculated']['pass_at_1']:.6f}")
    print(f"  Majority Voting (set 8): {result['calculated']['majority_voting_set_8']:.6f}")
    print(f"  Confidence Voting (set 8): {result['calculated']['confidence_voting_set_8']:.6f}")

    print(f"\n메트릭 파일 값:")
    print(f"  Pass@1: {result['expected']['pass_at_1']:.6f}")
    print(f"  Majority Voting (set 8): {result['expected']['majority_voting_set_8']:.6f}")
    print(f"  Confidence Voting (set 8): {result['expected']['confidence_voting_set_8']:.6f}")

    print(f"\n차이:")
    pass_diff = abs(result['calculated']['pass_at_1'] - result['expected']['pass_at_1'])
    maj_diff = abs(result['calculated']['majority_voting_set_8'] - result['expected']['majority_voting_set_8'])
    conf_diff = abs(result['calculated']['confidence_voting_set_8'] - result['expected']['confidence_voting_set_8'])

    print(f"  Pass@1: {pass_diff:.6f} {'✅' if pass_diff < 0.001 else '⚠️'}")
    print(f"  Majority Voting: {maj_diff:.6f} {'✅' if maj_diff < 0.001 else '⚠️'}")
    print(f"  Confidence Voting: {conf_diff:.6f} {'✅' if conf_diff < 0.001 else '⚠️'}")

    if result['diff_count'] > 0:
        print(f"\n⚠️  Majority와 Confidence가 다른 그룹: {result['diff_count']}개")

print("\n" + "=" * 80)
print("전체 요약")
print("=" * 80)

for result in all_results:
    print(f"\n{result['dataset_name']}:")
    maj_val = result['calculated']['majority_voting_set_8']
    conf_val = result['calculated']['confidence_voting_set_8']

    if abs(maj_val - conf_val) < 0.0001:
        print(f"  ⚠️  Majority와 Confidence가 같은 값: {maj_val:.6f}")
        print(f"      이는 우연히 같아진 것입니다 (평균 계산 결과)")
    else:
        print(f"  ✅ Majority: {maj_val:.6f}, Confidence: {conf_val:.6f} (다름)")
        print(f"      차이: {abs(maj_val - conf_val):.6f}")
