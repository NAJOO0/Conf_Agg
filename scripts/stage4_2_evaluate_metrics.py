"""
Stage 4-2: 저장된 Solution 결과로 메트릭 평가
Pass@k, Majority Voting, Confidence Weighted Voting 계산

주요 변경사항:
1. Pass@k: 샘플링 기반 (겹침 허용)
   - 128개: Pass@1~8(64셋), Pass@16(32셋), Pass@32(16셋)
   
2. Majority/Confidence Voting: 순차 분할 (겹침 없음)
   - 128개를 16개 셋으로 고정 분할 (각 8개)
   - 64개를 8개 셋으로 고정 분할 (각 8개)
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import json
from collections import defaultdict
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.math_verifier import MathVerifier
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_sampled_sets(solutions: list, k: int, num_sets: int, seed: int = 42) -> list:
    """
    미리 샘플링된 셋들을 생성 (Pass@k용)
    
    Args:
        solutions: 전체 solution 리스트
        k: 각 셋의 크기
        num_sets: 생성할 셋의 개수
        seed: 랜덤 시드
    
    Returns:
        샘플링된 셋들의 리스트 (각 셋은 k개의 solution 인덱스를 가짐)
    """
    np.random.seed(seed)
    total_solutions = len(solutions)
    
    sampled_sets = []
    for _ in range(num_sets):
        # k개를 랜덤 샘플링 (중복 없이)
        sampled_indices = np.random.choice(total_solutions, size=k, replace=False)
        sampled_sets.append(sampled_indices.tolist())
    
    return sampled_sets


def create_sequential_sets(num_solutions: int, set_size: int = 8) -> list:
    """
    순차 분할 셋 생성 (Majority/Confidence Voting용)
    
    Args:
        num_solutions: 전체 solution 개수
        set_size: 각 셋의 크기 (기본 8)
    
    Returns:
        순차 분할된 셋들의 리스트
        예: 128개 → [[0,1,2,3,4,5,6,7], [8,9,...,15], ..., [120,...,127]]
    """
    num_sets = num_solutions // set_size
    sequential_sets = []
    
    for i in range(num_sets):
        start_idx = i * set_size
        end_idx = start_idx + set_size
        sequential_sets.append(list(range(start_idx, end_idx)))
    
    return sequential_sets


def create_sequential_sets_for_sizes(num_solutions: int, set_sizes: list) -> dict:
    """
    다양한 set size에 대해 순차 분할 셋 생성
    
    Args:
        num_solutions: 전체 solution 개수
        set_sizes: 생성할 set size 리스트
    
    Returns:
        set_size별 순차 셋 딕셔너리
    """
    sequential_sets_dict = {}
    for set_size in set_sizes:
        if num_solutions < set_size:
            continue
        sequential_sets = create_sequential_sets(num_solutions, set_size)
        if sequential_sets:
            sequential_sets_dict[set_size] = sequential_sets
    return sequential_sets_dict


def get_set_configurations(num_solutions: int) -> dict:
    """
    Solution 개수에 따른 Pass@k 셋 구성 결정
    
    Args:
        num_solutions: 전체 solution 개수
    
    Returns:
        k별 셋 개수 딕셔너리
        예: {1: 64, 4: 64, 8: 64, 16: 32, 32: 16}
    """
    # 기준: 128개일 때
    base_configs = {
        1: 64,
        4: 64,
        8: 64,
        16: 32,
        32: 16,
        64: 8
    }
    
    # Solution 개수가 64개 이하면 절반으로
    if num_solutions <= 64:
        configs = {k: max(1, v // 2) for k, v in base_configs.items() if k <= num_solutions}
    else:
        configs = {k: v for k, v in base_configs.items() if k <= num_solutions}
    
    return configs


def calculate_pass_at_k_with_sets(
    solutions: list,
    ground_truth: str,
    sampled_sets_dict: dict,
    math_verifier: MathVerifier
) -> dict:
    """
    Pass@k 메트릭 계산 (미리 생성된 샘플링 셋 사용)
    
    Args:
        solutions: 전체 solution 리스트
        ground_truth: 정답
        sampled_sets_dict: k별 미리 샘플링된 셋들 {k: [set1, set2, ...]}
        math_verifier: Math verifier
    
    Returns:
        k별 Pass@k 결과
    """
    results = {}
    
    for k, sampled_sets in sampled_sets_dict.items():
        correct_count = 0
        
        for sampled_indices in sampled_sets:
            # 이 셋에서 정답이 하나라도 있는지 확인
            has_correct = False
            for idx in sampled_indices:
                sol = solutions[idx]
                if sol.get("final_answer") and math_verifier.verify_answer(
                    sol["final_answer"], ground_truth
                ):
                    has_correct = True
                    break
            
            if has_correct:
                correct_count += 1
        
        # 정답이 있는 셋의 비율
        num_sets = len(sampled_sets)
        accuracy = correct_count / num_sets if num_sets > 0 else 0.0
        results[k] = accuracy
    
    return results


def majority_voting_sequential(
    solutions: list,
    ground_truth: str,
    sequential_sets_dict: dict,
    math_verifier: MathVerifier
) -> dict:
    """
    Majority Voting (순차 분할 셋 사용)
    
    Args:
        solutions: 전체 solution 리스트
        ground_truth: 정답
        sequential_sets: 순차 분할된 셋들
        math_verifier: Math verifier
    
    Returns:
        정확도와 통계
    """
    results = {}
    
    for set_size, sequential_sets in sequential_sets_dict.items():
        correct_count = 0
        for set_indices in sequential_sets:
            answers = []
            for idx in set_indices:
                sol = solutions[idx]
                if sol.get("final_answer"):
                    answers.append(sol["final_answer"])
            
            if not answers:
                continue
            
            answer_counts = defaultdict(int)
            for answer in answers:
                answer_counts[answer] += 1
            
            if answer_counts:
                majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                if math_verifier.verify_answer(majority_answer, ground_truth):
                    correct_count += 1
        
        num_sets = len(sequential_sets)
        results[set_size] = {
            "correct_count": correct_count,
            "total_sets": num_sets,
            "accuracy": correct_count / num_sets if num_sets > 0 else 0.0
        }
    
    return results


def confidence_weighted_voting_sequential(
    solutions: list,
    ground_truth: str,
    sequential_sets_dict: dict,
    confidence_metric: str,
    math_verifier: MathVerifier
) -> dict:
    """
    Confidence Weighted Voting (순차 분할 셋 사용)
    
    Args:
        solutions: 전체 solution 리스트
        ground_truth: 정답
        sequential_sets: 순차 분할된 셋들
        confidence_metric: 사용할 confidence 메트릭
        math_verifier: Math verifier
    
    Returns:
        정확도와 통계
    """
    results = {}
    
    for set_size, sequential_sets in sequential_sets_dict.items():
        correct_count = 0
        for set_indices in sequential_sets:
            weighted_votes = defaultdict(float)
            for idx in set_indices:
                sol = solutions[idx]
                answer = sol.get("final_answer")
                if not answer:
                    continue
                conf = sol.get("confidence_scores", {}).get(confidence_metric, 0.0)
                weighted_votes[answer] += conf
            
            if weighted_votes:
                best_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
                if math_verifier.verify_answer(best_answer, ground_truth):
                    correct_count += 1
        
        num_sets = len(sequential_sets)
        results[set_size] = {
            "correct_count": correct_count,
            "total_sets": num_sets,
            "accuracy": correct_count / num_sets if num_sets > 0 else 0.0
        }
    
    return results


def evaluate_solutions(
    solutions: list,
    ground_truth: str,
    math_verifier: MathVerifier,
    seed: int = 42
) -> dict:
    """
    Solution 리스트에 대해 모든 메트릭 계산
    
    평가 방식:
    1. Pass@k: 샘플링 기반 (겹침 허용)
    2. Majority/Confidence Voting: 순차 분할 (겹침 없음, 16개 셋)
    """
    results = {}
    num_solutions = len(solutions)
    
    # 1. Pass@k용 샘플링 셋 생성
    set_configs = get_set_configurations(num_solutions)
    logger.debug(f"Pass@k 셋 구성: {set_configs}")
    
    sampled_sets_dict = {}
    for k, num_sets in set_configs.items():
        sampled_sets = create_sampled_sets(solutions, k, num_sets, seed=seed)
        sampled_sets_dict[k] = sampled_sets
        logger.debug(f"  Pass@{k}: {num_sets}개 샘플링 셋 생성")
    
    # 2. Pass@k 계산 (샘플링 셋 사용)
    pass_at_k = calculate_pass_at_k_with_sets(
        solutions, ground_truth, sampled_sets_dict, math_verifier
    )
    results["pass_at_k"] = pass_at_k
    
    # 3. Majority/Confidence Voting용 순차 분할 셋 생성 (4,8,16,32,64)
    set_sizes = [4, 8, 16, 32, 64]
    sequential_sets_dict = create_sequential_sets_for_sizes(num_solutions, set_sizes)
    logger.debug(
        "Majority/Confidence Voting 순차 셋 구성: %s",
        {size: len(sets) for size, sets in sequential_sets_dict.items()}
    )
    
    # 4. Majority Voting (순차 분할 셋 사용)
    majority_result = majority_voting_sequential(
        solutions, ground_truth, sequential_sets_dict, math_verifier
    )
    results["majority_voting"] = majority_result
    
    # 5. Confidence Weighted Voting (순차 분할 셋 사용)
    confidence_metrics = [
        "bottom_10_percent_confidence",
        "tail_confidence",
        "mean_group_confidence",
        "lowest_group_confidence"
    ]
    
    confidence_results = {}
    for metric in confidence_metrics:
        voting_result = confidence_weighted_voting_sequential(
            solutions, ground_truth, sequential_sets_dict, metric, math_verifier
        )
        confidence_results[metric] = voting_result
    results["confidence_weighted_voting"] = confidence_results
    
    return results


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4-2: 메트릭 평가 메인 함수"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage4_2_evaluate_metrics.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 4-2: 메트릭 평가 시작")
    logger.info("📊 평가 방식:")
    logger.info("   - Pass@k: 샘플링 기반 (겹침 허용)")
    logger.info("     * 128개: Pass@1~8(64셋), Pass@16(32셋), Pass@32(16셋)")
    logger.info("     * 64개: Pass@1~8(32셋), Pass@16(16셋), Pass@32(8셋)")
    logger.info("   - Majority/Confidence Voting: 순차 분할 (겹침 없음)")
    logger.info("     * 128개 → 16개 셋 (각 8개)")
    logger.info("     * 64개 → 8개 셋 (각 8개)")
    
    # 디렉토리 설정
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    # results_dir = os.path.join(results_dir, "think_prev_3600")
    results_dir = os.path.join(results_dir, "Qwen_Qwen3-1.7B_think_True")
    # results_dir = os.path.join(results_dir, "Qwen_Qwen3-1.7B_think_False_temperature1.0")
    
    # Math Verifier 초기화
    math_verifier = MathVerifier(
        timeout=cfg.evaluation.benchmarks.evaluation.timeout
    )
    
    # 벤치마크 데이터셋 설정
    benchmark_datasets = [
        {"name": "AIME24", "path": "math-ai/aime24"},
        {"name": "AIME25", "path": "math-ai/aime25"},
        {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
        {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
    ]
    
    all_results = {}
    
    # 각 데이터셋에 대해 평가
    for benchmark in benchmark_datasets:
        dataset_name = benchmark["name"]
        dataset_path = benchmark["path"]
        dataset_safe_name = dataset_path.replace('/', '_')
        
        logger.info("=" * 60)
        logger.info(f"데이터셋 평가: {dataset_name}")
        logger.info("=" * 60)
        
        results = {
            "dataset_name": dataset_path,
            "total_problems": 0
        }
        
        # Baseline 평가
        baseline_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_baseline_generated.json"
        )
        
        if os.path.exists(baseline_path):
            logger.info(f"Baseline 결과 로드: {baseline_path}")
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            
            baseline_metrics = defaultdict(list)
            
            for idx, problem_data in enumerate(baseline_data["generated_solutions"]):
                solutions = problem_data["solutions"]
                ground_truth = problem_data["ground_truth"]
                
                # 각 문제마다 다른 시드 사용 (재현 가능성 유지)
                problem_results = evaluate_solutions(
                    solutions, ground_truth, math_verifier, seed=42 + idx
                )
                
                # Pass@k 메트릭 누적
                for k, v in problem_results.get("pass_at_k", {}).items():
                    baseline_metrics[f"pass_at_{k}"].append(v)
                
                # Majority Voting 메트릭
                for set_size, stats in problem_results.get("majority_voting", {}).items():
                    baseline_metrics[f"majority_voting_set_{set_size}"].append(
                        stats["accuracy"]
                    )
                
                # Confidence Weighted Voting 메트릭
                for metric, size_results in problem_results.get("confidence_weighted_voting", {}).items():
                    for set_size, stats in size_results.items():
                        baseline_metrics[f"confidence_{metric}_set_{set_size}"].append(
                            stats["accuracy"]
                        )
            
            # 평균 계산
            baseline_final = {}
            for key, values in baseline_metrics.items():
                baseline_final[key] = np.mean(values) if values else 0.0
            
            results["baseline"] = baseline_final
            results["total_problems"] = len(baseline_data["generated_solutions"])
            
            logger.info(f"Baseline 평가 완료: {len(baseline_data['generated_solutions'])}개 문제")
            # Pass@k 결과 출력
            pass_k_keys = sorted([int(k.replace('pass_at_', '')) for k in baseline_final.keys() if k.startswith('pass_at_')])
            for k in pass_k_keys:
                logger.info(f"  Pass@{k}: {baseline_final[f'pass_at_{k}']:.4f}")
            majority_keys = sorted(
                [
                    (int(key.split('_')[-1]), key)
                    for key in baseline_final.keys()
                    if key.startswith('majority_voting_set_')
                ]
            )
            for size, key in majority_keys:
                logger.info(f"  Majority Voting (set {size}): {baseline_final.get(key, 0.0):.4f}")
        else:
            logger.warning(f"Baseline 결과 파일 없음: {baseline_path}")
        
        # AggLLM 평가
        aggllm_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_aggllm_generated.json"
        )
        
        if os.path.exists(aggllm_path):
            logger.info(f"AggLLM 결과 로드: {aggllm_path}")
            with open(aggllm_path, 'r', encoding='utf-8') as f:
                aggllm_data = json.load(f)
            
            aggllm_metrics = defaultdict(list)
            
            for idx, problem_data in enumerate(aggllm_data["generated_solutions"]):
                solutions = problem_data["solutions"]
                ground_truth = problem_data["ground_truth"]
                
                # 각 문제마다 다른 시드 사용
                problem_results = evaluate_solutions(
                    solutions, ground_truth, math_verifier, seed=42 + idx
                )
                
                # Pass@k 메트릭 누적
                for k, v in problem_results.get("pass_at_k", {}).items():
                    aggllm_metrics[f"pass_at_{k}"].append(v)
                
                # Majority Voting 메트릭
                for set_size, stats in problem_results.get("majority_voting", {}).items():
                    aggllm_metrics[f"majority_voting_set_{set_size}"].append(
                        stats["accuracy"]
                    )
                
                # Confidence Weighted Voting 메트릭
                for metric, size_results in problem_results.get("confidence_weighted_voting", {}).items():
                    for set_size, stats in size_results.items():
                        aggllm_metrics[f"confidence_{metric}_set_{set_size}"].append(
                            stats["accuracy"]
                        )
            
            # 평균 계산
            aggllm_final = {}
            for key, values in aggllm_metrics.items():
                aggllm_final[key] = np.mean(values) if values else 0.0
            
            results["aggllm"] = aggllm_final
            
            logger.info(f"AggLLM 평가 완료: {len(aggllm_data['generated_solutions'])}개 문제")
            # Pass@k 결과 출력
            pass_k_keys = sorted([int(k.replace('pass_at_', '')) for k in aggllm_final.keys() if k.startswith('pass_at_')])
            for k in pass_k_keys:
                logger.info(f"  Pass@{k}: {aggllm_final[f'pass_at_{k}']:.4f}")
            majority_keys = sorted(
                [
                    (int(key.split('_')[-1]), key)
                    for key in aggllm_final.keys()
                    if key.startswith('majority_voting_set_')
                ]
            )
            for size, key in majority_keys:
                logger.info(f"  Majority Voting (set {size}): {aggllm_final.get(key, 0.0):.4f}")
        else:
            logger.warning(f"AggLLM 결과 파일 없음: {aggllm_path}")
        
        # 결과 저장
        metrics_path = os.path.join(results_dir, f"{dataset_safe_name}_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"메트릭 결과 저장: {metrics_path}")
        all_results[dataset_name] = results
    
    # 전체 요약
    logger.info("=" * 60)
    logger.info("전체 메트릭 평가 결과 요약")
    logger.info("=" * 60)
    
    for dataset_name, results in all_results.items():
        logger.info(f"\n{dataset_name}:")
        if "baseline" in results:
            logger.info("  Baseline:")
            baseline_k_values = sorted([int(k.replace('pass_at_', '')) for k in results['baseline'].keys() if k.startswith('pass_at_')])
            for k in baseline_k_values:
                logger.info(f"    Pass@{k}: {results['baseline'].get(f'pass_at_{k}', 0.0):.4f}")
            baseline_majority_keys = sorted(
                [
                    (int(key.split('_')[-1]), key)
                    for key in results['baseline'].keys()
                    if key.startswith('majority_voting_set_')
                ]
            )
            for size, key in baseline_majority_keys:
                logger.info(f"    Majority Voting (set {size}): {results['baseline'].get(key, 0.0):.4f}")
        
        if "aggllm" in results:
            logger.info("  AggLLM:")
            aggllm_k_values = sorted([int(k.replace('pass_at_', '')) for k in results['aggllm'].keys() if k.startswith('pass_at_')])
            for k in aggllm_k_values:
                logger.info(f"    Pass@{k}: {results['aggllm'].get(f'pass_at_{k}', 0.0):.4f}")
            aggllm_majority_keys = sorted(
                [
                    (int(key.split('_')[-1]), key)
                    for key in results['aggllm'].keys()
                    if key.startswith('majority_voting_set_')
                ]
            )
            for size, key in aggllm_majority_keys:
                logger.info(f"    Majority Voting (set {size}): {results['aggllm'].get(key, 0.0):.4f}")
    
    logger.info("\n✅ Stage 4-2: 메트릭 평가 완료")


if __name__ == "__main__":
    main()