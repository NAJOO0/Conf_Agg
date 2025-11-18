"""
Stage 4-2: 저장된 Solution 결과로 메트릭 평가
Pass@k, Majority Voting, Confidence Weighted Voting 계산
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


def calculate_pass_at_k(solutions: list, ground_truth: str, k_values: list, math_verifier: MathVerifier) -> dict:
    """
    Pass@k 메트릭 계산
    
    Pass@k 정의: solution을 k개씩 묶어서 set을 만들고, 각 set에 정답이 있는 비율 계산
    - 예: Pass@4이고 num_solutions가 32이면, 8개의 set을 만들어서 각 set에 정답이 있는 비율 측정
    """
    results = {}
    total_solutions = len(solutions)
    
    for k in k_values:
        # k가 실제 solution 개수보다 크면 건너뜀
        if k > total_solutions:
            continue
        
        # k개씩 묶어서 set 생성
        num_sets = total_solutions // k
        if num_sets == 0:
            continue
        
        correct_count = 0
        
        for i in range(num_sets):
            start_idx = i * k
            end_idx = start_idx + k
            set_solutions = solutions[start_idx:end_idx]
            
            # 각 set에 대해 정답이 있는지 확인
            has_correct = False
            for sol in set_solutions:
                if sol.get("final_answer") and math_verifier.verify_answer(sol["final_answer"], ground_truth):
                    has_correct = True
                    break
            
            if has_correct:
                correct_count += 1
        
        # 정답이 있는 set의 비율
        accuracy = correct_count / num_sets if num_sets > 0 else 0.0
        results[k] = accuracy
    
    return results


def majority_voting(solutions: list, samples_per_set: int, math_verifier: MathVerifier, ground_truth: str) -> dict:
    """
    Majority Voting 수행
    
    16개 solution을 samples_per_set개씩 나눠서 각 set에 대해 majority voting 수행
    """
    total_solutions = len(solutions)
    num_sets = total_solutions // samples_per_set
    correct_count = 0
    
    for i in range(num_sets):
        start_idx = i * samples_per_set
        end_idx = start_idx + samples_per_set
        set_solutions = solutions[start_idx:end_idx]
        
        # 각 sample의 final_answer 추출
        answers = [sol["final_answer"] for sol in set_solutions if sol.get("final_answer")]
        
        if not answers:
            continue
        
        # 가장 많이 나온 답안 선택
        answer_counts = defaultdict(int)
        for answer in answers:
            answer_counts[answer] += 1
        
        if answer_counts:
            majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
            if math_verifier.verify_answer(majority_answer, ground_truth):
                correct_count += 1
    
    return {
        "correct_count": correct_count,
        "total_sets": num_sets,
        "accuracy": correct_count / num_sets if num_sets > 0 else 0.0
    }


def confidence_weighted_voting(
    solutions: list, 
    samples_per_set: int, 
    confidence_metric: str,
    math_verifier: MathVerifier,
    ground_truth: str
) -> dict:
    """
    Confidence Weighted Voting 수행
    
    16개 solution을 samples_per_set개씩 나눠서 각 set에 대해 confidence weighted voting 수행
    """
    total_solutions = len(solutions)
    num_sets = total_solutions // samples_per_set
    correct_count = 0
    
    for i in range(num_sets):
        start_idx = i * samples_per_set
        end_idx = start_idx + samples_per_set
        set_solutions = solutions[start_idx:end_idx]
        
        # 각 sample의 final_answer와 confidence 추출
        weighted_votes = defaultdict(float)
        for sol in set_solutions:
            answer = sol.get("final_answer")
            if not answer:
                continue
            conf = sol.get("confidence_scores", {}).get(confidence_metric, 0.0)
            weighted_votes[answer] += conf
        
        if weighted_votes:
            best_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
            if math_verifier.verify_answer(best_answer, ground_truth):
                correct_count += 1
    
    return {
        "correct_count": correct_count,
        "total_sets": num_sets,
        "accuracy": correct_count / num_sets if num_sets > 0 else 0.0
    }


def get_adaptive_k_values(num_solutions: int) -> list:
    """
    Solution 개수에 맞게 k 값들을 동적으로 생성
    2, 4, 8, 16, 32 중에서 실제 solution 개수에 맞게 선택
    """
    # 기본 k 값 후보: 1, 2, 4, 8, 16, 32
    candidate_k_values = [1, 2, 4, 8, 16, 32]
    
    # 실제 solution 개수에 맞게 필터링
    k_values = [k for k in candidate_k_values if k <= num_solutions]
    
    # 중간 값들도 추가 (2~16 사이의 모든 값)
    if num_solutions >= 2:
        k_values.extend([k for k in range(2, min(17, num_solutions + 1)) if k not in k_values])
    
    # 정렬 및 중복 제거
    k_values = sorted(list(set(k_values)))
    
    return k_values


def get_adaptive_samples_per_set(num_solutions: int) -> list:
    """
    Solution 개수에 맞게 samples_per_set 값들을 동적으로 생성
    2, 4, 8, 16, 32 중에서 실제 solution 개수에 맞게 선택
    """
    # 기본 후보: 2, 4, 8, 16, 32
    candidate_values = [2, 4, 8, 16, 32]
    
    # 실제 solution 개수에 맞게 필터링
    samples_per_set = [v for v in candidate_values if v <= num_solutions]
    
    # 중간 값들도 추가 (2~16 사이의 모든 값)
    if num_solutions >= 2:
        samples_per_set.extend([v for v in range(2, min(17, num_solutions + 1)) if v not in samples_per_set])
    
    # 정렬 및 중복 제거
    samples_per_set = sorted(list(set(samples_per_set)))
    
    return samples_per_set


def evaluate_solutions(
    solutions: list,
    ground_truth: str,
    math_verifier: MathVerifier
) -> dict:
    """Solution 리스트에 대해 모든 메트릭 계산"""
    results = {}
    num_solutions = len(solutions)
    
    # Pass@k 계산 - 동적으로 k 값 생성
    k_values = get_adaptive_k_values(num_solutions)
    pass_at_k = calculate_pass_at_k(solutions, ground_truth, k_values, math_verifier)
    results["pass_at_k"] = pass_at_k
    
    # Majority Voting - 동적으로 samples_per_set 생성
    majority_results = {}
    samples_per_set_list = get_adaptive_samples_per_set(num_solutions)
    for samples_per_set in samples_per_set_list:
        voting_result = majority_voting(solutions, samples_per_set, math_verifier, ground_truth)
        majority_results[f"{samples_per_set}_samples"] = voting_result
    results["majority_voting"] = majority_results
    
    # Confidence Weighted Voting - 동적으로 samples_per_set 생성
    confidence_metrics = [
        "bottom_10_percent_confidence",
        "tail_confidence",
        "mean_group_confidence",
        "lowest_group_confidence"
    ]
    
    confidence_results = {}
    for metric in confidence_metrics:
        metric_results = {}
        for samples_per_set in samples_per_set_list:
            voting_result = confidence_weighted_voting(
                solutions,
                samples_per_set,
                metric,
                math_verifier,
                ground_truth
            )
            metric_results[f"{samples_per_set}_samples"] = voting_result
        confidence_results[metric] = metric_results
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
    
    # 디렉토리 설정
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    # results_dir = os.path.join(results_dir, "Qwen_Qwen3-1.7B_think_True")
    results_dir = os.path.join(results_dir, "think_prev_3600")
    
    # results_dir = os.path.join(results_dir, "think" if cfg.evaluation.benchmarks.evaluation.get("enable_thinking", False) else "no_think")
    # results_dir = os.path.join(results_dir, "think") 
    
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
            
            for problem_data in baseline_data["generated_solutions"]:
                solutions = problem_data["solutions"]
                ground_truth = problem_data["ground_truth"]
                
                problem_results = evaluate_solutions(solutions, ground_truth, math_verifier)
                
                # 메트릭 누적
                for k, v in problem_results.get("pass_at_k", {}).items():
                    baseline_metrics[f"pass_at_{k}"].append(v)
                
                for key, value in problem_results.get("majority_voting", {}).items():
                    baseline_metrics[f"majority_voting_{key}"].append(value["accuracy"])
                
                for metric, metric_results in problem_results.get("confidence_weighted_voting", {}).items():
                    for key, value in metric_results.items():
                        baseline_metrics[f"confidence_weighted_{metric}_{key}"].append(value["accuracy"])
            
            # 평균 계산
            baseline_final = {}
            for key, values in baseline_metrics.items():
                baseline_final[key] = np.mean(values) if values else 0.0
            
            results["baseline"] = baseline_final
            results["total_problems"] = len(baseline_data["generated_solutions"])
            logger.info(f"Baseline 평가 완료: {len(baseline_data['generated_solutions'])}개 문제")
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
            
            for problem_data in aggllm_data["generated_solutions"]:
                solutions = problem_data["solutions"]
                ground_truth = problem_data["ground_truth"]
                
                problem_results = evaluate_solutions(solutions, ground_truth, math_verifier)
                
                # 메트릭 누적
                for k, v in problem_results.get("pass_at_k", {}).items():
                    aggllm_metrics[f"pass_at_{k}"].append(v)
                
                for key, value in problem_results.get("majority_voting", {}).items():
                    aggllm_metrics[f"majority_voting_{key}"].append(value["accuracy"])
                
                for metric, metric_results in problem_results.get("confidence_weighted_voting", {}).items():
                    for key, value in metric_results.items():
                        aggllm_metrics[f"confidence_weighted_{metric}_{key}"].append(value["accuracy"])
            
            # 평균 계산
            aggllm_final = {}
            for key, values in aggllm_metrics.items():
                aggllm_final[key] = np.mean(values) if values else 0.0
            
            results["aggllm"] = aggllm_final
            logger.info(f"AggLLM 평가 완료: {len(aggllm_data['generated_solutions'])}개 문제")
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
            # 동적으로 k 값 추출 (실제 계산된 값들만)
            baseline_k_values = sorted([int(k.replace('pass_at_', '')) for k in results['baseline'].keys() if k.startswith('pass_at_')])
            for k in baseline_k_values:
                logger.info(f"    Pass@{k}: {results['baseline'].get(f'pass_at_{k}', 0.0):.3f}")
        if "aggllm" in results:
            logger.info("  AggLLM:")
            # 동적으로 k 값 추출 (실제 계산된 값들만)
            aggllm_k_values = sorted([int(k.replace('pass_at_', '')) for k in results['aggllm'].keys() if k.startswith('pass_at_')])
            for k in aggllm_k_values:
                logger.info(f"    Pass@{k}: {results['aggllm'].get(f'pass_at_{k}', 0.0):.3f}")
    
    logger.info("\n✅ Stage 4-2: 메트릭 평가 완료")


if __name__ == "__main__":
    main()


