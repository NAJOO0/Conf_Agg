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
    
    Pass@k의 표준 정의: k개의 solution 중 하나라도 정답이면 1.0, 아니면 0.0
    - Pass@1~16: 각 k에 대해 처음 k개 solution 중 하나라도 정답인지 확인
    """
    results = {}
    
    for k in k_values:
        # 표준 Pass@k 정의: k개 중 하나라도 정답이면 통과
        is_correct = False
        for sol in solutions[:k]:
            if math_verifier.verify_answer(sol["final_answer"], ground_truth):
                is_correct = True
                break
        results[k] = 1.0 if is_correct else 0.0
    
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


def evaluate_solutions(
    solutions: list,
    ground_truth: str,
    math_verifier: MathVerifier
) -> dict:
    """Solution 리스트에 대해 모든 메트릭 계산"""
    results = {}
    
    # Pass@k 계산
    pass_at_k = calculate_pass_at_k(solutions, ground_truth, [1] + list(range(2, 16)) + [16], math_verifier)
    results["pass_at_k"] = pass_at_k
    
    # Majority Voting
    majority_results = {}
    for samples_per_set in list(range(2, 17)):
        voting_result = majority_voting(solutions, samples_per_set, math_verifier, ground_truth)
        majority_results[f"{samples_per_set}_samples"] = voting_result
    results["majority_voting"] = majority_results
    
    # Confidence Weighted Voting
    confidence_metrics = [
        "bottom_10_percent_confidence",
        "tail_confidence",
        "mean_group_confidence",
        "lowest_group_confidence"
    ]
    
    confidence_results = {}
    for metric in confidence_metrics:
        metric_results = {}
        for samples_per_set in list(range(2, 17)):
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
    
    # results_dir = os.path.join(results_dir, "think" if cfg.evaluation.benchmarks.evaluation.get("enable_thinking", False) else "no_think")
    results_dir = os.path.join(results_dir, "think") 
    
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
            for k in [1] + list(range(2, 17)):
                logger.info(f"    Pass@{k}: {results['baseline'].get(f'pass_at_{k}', 0.0):.3f}")
        if "aggllm" in results:
            logger.info("  AggLLM:")
            for k in [1] + list(range(2, 17)):
                logger.info(f"    Pass@{k}: {results['aggllm'].get(f'pass_at_{k}', 0.0):.3f}")
    
    logger.info("\n✅ Stage 4-2: 메트릭 평가 완료")


if __name__ == "__main__":
    main()


