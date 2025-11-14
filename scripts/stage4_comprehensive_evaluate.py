"""
Stage 4: 종합 벤치마크 평가 스크립트
Baseline과 AggLLM 모델의 다양한 aggregation 방법 비교
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import json

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.comprehensive_benchmark import ComprehensiveBenchmarkEvaluator
from src.data.confidence import ConfidenceCalculator
from src.evaluation.math_verifier import MathVerifier
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4 종합 평가 메인 함수"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage4_comprehensive_evaluate.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 4: 종합 벤치마크 평가 시작")
    
    # 디렉토리 생성
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 모델 경로 확인
    checkpoint_num = cfg.evaluation.benchmarks.evaluation.checkpoint_num
    if checkpoint_num is not None:
        aggllm_model_path = os.path.join(cfg.paths.model_dir, f"checkpoint-{checkpoint_num}")
    else:
        # 기본값: checkpoint-final
        aggllm_model_path = os.path.join(cfg.paths.model_dir, "checkpoint-final")
    
    if not os.path.exists(aggllm_model_path):
        logger.warning(f"AggLLM 모델을 찾을 수 없습니다: {aggllm_model_path}")
        logger.warning("Baseline 모델만 사용하여 평가합니다.")
        aggllm_model_path = None
    
    # 평가 구성 요소 초기화
    logger.info("평가 구성 요소 초기화 중...")
    
    # Confidence group_size 설정 (enable_thinking에 따라)
    eval_config = cfg.evaluation.benchmarks.evaluation
    enable_thinking = eval_config.get("enable_thinking", False)
    confidence_group_size = eval_config.get("confidence_group_size", 512)
    
    confidence_calculator = ConfidenceCalculator(
        group_size=confidence_group_size
    )
    
    logger.info(f"enable_thinking: {enable_thinking}, confidence_group_size: {confidence_group_size}")
    
    math_verifier = MathVerifier(
        timeout=cfg.evaluation.benchmarks.evaluation.timeout
    )
    
    # 벤치마크 평가기 초기화
    evaluator = ComprehensiveBenchmarkEvaluator(
        model_name=cfg.model.base_model,
        aggllm_model_path=aggllm_model_path,
        confidence_calculator=confidence_calculator,
        math_verifier=math_verifier,
        num_solutions=16,
        temperature=eval_config.temperature,
        max_tokens=eval_config.max_tokens,
        top_p=eval_config.get("top_p", 0.8),
        top_k=eval_config.get("top_k", 20),
        min_p=eval_config.get("min_p", 0.0),
        logprobs=eval_config.get("logprobs", 5),
        enable_thinking=enable_thinking,
        gpu_memory_utilization=eval_config.get("gpu_memory_utilization", 0.4),
        aggllm_gpu_memory_utilization=eval_config.get("aggllm_gpu_memory_utilization", None),
        max_model_len=eval_config.get("max_model_len", None)
    )
    
    # 벤치마크 데이터셋 설정
    benchmark_datasets = [
        {"name": "AIME24", "path": "math-ai/aime24"},
        {"name": "AIME25", "path": "math-ai/aime25"},
        {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
        {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
    ]
    
    # 각 벤치마크 평가
    all_results = {}
    
    for benchmark in benchmark_datasets:
        dataset_name = benchmark["name"]
        dataset_path = benchmark["path"]
        
        logger.info("=" * 50)
        logger.info(f"벤치마크 평가: {dataset_name}")
        logger.info("=" * 50)
        
        try:
            results = evaluator.evaluate_benchmark(
                dataset_name=dataset_path,
                output_dir=results_dir
            )
            all_results[dataset_name] = results
        except Exception as e:
            logger.error(f"{dataset_name} 평가 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 전체 결과 요약
    logger.info("=" * 50)
    logger.info("전체 벤치마크 평가 결과 요약")
    logger.info("=" * 50)
    
    # Baseline 결과 요약
    logger.info("\n📊 Baseline 모델 결과:")
    baseline_summary = {}
    for benchmark_name, results in all_results.items():
        if "baseline" in results:
            baseline_metrics = results["baseline"]
            logger.info(f"\n{benchmark_name}:")
            logger.info(f"  Pass@1: {baseline_metrics.get('pass_at_1', 0.0):.3f}")
            logger.info(f"  Pass@4: {baseline_metrics.get('pass_at_4', 0.0):.3f}")
            logger.info(f"  Pass@16: {baseline_metrics.get('pass_at_16', 0.0):.3f}")
            
            # Majority Voting
            logger.info("  Majority Voting:")
            for key in ["2_samples", "4_samples", "8_samples", "16_samples"]:
                acc = baseline_metrics.get(f"majority_voting_{key}", 0.0)
                logger.info(f"    {key}: {acc:.3f}")
            
            # Confidence Weighted Voting
            logger.info("  Confidence Weighted Voting:")
            for metric in ["tail_confidence", "mean_group_confidence", "bottom_10_percent_confidence", "lowest_group_confidence"]:
                logger.info(f"    {metric}:")
                for key in ["2_samples", "4_samples", "8_samples", "16_samples"]:
                    acc = baseline_metrics.get(f"confidence_weighted_{metric}_{key}", 0.0)
                    logger.info(f"      {key}: {acc:.3f}")
            
            # Prompt Aggregation
            logger.info(f"  Prompt Aggregation: {baseline_metrics.get('prompt_aggregation', 0.0):.3f}")
            
            baseline_summary[benchmark_name] = baseline_metrics
    
    # AggLLM 결과 요약
    logger.info("\n📊 AggLLM 모델 결과:")
    aggllm_summary = {}
    for benchmark_name, results in all_results.items():
        if "aggllm" in results:
            aggllm_metrics = results["aggllm"]
            logger.info(f"\n{benchmark_name}:")
            logger.info(f"  Pass@1: {aggllm_metrics.get('pass_at_1', 0.0):.3f}")
            logger.info(f"  Pass@4: {aggllm_metrics.get('pass_at_4', 0.0):.3f}")
            logger.info(f"  Pass@16: {aggllm_metrics.get('pass_at_16', 0.0):.3f}")
            
            # Majority Voting
            logger.info("  Majority Voting:")
            for key in ["2_samples", "4_samples", "8_samples", "16_samples"]:
                acc = aggllm_metrics.get(f"majority_voting_{key}", 0.0)
                logger.info(f"    {key}: {acc:.3f}")
            
            # Confidence Weighted Voting
            logger.info("  Confidence Weighted Voting:")
            for metric in ["tail_confidence", "mean_group_confidence", "bottom_10_percent_confidence", "lowest_group_confidence"]:
                logger.info(f"    {metric}:")
                for key in ["2_samples", "4_samples", "8_samples", "16_samples"]:
                    acc = aggllm_metrics.get(f"confidence_weighted_{metric}_{key}", 0.0)
                    logger.info(f"      {key}: {acc:.3f}")
            
            # Prompt Aggregation
            logger.info(f"  Prompt Aggregation: {aggllm_metrics.get('prompt_aggregation', 0.0):.3f}")
            
            aggllm_summary[benchmark_name] = aggllm_metrics
    
    # Aggregation 결과 요약
    logger.info("\n📊 Aggregation 결과:")
    for benchmark_name, results in all_results.items():
        logger.info(f"\n{benchmark_name}:")
        if "baseline_to_aggllm_aggregation" in results:
            agg_result = results["baseline_to_aggllm_aggregation"]
            logger.info(f"  Baseline → AggLLM: {agg_result.get('accuracy', 0.0):.3f}")
        if "aggllm_to_aggllm_aggregation" in results:
            agg_result = results["aggllm_to_aggllm_aggregation"]
            logger.info(f"  AggLLM → AggLLM: {agg_result.get('accuracy', 0.0):.3f}")
    
    # 전체 요약 저장
    summary = {
        "baseline": baseline_summary,
        "aggllm": aggllm_summary,
        "aggregation": {
            name: {
                "baseline_to_aggllm": results.get("baseline_to_aggllm_aggregation", {}),
                "aggllm_to_aggllm": results.get("aggllm_to_aggllm_aggregation", {})
            }
            for name, results in all_results.items()
        }
    }
    
    summary_path = os.path.join(results_dir, "comprehensive_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ Stage 4 종합 평가 완료")
    logger.info(f"결과 저장 위치: {results_dir}")
    logger.info(f"요약 저장 위치: {summary_path}")


if __name__ == "__main__":
    main()

