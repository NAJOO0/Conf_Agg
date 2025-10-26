"""
Stage 4: 벤치마크 평가 스크립트
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

from src.evaluation.benchmark import BenchmarkEvaluator
from src.data.confidence import ConfidenceCalculator
from src.evaluation.math_verifier import MathVerifier
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4 메인 함수"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage4_evaluate.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 4: 벤치마크 평가 시작")
    logger.info(f"설정: {cfg.evaluation.benchmarks}")
    
    # 디렉토리 생성
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.output_dir, "results"), exist_ok=True)
    
    # 모델 경로 확인
    model_path = os.path.join(cfg.paths.model_dir, "checkpoint-final")
    if not os.path.exists(model_path):
        logger.error(f"훈련된 모델을 찾을 수 없습니다: {model_path}")
        logger.error("먼저 Stage 3을 실행해주세요: python scripts/stage3_train.py")
        return
    
    # 평가 구성 요소 초기화
    logger.info("평가 구성 요소 초기화 중...")
    
    confidence_calculator = ConfidenceCalculator(
        group_size=cfg.data.raw_dataset.confidence.group_size
    )
    
    math_verifier = MathVerifier(
        timeout=cfg.evaluation.benchmarks.evaluation.timeout
    )
    
    # 벤치마크 평가기 초기화
    evaluator = BenchmarkEvaluator(
        model_path=model_path,
        base_model_name=cfg.model.base_model,
        confidence_calculator=confidence_calculator,
        math_verifier=math_verifier,
        num_candidates=cfg.evaluation.benchmarks.evaluation.num_candidates,
        temperature=cfg.evaluation.benchmarks.evaluation.temperature,
        max_tokens=cfg.evaluation.benchmarks.evaluation.max_tokens
    )
    
    # 벤치마크 평가 실행
    logger.info("벤치마크 평가 실행 중...")
    
    benchmark_configs = cfg.evaluation.benchmarks.datasets
    output_dir = os.path.join(cfg.paths.output_dir, "results")
    
    summary_results = evaluator.evaluate_all_benchmarks(
        benchmark_configs=benchmark_configs,
        output_dir=output_dir
    )
    
    # 결과 출력
    logger.info("=" * 50)
    logger.info("벤치마크 평가 결과 요약")
    logger.info("=" * 50)
    
    for benchmark_name, results in summary_results["benchmarks"].items():
        logger.info(f"{benchmark_name}:")
        logger.info(f"  - Pass@1: {results['pass_at_1']:.3f}")
        logger.info(f"  - Pass@5: {results['pass_at_5']:.3f}")
        logger.info(f"  - Pass@10: {results['pass_at_10']:.3f}")
        logger.info(f"  - 신뢰도 상관관계: {results['confidence_correlation']:.3f}")
        logger.info(f"  - 총 문제 수: {results['total_problems']}")
        logger.info(f"  - 정답 수: {results['correct_predictions']}")
        logger.info("")
    
    logger.info("전체 평균:")
    logger.info(f"  - Pass@1: {summary_results['overall_pass_at_1']:.3f}")
    logger.info(f"  - Pass@5: {summary_results['overall_pass_at_5']:.3f}")
    logger.info(f"  - Pass@10: {summary_results['overall_pass_at_10']:.3f}")
    logger.info(f"  - 신뢰도 상관관계: {summary_results['overall_confidence_correlation']:.3f}")
    
    logger.info("✅ Stage 4 완료")
    logger.info(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()

