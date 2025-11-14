"""
전체 파이프라인 실행 스크립트
"""
import os
import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import time

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run_stage(stage_script: str, stage_name: str) -> bool:
    """
    개별 스테이지를 실행합니다.
    
    Args:
        stage_script: 실행할 스크립트 경로
        stage_name: 스테이지 이름
    
    Returns:
        실행 성공 여부
    """
    logger.info(f"🚀 {stage_name} 시작")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, stage_script],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"✅ {stage_name} 완료 (소요 시간: {elapsed_time:.1f}초)")
            return True
        else:
            logger.error(f"❌ {stage_name} 실패")
            logger.error(f"오류 출력: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {stage_name} 실행 중 오류: {e}")
        return False


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """전체 파이프라인 메인 함수"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "full_pipeline.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🎯 Conf-AggLLM 전체 파이프라인 시작")
    logger.info("=" * 60)
    
    # 스테이지별 스크립트 경로
    stages = [
        ("scripts/stage1_generate.py", "Stage 1: 원시 데이터 생성"),
        ("scripts/stage2_curate.py", "Stage 2: 데이터 큐레이션"),
        ("scripts/stage3_train.py", "Stage 3: 모델 훈련"),
        ("scripts/stage4_evaluate.py", "Stage 4: 벤치마크 평가")
    ]
    
    # 각 스테이지 실행
    total_start_time = time.time()
    success_count = 0
    
    for stage_script, stage_name in stages:
        success = run_stage(stage_script, stage_name)
        if success:
            success_count += 1
        else:
            logger.error(f"파이프라인 중단: {stage_name}에서 실패")
            break
    
    # 전체 결과 요약
    total_elapsed_time = time.time() - total_start_time
    
    logger.info("=" * 60)
    logger.info("🎯 전체 파이프라인 완료")
    logger.info(f"성공한 스테이지: {success_count}/{len(stages)}")
    logger.info(f"총 소요 시간: {total_elapsed_time:.1f}초")
    
    if success_count == len(stages):
        logger.info("🎉 모든 스테이지가 성공적으로 완료되었습니다!")
        logger.info(f"결과 확인: {cfg.paths.output_dir}")
    else:
        logger.error("❌ 일부 스테이지에서 실패했습니다.")
        logger.error("로그 파일을 확인하여 문제를 해결해주세요.")


if __name__ == "__main__":
    main()

