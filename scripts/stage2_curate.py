"""
Stage 2: 데이터 큐레이션 스크립트
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.curation import DataCurator
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 2 메인 함수"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage2_curate.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 2: 데이터 큐레이션 시작")
    logger.info(f"설정: {cfg.data.curation}")
    
    # 디렉토리 생성
    os.makedirs(os.path.join(cfg.paths.data_dir, "curated"), exist_ok=True)
    
    # 입력 파일 경로
    generated_data_path = os.path.join(cfg.paths.data_dir, "generated", "raw_generated.parquet")
    
    if not os.path.exists(generated_data_path):
        logger.error(f"Stage 1 결과 파일을 찾을 수 없습니다: {generated_data_path}")
        logger.error("먼저 Stage 1을 실행해주세요: python scripts/stage1_generate.py")
        return
    
    # 데이터 큐레이션 실행
    curator = DataCurator(
        strategy=cfg.data.curation.strategy,
        easy_sample_percentage=cfg.data.curation.easy_sample_percentage,
        num_sets_per_problem=cfg.data.curation.num_sets_per_problem,
        set_size=cfg.data.curation.set_size,
        timeout=cfg.data.curation.verification.timeout
    )
    
    output_dir = os.path.join(cfg.paths.data_dir, "curated")
    train_path, validation_path = curator.curate_data(generated_data_path, output_dir)
    
    logger.info("✅ Stage 2 완료")
    logger.info(f"훈련 데이터: {train_path}")
    logger.info(f"검증 데이터: {validation_path}")


if __name__ == "__main__":
    main()

