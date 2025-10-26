"""
Stage 3: 모델 훈련 스크립트
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import torch

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.models.grpo_trainer import GRPOTrainer
from src.data.training_dataset import create_training_dataloader, create_validation_dataloader
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 3 메인 함수"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage3_train.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 3: 모델 훈련 시작")
    logger.info(f"설정: {cfg.training}")
    
    # 디렉토리 생성
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    
    # 입력 파일 경로 확인
    train_data_path = os.path.join(cfg.paths.data_dir, "curated", "train_curated.parquet")
    validation_data_path = os.path.join(cfg.paths.data_dir, "curated", "validation_curated.parquet")
    
    if not os.path.exists(train_data_path):
        logger.error(f"훈련 데이터 파일을 찾을 수 없습니다: {train_data_path}")
        logger.error("먼저 Stage 2를 실행해주세요: python scripts/stage2_curate.py")
        return
    
    # 데이터로더 생성
    logger.info("데이터로더 생성 중...")
    train_dataloader = create_training_dataloader(
        train_data_path,
        batch_size=cfg.training.training.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    validation_dataloader = None
    if os.path.exists(validation_data_path):
        validation_dataloader = create_validation_dataloader(
            validation_data_path,
            batch_size=cfg.training.training.batch_size,
            shuffle=False,
            num_workers=4
        )
        logger.info(f"검증 데이터로더 생성: {len(validation_dataloader)} 배치")
    
    logger.info(f"훈련 데이터로더 생성: {len(train_dataloader)} 배치")
    
    # GRPO 트레이너 초기화
    logger.info("GRPO 트레이너 초기화 중...")
    trainer = GRPOTrainer(
        model_name=cfg.model.base_model,
        lora_config=cfg.training.lora if cfg.training.method == "lora" else None,
        grpo_config=cfg.training.grpo,
        training_config=cfg.training.training,
        device=cfg.experiment.device
    )
    
    # 훈련 실행
    logger.info("모델 훈련 시작...")
    trainer.train(
        train_dataset=train_dataloader,
        validation_dataset=validation_dataloader,
        save_dir=cfg.paths.model_dir
    )
    
    logger.info("✅ Stage 3 완료")
    logger.info(f"훈련된 모델 저장 위치: {cfg.paths.model_dir}")


if __name__ == "__main__":
    main()

