"""
로깅 유틸리티 모듈
"""
import logging
import os
from typing import Optional
import wandb
from hydra.core.hydra_config import HydraConfig


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    wandb_enabled: bool = False,
    wandb_project: str = "conf-agg-llm",
    wandb_tags: Optional[list] = None
) -> logging.Logger:
    """
    로깅 시스템을 설정합니다.
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        log_file: 로그 파일 경로 (선택사항)
        wandb_enabled: WandB 로깅 활성화 여부
        wandb_project: WandB 프로젝트 이름
        wandb_tags: WandB 태그 리스트
    
    Returns:
        설정된 로거
    """
    # 로거 생성
    logger = logging.getLogger("conf_agg_llm")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택사항)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # WandB 초기화 (선택사항)
    if wandb_enabled and wandb.api.api_key:
        try:
            hydra_cfg = HydraConfig.get()
            run_name = f"{hydra_cfg.job.name}_{hydra_cfg.job.num}"
            
            wandb.init(
                project=wandb_project,
                name=run_name,
                tags=wandb_tags or [],
                config=hydra_cfg.cfg
            )
            logger.info(f"WandB 초기화 완료: {wandb_project}")
        except Exception as e:
            logger.warning(f"WandB 초기화 실패: {e}")
    
    logger.info(f"로깅 시스템 초기화 완료 (레벨: {log_level})")
    return logger


def get_logger(name: str = "conf_agg_llm") -> logging.Logger:
    """로거를 가져옵니다."""
    return logging.getLogger(name)

